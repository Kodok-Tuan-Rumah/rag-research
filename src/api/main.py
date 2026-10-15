from __future__ import annotations

import time
import logging
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.api.config import settings
from src.controller.policy_engine import PolicyEngine
from src.controller.agent_state import MAX_RETRIEVAL_RETRIES
from src.generation.generator import PARTIAL_TRUST_CHUNK_LIMIT, apply_partial_trust_policy
from typing import Optional

load_dotenv()

DEFAULT_USE_RERANKER = False
policy_engine = PolicyEngine()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("rag.api")

# Lazy globals 
store = None
retriever = None
generator = None
confidence_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever, generator, confidence_engine
    logger.info("[STARTUP] Initializing RAG Research Master Engine (PHASE 1 - BASELINE)...")

    try:
        from src.controller.confidence_engine import ConfidenceEngine
        from src.retrieval.qdrant_store import QdrantVectorStore
        from src.retrieval.hybrid_retriever import MasterHybridRetriever
        from src.generation.generator import GroqGenerator

        confidence_engine = ConfidenceEngine()
        store = QdrantVectorStore()
        retriever = MasterHybridRetriever(
            vector_store=store,
            bm25_chunks_path=settings.bm25_chunks_path,
            rrf_k=settings.hybrid_rrf_k,
            # use_multi_query / use_hyde are no longer constructor arguments.
            # They were overwritten unconditionally on every search from the
            # per-query plan, so passing them here had no effect and only
            # suggested the retriever was configurable globally. The expansion
            # flags now travel with the search call instead.
        )

        index_info = store.require_nonempty_index(
            expected_corpus_chunks=len(retriever.chunks),
            consumer="FastAPI startup",
        )
        logger.info("[STARTUP] Qdrant index: %s", json.dumps(index_info, sort_keys=True))

        generator = GroqGenerator(model=settings.groq_model)
        logger.info(
            "[STARTUP] Success — %d vectors available in collection '%s'",
            index_info["vector_count"], index_info["collection"],
        )
        logger.info(
            f"[STARTUP] CONFIG — multi_query={settings.use_multi_query} | "
            f"hyde={settings.use_hyde} | model={settings.groq_model}"
        )
    except Exception as e:
        logger.error(f"[STARTUP] Critical Error: {str(e)}")
        if store:
            store.close()
        raise e
    yield

    if store:
        logger.info("[SHUTDOWN] Closing Vector Store connections...")
        store.close()

app = FastAPI(
    title="RAG Research Master Engine",
    description="Production-ready RAG (PHASE 1 - Simplified Baseline)",
    version="1.0.0-baseline",
    lifespan=lifespan,
)

# Pydantic Models 
class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)
    ranking_mode: Optional[str] = None

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_reranker: bool = Field(default=DEFAULT_USE_RERANKER)
class GenerateResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[ChunkResult]
    retrieval_method: str
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_total_ms: float
    model: str
    confidence: float
    decision: str
    retrieval_confidence: float | None = None
    post_generation_confidence: float | None = None
    post_generation_decision: str | None = None
    retry_triggered: bool = False
    # Category of the failure, e.g. retrieval_miss / low_confidence /
    # insufficient_context / pipeline_error.
    failure_type: str = "none"
    # Specific cause behind failure_type, taken from RAGResponse.reason.
    # PROJECT_GOALS.md §4 requires structured failure observability; without this
    # a caller cannot distinguish a provider rejection from a model refusal or a
    # sanity-check rejection.
    abstain_reason: str | None = None
    generator_status: str = "UNKNOWN"
    generator_confidence: float | None = None
    llm_confidence: float | None = None
    generation_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)

# Helpers 
def _format_chunks(chunks: list, limit: int | None = None) -> list[ChunkResult]:
    target = chunks[:limit] if limit else chunks
    return [
        ChunkResult(
            chunk_id=c.get("chunk_id", "unk"),
            doc_id=c.get("doc_id", "unk"),
            text=c.get("text", ""),
            score=round(float(c.get("retrieval_score", 0)), 4),
            metadata=c.get("metadata", {}),
            ranking_mode=c.get("ranking_mode"),
        )
        for c in target
    ]


def _chunk_ids(chunks: list) -> list[str]:
    result = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else getattr(chunk, "chunk_id", None)
        if chunk_id is not None:
            result.append(str(chunk_id))
    return result


def _chunk_scores(chunks: list, key: str) -> list:
    """Score series for the §4 log schema, parallel to `chunks`.

    Missing scores stay an explicit `None` so a reader can tell "not reranked"
    apart from "scored zero" instead of seeing a fabricated 0.0.
    """
    scores = []
    for chunk in chunks:
        value = chunk.get(key) if isinstance(chunk, dict) else getattr(chunk, key, None)
        scores.append(round(float(value), 4) if value is not None else None)
    return scores


def _log_query_event(**event) -> None:
    """Emit one machine-readable event for every completed API query."""
    logger.info(json.dumps(event, ensure_ascii=False, default=str))


def _abstain_response(
    *,
    query: str,
    chunks: list,
    retrieval_method: str,
    latency_retrieval_ms: float,
    latency_total_ms: float,
    confidence: float,
    retry_triggered: bool,
    failure_type: str,
    abstain_reason: str | None = None,
) -> GenerateResponse:
    return GenerateResponse(
        query=query,
        answer="The available information is not sufficient to provide a reliable answer.",
        retrieved_chunks=_format_chunks(chunks, limit=2),
        retrieval_method=retrieval_method,
        latency_retrieval_ms=latency_retrieval_ms,
        latency_generation_ms=0.0,
        latency_total_ms=latency_total_ms,
        model="none",
        confidence=confidence,
        decision="REJECT",
        retrieval_confidence=confidence,
        retry_triggered=retry_triggered,
        failure_type=failure_type,
        abstain_reason=abstain_reason,
        generator_status="INSUFFICIENT_CONTEXT",
    )

# Endpoints
@app.get("/health")
async def health():
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")
    count = store.client.count(store.collection_name).count
    # Health polling must not load the optional CrossEncoder as a side effect.
    reranker_model_loaded = getattr(retriever, "reranker_loaded", False)
    return {
        "status": "ready",
        "vectors": count,
        "mode": "PHASE1_BASELINE",
        "multi_query": settings.use_multi_query,
        "hyde": settings.use_hyde,
        "reranker_model_loaded": reranker_model_loaded,
        "reranker_default_enabled": DEFAULT_USE_RERANKER,
        "reranker": reranker_model_loaded,  # deprecated: gunakan reranker_model_loaded
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    if not all([retriever, generator, confidence_engine]):
        raise HTTPException(status_code=503, detail="Engine not ready")
    rid = str(uuid.uuid4())[:8]
    t_api_start = time.time()
    retry_triggered = False
    rerank_ms = 0.0
    query_type = "unknown"
    top_k = req.top_k

    try:
        plan = policy_engine.initial_plan(req.query)
        query_type = plan.query_type
        top_k = req.top_k if req.top_k is not None else plan.top_k_hint
        # Shared with Agent: initial retrieval plus the configured retry limit.
        t_retrieval_start = time.time()
        chunks = []
        conf_eval = {"confidence_score": 0.0, "decision": "REJECT", "signals": {}}
        for attempt in range(MAX_RETRIEVAL_RETRIES + 1):
            search_kwargs = {
                "query": req.query,
                "top_k": top_k,
                "rerank": req.use_reranker,
                "allow_references": plan.allow_references,
                "max_ref_ratio": plan.max_ref_ratio,
                "use_multi_query": plan.allow_multi_query,
                "use_hyde": plan.allow_hyde,
            }
            if attempt > 0:
                retry_triggered = True
                search_kwargs.update({"dense_weight": 0.3, "bm25_weight": 0.7})

            # search_with_trace is the same canonical retrieval as search() but
            # also returns the per-stage trace, which is the only way this
            # adapter can report latency_ms.reranker without re-timing a stage
            # that runs inside the retriever.
            search_result = await run_in_threadpool(
                retriever.search_with_trace, **search_kwargs
            )
            chunks = search_result["results"]
            rerank_ms = (
                search_result.get("trace", {})
                .get("latency_ms", {})
                .get("reranker", 0.0)
            )
            conf_eval = confidence_engine.calculate_confidence(
                chunks,
                query_type=query_type,
                query=req.query,
            )
            if chunks and conf_eval["decision"] != "REJECT":
                break

        latency_retrieval_ms = round((time.time() - t_retrieval_start) * 1000, 2)
        confidence = conf_eval["confidence_score"]
        decision   = conf_eval["decision"]

        logger.info(
            f"[{rid}] RETRIEVAL DONE | "
            f"chunks={len(chunks)} | confidence={confidence} | decision={decision} | "
            f"retrieval_ms={latency_retrieval_ms}"
        )

        # NO RESULTS 
        if not chunks:
            logger.warning(f"[{rid}] NO_RESULTS | query='{req.query[:60]}'")
            result = _abstain_response(
                query=req.query, chunks=[], retrieval_method="none",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                confidence=0.0, retry_triggered=retry_triggered,
                failure_type="retrieval_miss",
                abstain_reason="retrieval returned no usable chunks",
            )
            _log_query_event(
                query_id=rid, query=req.query,
                query_type=query_type,
                answer_status="abstained",
                failure_type=result.failure_type,
                abstain_reason=result.abstain_reason,
                confidence_score=0.0,
                retrieval_scores=[], reranker_scores=[], chunks_used=[],
                retry_triggered=retry_triggered,
                latency_ms={"retrieval": latency_retrieval_ms, "reranker": rerank_ms, "generation": 0.0, "total": result.latency_total_ms},
            )
            return result

        # REJECT
        if decision == "REJECT":
            logger.warning(
                f"[{rid}] REJECT | query='{req.query[:60]}' | "
                f"confidence={confidence} | signals={conf_eval['signals']}"
            )
            result = _abstain_response(
                query=req.query, chunks=chunks, retrieval_method="rejected_low_confidence",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                confidence=confidence, retry_triggered=retry_triggered,
                failure_type="low_confidence",
                abstain_reason=f"confidence {confidence:.3f} below reject threshold",
            )
            _log_query_event(
                query_id=rid, query=req.query,
                query_type=query_type,
                answer_status="abstained",
                failure_type=result.failure_type,
                abstain_reason=result.abstain_reason,
                confidence_score=confidence,
                retrieval_scores=_chunk_scores(chunks, "retrieval_score"),
                reranker_scores=_chunk_scores(chunks, "rerank_score"),
                chunks_used=[],
                retry_triggered=retry_triggered,
                latency_ms={"retrieval": latency_retrieval_ms, "reranker": rerank_ms, "generation": 0.0, "total": result.latency_total_ms},
            )
            return result
        
        if decision == "PARTIAL_TRUST":
            logger.warning(
                f"[{rid}] PARTIAL_TRUST | query='{req.query[:60]}' | "
                f"confidence={confidence} | generating with top-3 chunks only"
            )
            t_gen_start = time.time()
            response = await run_in_threadpool(
                generator.generate,
                req.query,
                chunks[:PARTIAL_TRUST_CHUNK_LIMIT],  
                query_type=query_type,
                max_tokens=int(plan.generation_hint["max_tokens"]),
                temperature=plan.generation_hint["temperature"],
                retrieval_decision=decision,
                retrieval_confidence=confidence,
            )
            latency_generation_ms = round((time.time() - t_gen_start) * 1000, 2)

            # One implementation of the PARTIAL_TRUST rule, shared with the
            # Agent and the generator. The inline `chunks[:3]` plus
            # f"[Low confidence...] {response.answer}" that used to be here
            # prefixed abstentions too, so a generator abstain inside this
            # branch was returned to the client as the literal string
            # "[Low confidence — verify independently] INSUFFICIENT_CONTEXT".
            presented_chunks, presented_answer = apply_partial_trust_policy(
                chunks, getattr(response, "answer", ""), decision
            )

            result = GenerateResponse(
                query=req.query,
                answer=presented_answer,
                retrieved_chunks=_format_chunks(presented_chunks, limit=PARTIAL_TRUST_CHUNK_LIMIT),
                retrieval_method=response.retrieval_method,
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=latency_generation_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                model=response.model,
                confidence=confidence,
                decision=decision,
                retrieval_confidence=confidence,
                post_generation_confidence=getattr(
                    response,
                    "post_generation_confidence_score",
                    getattr(response, "confidence_score", None),
                ),
                post_generation_decision=getattr(response, "post_generation_decision", None),
                retry_triggered=retry_triggered,
                generator_status=getattr(response, "status", "UNKNOWN"),
                generator_confidence=getattr(
                    response,
                    "post_generation_confidence_score",
                    getattr(response, "confidence_score", None),
                ),
                llm_confidence=getattr(response, "llm_confidence_score", None),
                generation_chunk_ids=_chunk_ids(getattr(response, "retrieved_chunks", [])),
                context_chunk_ids=getattr(response, "context_chunk_ids", []),
                failure_type=(
                    "insufficient_context"
                    if getattr(response, "status", "") == "INSUFFICIENT_CONTEXT"
                    else "none"
                ),
                abstain_reason=(
                    getattr(response, "reason", None)
                    if getattr(response, "status", "") == "INSUFFICIENT_CONTEXT"
                    else None
                ),
            )
            generation_abstained = result.generator_status == "INSUFFICIENT_CONTEXT"
            used_chunks = chunks[:PARTIAL_TRUST_CHUNK_LIMIT]
            _log_query_event(
                query_id=rid, query=req.query,
                query_type=query_type,
                answer_status="abstained" if generation_abstained else "answered",
                failure_type="insufficient_context" if generation_abstained else "none",
                abstain_reason=getattr(response, "reason", None) if generation_abstained else None,
                confidence_score=confidence,
                retrieval_scores=_chunk_scores(used_chunks, "retrieval_score"),
                reranker_scores=_chunk_scores(used_chunks, "rerank_score"),
                chunks_used=_chunk_ids(used_chunks),
                generator_status=result.generator_status,
                generator_confidence=result.generator_confidence,
                llm_confidence=result.llm_confidence,
                generation_chunk_ids=result.generation_chunk_ids,
                context_chunk_ids=result.context_chunk_ids,
                retry_triggered=retry_triggered,
                latency_ms={"retrieval": latency_retrieval_ms, "reranker": rerank_ms, "generation": latency_generation_ms, "total": result.latency_total_ms},
            )
            return result

        # GENERATE 
        t_gen_start = time.time()
        response = await run_in_threadpool(
            generator.generate,
            req.query,
            chunks,
            query_type=query_type,
            max_tokens=int(plan.generation_hint["max_tokens"]),
            temperature=plan.generation_hint["temperature"],
            retrieval_decision=decision,
            retrieval_confidence=confidence,
        )
        latency_generation_ms = round((time.time() - t_gen_start) * 1000, 2)
        latency_total_ms = round((time.time() - t_api_start) * 1000, 2)
        logger.info(
            f"[{rid} COMPLETE]"
            f"retrieval_ms={latency_retrieval_ms}|"
            f"generation_ms={latency_generation_ms}|"
        )
        result = GenerateResponse(
            query=req.query,
            answer=response.answer,
            retrieved_chunks=_format_chunks(chunks),
            retrieval_method=response.retrieval_method,
            latency_retrieval_ms=latency_retrieval_ms,
            latency_generation_ms=latency_generation_ms,
            latency_total_ms=latency_total_ms,
            model=response.model,
            confidence=confidence,
            decision=decision,
            retrieval_confidence=confidence,
            post_generation_confidence=getattr(
                response,
                "post_generation_confidence_score",
                getattr(response, "confidence_score", None),
            ),
            post_generation_decision=getattr(response, "post_generation_decision", None),
            retry_triggered=retry_triggered,
            generator_status=getattr(response, "status", "UNKNOWN"),
            generator_confidence=getattr(
                response,
                "post_generation_confidence_score",
                getattr(response, "confidence_score", None),
            ),
            llm_confidence=getattr(response, "llm_confidence_score", None),
            generation_chunk_ids=_chunk_ids(getattr(response, "retrieved_chunks", [])),
            context_chunk_ids=getattr(response, "context_chunk_ids", []),
            # Retrieval succeeded but the generator declined to answer. Without
            # this the response reported failure_type="none" for a query that
            # produced no answer, which PROJECT_GOALS.md §4 forbids.
            failure_type=(
                "insufficient_context"
                if getattr(response, "status", "") == "INSUFFICIENT_CONTEXT"
                else "none"
            ),
            abstain_reason=(
                getattr(response, "reason", None)
                if getattr(response, "status", "") == "INSUFFICIENT_CONTEXT"
                else None
            ),
        )
        generation_abstained = result.generator_status == "INSUFFICIENT_CONTEXT"
        _log_query_event(
            query_id=rid, query=req.query,
            query_type=query_type,
            answer_status="abstained" if generation_abstained else "answered",
            failure_type="insufficient_context" if generation_abstained else "none",
            abstain_reason=getattr(response, "reason", None) if generation_abstained else None,
            confidence_score=confidence,
            retrieval_scores=_chunk_scores(chunks, "retrieval_score"),
            reranker_scores=_chunk_scores(chunks, "rerank_score"),
            chunks_used=_chunk_ids(chunks),
            generator_status=result.generator_status,
            generator_confidence=result.generator_confidence,
            llm_confidence=result.llm_confidence,
            generation_chunk_ids=result.generation_chunk_ids,
            context_chunk_ids=result.context_chunk_ids,
            retry_triggered=retry_triggered,
            latency_ms={"retrieval": latency_retrieval_ms, "reranker": rerank_ms, "generation": latency_generation_ms, "total": latency_total_ms},
        )
        return result
    
    except Exception as e:
        logger.error(f"[{rid}] CRITICAL ERROR | {str(e)}", exc_info=True)
        result = _abstain_response(
            query=req.query, chunks=[], retrieval_method="failed_safe_abstain",
            latency_retrieval_ms=0.0,
            latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
            confidence=0.0, retry_triggered=retry_triggered,
            failure_type="pipeline_error",
            abstain_reason=f"{type(e).__name__}: {e}"[:300],
        )
        _log_query_event(
            query_id=rid, query=req.query,
            query_type=query_type,
            answer_status="abstained",
            failure_type=result.failure_type,
            abstain_reason=result.abstain_reason,
            confidence_score=0.0,
            retrieval_scores=[], reranker_scores=[], chunks_used=[],
            retry_triggered=retry_triggered,
            latency_ms={"retrieval": 0.0, "reranker": 0.0, "generation": 0.0, "total": result.latency_total_ms},
        )
        return result


@app.get("/metrics")
async def metrics():
    if not store:
        raise HTTPException(status_code=503, detail="Service not ready")
    count = store.client.count(store.collection_name).count
    return {
        "vectors_indexed": count,
        "engine_version": "1.0.0-baseline",
        "phase": "PHASE_1",
        "retrieval_stack": ["Qdrant", "BM25"],
        "reranker": settings.reranker_model,
        "multi_query": settings.use_multi_query,
        "hyde": settings.use_hyde,
        "governor": "ConfidenceEngine_v1",
        "llm_provider": "Groq",
        "llm_model": settings.groq_model,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=False)
