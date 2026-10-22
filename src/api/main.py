from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv
from src.api.config import settings
from src.api.cache import cache
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

# The canonical processed corpus is an immutable startup input. Rebuilding it
# from data/raw belongs to an explicit offline ingestion job, not to API
# startup: the runtime image has no raw PDFs/PyMuPDF and the chunker has a
# non-identical pure-Python fallback. Keep the count and fingerprint explicit so
# a corpus change is a deliberate version bump, not an accidental auto-seed.
CANONICAL_CORPUS_CHUNKS = 796
CANONICAL_CORPUS_SHA256 = (
    "4375a26668366674473a8917a3d0bc4ce7a2e53ace913457b80ff09be3481ba6"
)


def _resolve_corpus_path(corpus_path: str | Path) -> Path:
    path = Path(corpus_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _load_canonical_corpus(
    corpus_path: str | Path = settings.bm25_chunks_path,
    *,
    expected_count: int | None = CANONICAL_CORPUS_CHUNKS,
    expected_sha256: str | None = CANONICAL_CORPUS_SHA256,
) -> list[dict]:
    """Load and validate the processed corpus that startup may index.

    This deliberately refuses to fall back to ``document_loader`` or
    ``semantic_chunker``. An empty index may be seeded only from this exact
    artifact; a missing or changed corpus is a deployment error.
    """
    path = _resolve_corpus_path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical corpus not found at {path}. API startup will not run "
            "the ingestion pipeline; restore the processed corpus before starting."
        )

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Canonical corpus fingerprint mismatch at {path}: expected "
            f"{expected_sha256}, got {actual_sha256}. Refusing to auto-seed an "
            "unexpected corpus."
        )

    chunks = json.loads(raw.decode("utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Canonical corpus at {path} must be a non-empty JSON list.")
    if expected_count is not None and len(chunks) != expected_count:
        raise ValueError(
            f"Canonical corpus at {path} has {len(chunks)} chunks; "
            f"expected {expected_count}."
        )

    required_fields = {"chunk_id", "doc_id", "text"}
    malformed = [
        index for index, chunk in enumerate(chunks)
        if not isinstance(chunk, dict) or not required_fields.issubset(chunk)
    ]
    if malformed:
        raise ValueError(
            f"Canonical corpus at {path} has malformed chunks at indexes "
            f"{malformed[:5]}."
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"Canonical corpus at {path} contains duplicate chunk_id values.")

    return chunks


def _seed_canonical_index(
    store,
    canonical_chunks: list[dict],
    *,
    expected_count: int,
    consumer: str = "FastAPI startup",
) -> dict:
    logger.warning(
        "[STARTUP] Qdrant index is empty; seeding %d canonical chunks into '%s' "
        "without running ingestion.",
        expected_count,
        store.collection_name,
    )
    # reset=False is intentional. Startup may fill an empty collection, but it
    # must never delete a non-empty collection that might contain another corpus.
    store.index_chunks(canonical_chunks, reset=False)
    return store.require_nonempty_index(
        expected_corpus_chunks=expected_count,
        consumer=f"{consumer} (after canonical seed)",
    )


def _ensure_startup_index(
    store,
    canonical_chunks: list[dict],
    *,
    expected_count: int,
    consumer: str = "FastAPI startup",
) -> dict:
    """Require the canonical index, seeding only when the collection is empty."""
    from src.retrieval.qdrant_store import IndexEmptyError

    try:
        return store.require_nonempty_index(
            expected_corpus_chunks=expected_count,
            consumer=consumer,
        )
    except IndexEmptyError:
        return _seed_canonical_index(
            store,
            canonical_chunks,
            expected_count=expected_count,
            consumer=consumer,
        )

# Prometheus Metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total count of RAG generation requests",
    ["backend", "decision", "failure_type"]
)
RAG_LATENCY_SECONDS = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG latency in seconds",
    ["backend", "decision"],
    buckets=(0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, float("inf"))
)
RAG_CONFIDENCE_SCORE = Histogram(
    "rag_confidence_score",
    "Confidence score of RAG results",
    ["backend", "decision"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
RAG_ABSTAIN_TOTAL = Counter(
    "rag_abstain_total",
    "Total count of RAG abstentions",
    ["backend", "failure_type"]
)

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

        # Validate the canonical processed corpus before constructing the
        # retriever. MasterHybridRetriever silently disables BM25 when the file
        # is missing, so a deferred check could start a degraded service.
        canonical_chunks = _load_canonical_corpus(settings.bm25_chunks_path)

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

        index_info = _ensure_startup_index(
            store,
            canonical_chunks,
            expected_count=len(canonical_chunks),
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
    description=(
        "Single-turn research RAG QA. The frozen Phase 1 acceptance set passes; "
        "confidence remains heuristic and is not calibrated."
    ),
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
    # Complete ordered retrieval IDs, including abstentions whose user-facing
    # `retrieved_chunks` list is intentionally shortened. This keeps evaluation
    # metrics tied to the same retrieval call without a second search.
    retrieval_chunk_ids: list[str] = Field(default_factory=list)
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
        retrieval_chunk_ids=_chunk_ids(chunks),
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

def _record_metrics(decision: str, failure_type: str, latency_ms: float, confidence: float):
    backend = "research_rag"
    RAG_REQUESTS_TOTAL.labels(backend=backend, decision=decision, failure_type=failure_type).inc()
    RAG_LATENCY_SECONDS.labels(backend=backend, decision=decision).observe(latency_ms / 1000.0)
    RAG_CONFIDENCE_SCORE.labels(backend=backend, decision=decision).observe(confidence)
    if decision == "REJECT" or failure_type != "none":
        RAG_ABSTAIN_TOTAL.labels(backend=backend, failure_type=failure_type).inc()

@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    if not all([retriever, generator, confidence_engine]):
        raise HTTPException(status_code=503, detail="Engine not ready")
    rid = str(uuid.uuid4())[:8]
    t_api_start = time.time()

    cache_key = f"rag:gen:{req.query}:{req.top_k}:{req.use_reranker}"
    cached_data = cache.get(cache_key)
    if cached_data:
        try:
            cached_result = GenerateResponse(**cached_data)
        except Exception as e:
            logger.warning(
                f"[{rid}] CACHE INVALID for query: {req.query} | {type(e).__name__}: {e}"
            )
            cache.delete(cache_key)
        else:
            cached_abstained = (
                cached_result.decision == "REJECT"
                or cached_result.failure_type != "none"
                or cached_result.generator_status == "INSUFFICIENT_CONTEXT"
            )
            if cached_abstained:
                # Older entries could contain an abstention. Do not serve it
                # from cache: an abstention must be re-evaluated against the
                # current corpus/provider state.
                logger.info(f"[{rid}] CACHE HIT ignored abstention for query: {req.query}")
                cache.delete(cache_key)
            else:
                cache_latency_ms = round((time.time() - t_api_start) * 1000, 2)
                cached_result = cached_result.model_copy(
                    update={
                        "latency_retrieval_ms": 0.0,
                        "latency_generation_ms": 0.0,
                        "latency_total_ms": cache_latency_ms,
                    }
                )
                logger.info(f"[{rid}] CACHE HIT for query: {req.query}")
                _record_metrics(
                    decision=cached_result.decision,
                    failure_type=cached_result.failure_type,
                    latency_ms=cache_latency_ms,
                    confidence=cached_result.confidence,
                )
                _log_query_event(
                    query_id=rid,
                    query=req.query,
                    query_type="cache_hit",
                    answer_status="answered",
                    failure_type=cached_result.failure_type,
                    abstain_reason=cached_result.abstain_reason,
                    confidence_score=cached_result.confidence,
                    retrieval_scores=[],
                    reranker_scores=[],
                    chunks_used=(
                        cached_result.context_chunk_ids
                        or cached_result.generation_chunk_ids
                    ),
                    generator_status=cached_result.generator_status,
                    generator_confidence=cached_result.generator_confidence,
                    llm_confidence=cached_result.llm_confidence,
                    generation_chunk_ids=cached_result.generation_chunk_ids,
                    context_chunk_ids=cached_result.context_chunk_ids,
                    retry_triggered=False,
                    latency_ms={
                        "retrieval": 0.0,
                        "reranker": 0.0,
                        "generation": 0.0,
                        "total": cache_latency_ms,
                    },
                    cache_hit=True,
                )
                return cached_result

    retry_triggered = False
    rerank_ms = 0.0
    query_type = "unknown"
    top_k = req.top_k

    try:
        plan = policy_engine.initial_plan(req.query)
        query_type = plan.query_type
        top_k = req.top_k if req.top_k is not None else plan.top_k_hint
        # Keep retry behavior aligned with Agent: one retry may widen top_k,
        # shift ranking weight toward BM25, and retry unstable retrieval.
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
                search_kwargs.update({
                    "top_k": min(max(top_k + 1, int(top_k * 1.5)), 30),
                    "dense_weight": 0.3,
                    "bm25_weight": 0.7,
                    "min_score_threshold": -0.1,
                })

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
            signals = conf_eval.get("signals", {})
            unstable_retrieval = bool(signals.get("unstable_retrieval"))
            if (
                chunks
                and conf_eval["decision"] != "REJECT"
                and not unstable_retrieval
            ):
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
            _record_metrics(decision="REJECT", failure_type=result.failure_type, latency_ms=result.latency_total_ms, confidence=0.0)
            return result

        # Agent.run abstains when the one permitted retry still reports an
        # unstable retrieval signal. Keep the API adapter on the same contract
        # instead of generating from a pool it has already classified unstable.
        if conf_eval.get("signals", {}).get("unstable_retrieval"):
            result = _abstain_response(
                query=req.query, chunks=chunks,
                retrieval_method="rejected_unstable_retrieval",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                confidence=confidence, retry_triggered=retry_triggered,
                failure_type="low_confidence",
                abstain_reason="retrieval remained unstable after the permitted retry",
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
                latency_ms={"retrieval": latency_retrieval_ms, "reranker": rerank_ms,
                            "generation": 0.0, "total": result.latency_total_ms},
            )
            _record_metrics(
                decision="REJECT", failure_type=result.failure_type,
                latency_ms=result.latency_total_ms, confidence=confidence,
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
            _record_metrics(decision="REJECT", failure_type=result.failure_type, latency_ms=result.latency_total_ms, confidence=confidence)
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
                retrieval_chunk_ids=_chunk_ids(chunks),
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
            _record_metrics(decision=decision, failure_type=result.failure_type, latency_ms=result.latency_total_ms, confidence=confidence)
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
            retrieval_chunk_ids=_chunk_ids(chunks),
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
        # Only cache successful answers. Caching an abstention would keep
        # serving a transient provider/retrieval failure for the whole TTL.
        if (
            decision in {"GENERATE", "PARTIAL_TRUST"}
            and not generation_abstained
            and result.failure_type == "none"
        ):
            cache.set(cache_key, result.model_dump(), ttl_seconds=600)
        _record_metrics(decision=decision, failure_type=result.failure_type, latency_ms=result.latency_total_ms, confidence=confidence)
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
        _record_metrics(decision="REJECT", failure_type=result.failure_type, latency_ms=result.latency_total_ms, confidence=0.0)
        return result


@app.get("/metrics")
async def metrics():
    if not store:
        raise HTTPException(status_code=503, detail="Service not ready")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=False)
