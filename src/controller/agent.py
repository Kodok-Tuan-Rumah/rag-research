from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

from src.controller.agent_state import (
    MAX_RETRIEVAL_RETRIES,
    AgentState,
    AgentStatus,
)
from src.controller.policy_engine import PolicyEngine, PlanHints
from src.controller.confidence_engine import ConfidenceEngine
from src.api.config import settings
from src.generation.generator import GroqGenerator, apply_partial_trust_policy
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.tools import RetrievalTools

logger = logging.getLogger("rag.agent")

@dataclass
class AgentResponse:
    answer: Optional[str]
    status: AgentStatus
    state: AgentState
    response: Any = None
    retrieved_ids: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

class Agent:
    def __init__(self, max_steps: int = 8):
        self.policy_engine = PolicyEngine()
        self.confidence_engine = ConfidenceEngine()
        self.generator = GroqGenerator(model=settings.groq_model or self._default_model())
        self.store = QdrantVectorStore()
        self.retriever = MasterHybridRetriever(vector_store=self.store)
        self.tools = RetrievalTools(
            vector_store=self.store,
            retriever=self.retriever,
            confidence_engine=self.confidence_engine,
            generator=self.generator,
        )
        self.max_steps = max_steps

    def _default_model(self) -> str:
        return "openai/gpt-oss-20b"
    
    # EXECUTION
    async def _execute_retrieval(self, state: AgentState, params: dict):
        result = await self.tools.call_tool("search_hybrid", params)
        state.add_observation("search_hybrid", result)

        candidates = result.get("chunks", [])
        ranked = sorted(
            candidates,
            key=lambda chunk: (
                chunk.get("retrieval_score", 0.0)
                if isinstance(chunk, dict)
                else getattr(chunk, "retrieval_score", 0.0)
            ),
            reverse=True,
        )[:20]
        seen = set()
        state.retrieved_chunks = []
        for chunk in ranked:
            chunk_id = (
                chunk.get("chunk_id")
                if isinstance(chunk, dict)
                else getattr(chunk, "chunk_id", None)
            )
            if chunk_id not in seen:
                seen.add(chunk_id)
                state.retrieved_chunks.append(chunk)

    async def _execute_generation(
        self,
        state: AgentState,
        plan: PlanHints,
        retrieval_decision: str | None = None,
        retrieval_confidence: float | None = None,
    ) -> Optional[AgentResponse]:
        generation_chunks = apply_partial_trust_policy(
            state.retrieved_chunks, "", retrieval_decision
        )[0]
        result = await self.tools.call_tool("generate_answer", {
            "query": state.query,
            "chunks": generation_chunks,
            "max_tokens": plan.generation_hint["max_tokens"],
            "temperature": plan.generation_hint["temperature"],
            "source_chunk_id": state.source_chunk_id,
            "min_top1_score": 0.0,
            "query_type": plan.query_type,
            "retrieval_decision": retrieval_decision,
            "retrieval_confidence": retrieval_confidence,
        })

        state.add_observation("generate_answer", result)
        response = result.get("response")
        answer = response.answer if response else None
        _, answer = apply_partial_trust_policy(
            state.retrieved_chunks, answer, retrieval_decision
        )

        if response and response.status == "ANSWERED":
            state.status = AgentStatus.ANSWERED
            return AgentResponse(
                answer=answer,
                status=state.status,
                state=state,
                response=response,
                retrieved_ids=[
                    c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                    for c in state.retrieved_chunks
                ],
                latency_ms={}
                )
        
        state.status = AgentStatus.ABSTAINED     
        return AgentResponse(
            answer=answer,
            status=state.status,
            state=state,
            response=response,
            retrieved_ids=[
                c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                for c in state.retrieved_chunks
            ],
        )

    # STRATEGY
    def _build_retrieval_params(self, state: AgentState, plan: PlanHints) -> dict:
        params = {
            "query": state.query,
            "k": plan.top_k_hint,
            "use_multi_query": plan.allow_multi_query,
            "use_hyde": plan.allow_hyde,
            "dense_weight": 0.5,
            "bm25_weight": 0.5,
            "min_score_threshold": settings.min_score_threshold,
        }

        if getattr(state, 'retry_attempted', False) or getattr(state, 'force_expand', False):
            # §3.4 specifies exactly three adjustments on the single permitted
            # retry: widen the candidate pool, shift weight toward BM25, and
            # relax the score floor. It does not enable query expansion.
            #
            # `use_multi_query` used to be forced on here, which contradicted
            # PROJECT_GOALS.md §9 ("Do not re-enable") and added an LLM call to
            # every retry. §9's stated reason for keeping it off was "+0.17 MRR
            # at +3400ms cost — not viable until latency budget allows", and the
            # P95 budget is already missed. The retry path now follows the
            # contract exactly; plan.allow_multi_query still governs the value,
            # and it defaults to False via settings.
            params["k"] = min(int(plan.top_k_hint * 1.5), 30)
            params["dense_weight"] = 0.3
            params["bm25_weight"] = 0.7
            params["min_score_threshold"] = -0.1
            state.force_expand = False
        return params
    
    def _decide_next_action(self, signals: dict, decision: str) -> str:
        if decision not in {"REJECT", "PARTIAL_TRUST", "GENERATE"}:
            raise ValueError(f"Unknown confidence decision: {decision}")
        if decision == "REJECT":
            return "REJECT"
        if signals.get("unstable_retrieval"):
            return "RETRIEVE_AGAIN"
        if signals.get("spurious_match"):
            return "PARTIAL_TRUST"
        return decision
    
    # MAIN LOOP
    async def run(self, query: str, source_chunk_id: str | None = None) -> AgentResponse:
        state = AgentState(
            query=query,
            source_chunk_id=source_chunk_id,
            MAX_STEPS=self.max_steps,
        )
        plan = self.policy_engine.initial_plan(query)
        response = None
        import time
        t_start = time.time()
        while state.status == AgentStatus.RUNNING and not state.is_budget_exhausted():
            params = self._build_retrieval_params(state, plan)

            # Retrieval
            await self._execute_retrieval(state, params)
            if state.is_budget_exhausted():
                state.status = AgentStatus.ABSTAINED
                break
            logger.debug(
                "[DEBUG] chunks going to confidence: %d", len(state.retrieved_chunks)
            )
            for c in state.retrieved_chunks[:3]:
                logger.debug(
                    "  chunk_id=%s retrieval_score=%s",
                    c.get("chunk_id"), c.get("retrieval_score"),
                )


            # Confidence
            quality = await self.tools.assess_retrieval_quality(
                state.retrieved_chunks,
                state.query,
                query_type=plan.query_type,
            )
            state.add_observation("assess_retrieval_quality", quality)

            confidence = quality["confidence"]
            state.confidence_score = confidence["confidence_score"]
            signals = confidence["signals"]
            decision = confidence["decision"]

            current_top_score = state.retrieved_chunks[0].get("retrieval_score", 0.0) if state.retrieved_chunks else 0.0
            if abs(current_top_score - state.prev_top_score) < 0.1:
                state.stagnation_count += 1
            else:
                state.stagnation_count = 0
            state.prev_top_score  = current_top_score

            if state.stagnation_count >= 2:
                state.status = AgentStatus.ABSTAINED
                break

            action = self._decide_next_action(signals, decision)
            if action == "REJECT":
                if state.retrieval_retries < MAX_RETRIEVAL_RETRIES:
                    logger.info("[RETRY_CONTRACT] Attempt 1 rejected (confidence=%.3f). Triggering Attempt 2 retry per Section 3.4.", state.confidence_score)
                    state.retrieval_retries += 1
                    state.force_expand = True
                    continue
                else:
                    logger.warning("[RETRY_CONTRACT] Attempt 2 still rejected. Abstaining.")
                    state.status = AgentStatus.ABSTAINED
                    break
            if action == "RETRIEVE_AGAIN":
                if state.is_budget_exhausted():
                    state.status = AgentStatus.ABSTAINED
                    break
                # §3.4 permits one retry in total, regardless of which signal
                # triggered it. Previously this branch only consulted the step
                # budget, so a persistently unstable signal kept looping until
                # the budget ran out — 3 attempts at the default max_steps=8,
                # and scaling linearly with max_steps when scores moved enough to
                # defeat stagnation detection.
                if state.retrieval_retries < MAX_RETRIEVAL_RETRIES:
                    logger.info(
                        "[RETRY_CONTRACT] Retrieval unstable on attempt 1. "
                        "Triggering the single retry permitted by Section 3.4."
                    )
                    state.retrieval_retries += 1
                    state.force_expand = True
                    continue
                logger.warning(
                    "[RETRY_CONTRACT] Retrieval still unstable after the retry. Abstaining."
                )
                state.status = AgentStatus.ABSTAINED
                break

            if action in ("PARTIAL_TRUST", "GENERATE"):
                if state.is_budget_exhausted():
                    state.status = AgentStatus.ABSTAINED
                    break
                # Generation can raise for provider-side reasons (network, quota,
                # output-format rejection). scripts/run_single_query.py and
                # src/api/main.py both already convert such errors into a
                # structured abstain; without the same guard here the orchestrator
                # was the one place an operational error escaped to the caller.
                # PROJECT_GOALS.md §4: no silent failures, and §2.3 treats
                # abstention as a valid terminal outcome.
                try:
                    response = await self._execute_generation(
                        state,
                        plan,
                        retrieval_decision=action,
                        retrieval_confidence=state.confidence_score,
                    )
                except Exception as exc:  # noqa: BLE001 - boundary, must not leak
                    logger.error(
                        "[GENERATION_ERROR] %s: %s — degrading to abstain",
                        type(exc).__name__, exc, exc_info=True,
                    )
                    state.status = AgentStatus.ABSTAINED
                    state.failure_reason = f"generation_exception: {type(exc).__name__}: {exc}"
                    break
                if response:
                    response.latency_ms = {"total": round((time.time() - t_start) * 1000, 1)}
                    return response
                
        # fallback
        if state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.ABSTAINED
        return AgentResponse(
            answer=response.answer if response else None,
            status=state.status,
            state=state,
            response=None,
            retrieved_ids=[
                c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                for c in state.retrieved_chunks
            ],
            latency_ms={"total": round((time.time() - t_start) * 1000, 1)},
        )
