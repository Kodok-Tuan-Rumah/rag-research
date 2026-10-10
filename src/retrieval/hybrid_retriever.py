from __future__ import annotations

import logging
import json
import os
import numpy as np
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.retrieval.qdrant_store import QdrantVectorStore, RetrievalResult
from src.retrieval.chunk_filters import (
    is_reference_chunk as _cf_is_reference_chunk,
    is_title_or_author_chunk as _cf_is_title_or_author_chunk,
    is_low_quality_chunk as _cf_is_low_quality_chunk,
    passes_strict_filter as _cf_passes_strict_filter,
    passes_minimum_filter as _cf_passes_minimum_filter,
    filter_breakdown as _cf_filter_breakdown,
)
from src.controller.policy_engine import PolicyEngine
from src.api.config import settings
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger("rag.retriever")
load_dotenv()

def rrf_fuse(
    all_hits: List[List],
    k: int = 60,
    weights: Optional[List[float]] = None,
    section_boost_fn=None,
    source_labels: Optional[List[str]] = None,
) -> List:
    if weights is not None and len(weights) != len(all_hits):
        raise ValueError(f"weights length ({len(weights)}) != all_hits length ({len(all_hits)})")
    if source_labels is not None and len(source_labels) != len(all_hits):
        raise ValueError("source_labels length must match all_hits length")

    fused_scores: dict[str, float] = {}
    fused_sources: dict[str, dict[str, float]] = {}
    source_scores: dict[str, dict[str, float]] = {}
    chunk_map: dict[str, Any] = {}

    for list_idx, hits in enumerate(all_hits):
        w = weights[list_idx] if weights is not None else 1.0
        label = source_labels[list_idx] if source_labels is not None else f"source_{list_idx}"
        for rank, hit in enumerate(hits):
            cid = hit.get("chunk_id") if isinstance(hit, dict) else getattr(hit, "chunk_id", None)
            if not cid:
                continue
            boost = 1.0
            if section_boost_fn is not None:
                meta = (hit.get("metadata", {}) if isinstance(hit, dict)
                        else getattr(hit, "metadata", {})) or {}
                boost = section_boost_fn(meta)
            contribution = w * boost / (k + rank + 1)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + contribution
            fused_sources.setdefault(cid, {})
            fused_sources[cid][label] = fused_sources[cid].get(label, 0.0) + contribution
            score = hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", None)
            if score is not None:
                try:
                    source_scores.setdefault(cid, {}).setdefault(label, float(score))
                except (TypeError, ValueError):
                    pass
            if cid not in chunk_map:
                chunk_map[cid] = hit

    result = []
    for cid, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        item = chunk_map[cid]
        rrf_score = round(score, 6)
        per_source_scores = source_scores.get(cid, {})
        if isinstance(item, dict):
            item["retrieval_score"] = rrf_score
            item["rrf_score"] = rrf_score
            item["source_scores"] = per_source_scores
            if source_labels is not None:
                item["rrf_source_contributions"] = fused_sources.get(cid, {})
        else:
            item.retrieval_score = rrf_score
            item.rrf_score = rrf_score
            item.source_scores = per_source_scores
            if source_labels is not None:
                item.rrf_source_contributions = fused_sources.get(cid, {})
        result.append(item)
    return result


MULTI_QUERY_PROMPT = """Generate 2 alternative search queries for the following question.
STRICT RULES:
- Stay in the SAME technical domain as the original query
- Do NOT switch fields (e.g., from machine learning to neuroscience)
- Preserve key technical terms (e.g., "transformer", "attention")
- Rephrase, don't reinterpret

Original query: {query}

Alternative queries:"""

HYDE_PROMPT = """Write a short hypothetical passage (2-3 sentences) that would directly answer this question.
Be specific and technical. Write as if you are an expert answering from a research paper.

Question: {query}

Hypothetical passage:"""

class MasterHybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_chunks_path: str = "data/processed/chunks_semantic_v2.json",
        reranker_model: Optional[str] = None,
        rrf_k: int = 60,
        use_multi_query: bool = False,
        use_hyde: bool = False,
    ):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self._policy = PolicyEngine()
        self._chunks = []
        self._bm25 = None
        if reranker_model is not None and reranker_model != settings.reranker_model:
            raise ValueError(
                "Set reranker_model through RAG_RERANK_MODEL / settings.reranker_model "
                "before the shared ModelRegistry is initialized."
            )

        if Path(bm25_chunks_path).exists():
            with open(bm25_chunks_path, "r") as f:
                self._chunks = json.load(f)
            tokenized_corpus = [c["text"].lower().split() for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"[MasterRetriever] BM25 loaded: {len(self._chunks)} chunks")

        from src.retrieval.model_registry import ModelRegistry
        self.reranker = ModelRegistry.get().reranker
        logger.info(f"[MasterRetriever] Reranker loaded (disabled in baseline)")

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq = Groq(api_key=api_key)
            logger.info(f"[MasterRetriever] Groq available (expansion disabled)")
        else:
            self.groq = None
            logger.info(f"[MasterRetriever] No GROQ_API_KEY available")

    def _is_valid_section(self, metadata: dict) -> bool:
        section = str(metadata.get("section", "")).lower()
        return section not in ["references", "bibliography"]
    def _is_reference_chunk(self, chunk) -> bool:
        return _cf_is_reference_chunk(chunk)

    def _is_title_or_author_chunk(self, chunk) -> bool:
        """Deteksi chunk yang isinya judul paper atau baris author/afiliasi."""
        return _cf_is_title_or_author_chunk(chunk)

    @staticmethod
    def _section_boost(metadata: dict) -> float:
        section = str(metadata.get("section", "")).lower()

        if any(kw in section for kw in ("method", "architect", "approach", "model")):
            return 1.4   
        elif any(kw in section for kw in ("experiment", "result", "evaluat", "finding")):
            return 1.3  
        elif "abstract" in section:
            return 1.1   
        elif "introduction" in section:
            return 1.05 
        elif "conclusion" in section:
            return 1.0   
        elif any(ref in section for ref in ("reference", "bibliography")):
            return 0.3  
        else:
            return 1.0  

    # Query Expansion 
    def _call_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self.groq.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _expand_and_hyde_parallel(self, query: str):
        queries = [query]
        hyde_text = None

        if not getattr(self, "groq", None):
            return queries, hyde_text

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if self.use_multi_query:
                futures["mq"] = executor.submit(
                    self._call_groq,
                    MULTI_QUERY_PROMPT.format(query=query), 0.7, 100
                )
            if self.use_hyde:
                futures["hyde"] = executor.submit(
                    self._call_groq,
                    HYDE_PROMPT.format(query=query), 0.5, 150
                )

        if "mq" in futures:
            try:
                raw = futures["mq"].result(timeout=5)
                alternatives = [q.strip() for q in raw.split("\n") if q.strip()][:2]
                queries = [query] + alternatives
                logger.info(f"[MultiQuery] Expanded to {len(queries)} queries: {alternatives}")
            except Exception as e:
                logger.info(f"[MultiQuery] Failed: {e}")

        if "hyde" in futures:
            try:
                hyde_text = futures["hyde"].result(timeout=5)
                logger.info(f"[HyDE] Generated: {hyde_text[:80]}...")
            except Exception as e:
                logger.info(f"[HyDE] Failed: {e}")
        return queries, hyde_text

    def _expand_queries(self, query: str) -> List[str]:
        return [query]

    def _generate_hyde(self, query: str) -> Optional[str]:
        return None

    def _classify_query(self, query: str) -> int:
        words = len(query.strip().split())
        if words <= 5:
            return 3
        if words >= 15:
            return 7
        return 5

    def _resolve_top_k(self, query: str, top_k: int) -> int:
        if top_k <= 0:
            top_k = self._classify_query(query)
            logger.info(f"[TopK] auto-selected top_k={top_k} for query length={len(query.split())}")
        return top_k

    def _choose_candidate_k(self, top_k: int) -> int:
        return min(max(top_k * 4, 30), max(top_k, settings.candidate_k))

    def score_pairs(self, pairs: list) -> list:
        """Public cross-encoder scoring of (query, text) pairs."""
        if not self.reranker:
            return [0.0] * len(pairs)
        return list(self.reranker.predict(
            pairs,
            batch_size=32,
            max_length=self.reranker.max_seq_length,
        ))

    def _rerank(self, query: str, candidates: list, text_of=None) -> tuple[list, dict]:
        if not self.reranker or not candidates:
            return candidates, {
                "attempted": False,
                "succeeded": False,
                "error": None,
                "skipped_reason": "reranker unavailable" if not self.reranker else "no candidates",
            }

        attempted = False
        try:
            pairs = []
            for hit in candidates:
                text = text_of(hit) if text_of is not None else (
                    hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
                )
                pairs.append([query, text])

            attempted = True
            scores = [float(score) for score in self.score_pairs(pairs)]
            if len(scores) != len(candidates):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
                )
            ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
            ranked_candidates = [chunk for chunk, _score in ranked]
            for chunk, score in ranked:
                if isinstance(chunk, dict):
                    chunk["rerank_score"] = score
                else:
                    chunk.rerank_score = score

            return ranked_candidates, {
                "attempted": True,
                "succeeded": True,
                "error": None,
                "skipped_reason": None,
            }
        except Exception as exc:
            logger.warning("[Rerank] failed (%s): %s", type(exc).__name__, exc)
            return candidates, {
                "attempted": attempted,
                "succeeded": False,
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "skipped_reason": None,
            }

    def rerank(self, query: str, candidates: list, text_of=None) -> list:
        """Public rerank entry point.

        ``text_of`` opsional: callable (hit) -> string yang dijadikan pasangan
        rerank untuk hit tersebut (mis. window terbaik). Default: teks chunk.
        """
        ranked, _status = self._rerank(query, candidates, text_of=text_of)
        return ranked

    # Core Retrieval
    def _dense_search(self, query: str, k: int) -> List[RetrievalResult]:
        return self.vector_store.search(query, k=k)

    def dense_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Public dense-retrieval entry point (adapters must use this)."""
        return self._dense_search(query, k)

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "text": self._chunks[i]["text"],
                "score": float(scores[i]),
                "doc_id": self._chunks[i]["doc_id"],
                "metadata": self._chunks[i].get("metadata", {}),
            }
            for i in top_indices
        ]

    def bm25_search(self, query: str, k: int) -> List[Dict]:
        """Public BM25 entry point (adapters must use this)."""
        return self._bm25_search(query, k)

    @property
    def chunks(self) -> list:
        """Read-only view of the loaded BM25 corpus chunks."""
        return self._chunks
    def _rrf_fuse(
        self,
        all_dense_hits: List[List],
        all_bm25_hits: List[List],
        candidate_k: int,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List:
        all_hits = all_dense_hits + all_bm25_hits
        weights  = [dense_weight] * len(all_dense_hits) + [bm25_weight] * len(all_bm25_hits)

        results = rrf_fuse(
            all_hits=all_hits,
            k=self.rrf_k,
            weights=weights,
            section_boost_fn=self._section_boost,
            source_labels=["dense"] * len(all_dense_hits) + ["bm25"] * len(all_bm25_hits),
        )[:candidate_k]

        ref_penalized = sum(
            1 for c in results
            if self._section_boost(
                (c.get("metadata", {}) if isinstance(c, dict)
                 else getattr(c, "metadata", {})) or {}
            ) == 0.3
        )
        logger.info(
            f"[RRF_FILTER] candidates={len(results)} | "
            f"refs_penalized={ref_penalized} (boost=0.3, not blocked)"
        )
        return results

    def rrf_fuse_search(
        self,
        all_dense_hits: List[List],
        all_bm25_hits: List[List],
        candidate_k: int,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List:
        """Public RRF fusion entry point (adapters must use this)."""
        return self._rrf_fuse(all_dense_hits, all_bm25_hits, candidate_k, dense_weight, bm25_weight)

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank: bool = False,
        allow_references: Optional[bool] = None,
        max_ref_ratio: Optional[float] = None,
        use_multi_query: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        formatted, _trace = self._run_search(
            query, top_k=top_k, dense_weight=dense_weight,
            bm25_weight=bm25_weight, rerank=rerank,
            allow_references=allow_references, max_ref_ratio=max_ref_ratio,
            use_multi_query=use_multi_query, use_hyde=use_hyde,
            min_score_threshold=min_score_threshold,
        )
        return formatted

    def search_with_trace(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank: bool = False,
        allow_references: Optional[bool] = None,
        max_ref_ratio: Optional[float] = None,
        use_multi_query: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        min_score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Jalankan pipeline canonical dan kembalikan hasil + trace per stage."""
        formatted, trace = self._run_search(
            query, top_k=top_k, dense_weight=dense_weight,
            bm25_weight=bm25_weight, rerank=rerank,
            allow_references=allow_references, max_ref_ratio=max_ref_ratio,
            use_multi_query=use_multi_query, use_hyde=use_hyde,
            min_score_threshold=min_score_threshold,
        )
        return {"results": formatted, "trace": trace}

    def _run_search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank: bool = False,
        allow_references: Optional[bool] = None,
        max_ref_ratio: Optional[float] = None,
        use_multi_query: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        min_score_threshold: Optional[float] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t_search_start = time.time()
        top_k = self._resolve_top_k(query, top_k)
        plan = self._policy.initial_plan(query)
        allow_references = getattr(plan, "allow_references", False) if allow_references is None else allow_references
        max_ref_ratio = getattr(plan, "max_ref_ratio", 0.0) if max_ref_ratio is None else max_ref_ratio
        max_ref_ratio = min(max(float(max_ref_ratio), 0.0), 1.0)
        self.use_multi_query = plan.allow_multi_query if use_multi_query is None else use_multi_query
        self.use_hyde = plan.allow_hyde if use_hyde is None else use_hyde
        logger.info(
            f"[POLICY] query_type={plan.query_type} | "
            f"multi_query={self.use_multi_query} | hyde={self.use_hyde}"
        )
        candidate_k = self._choose_candidate_k(top_k)

        query_variants, hyde_query = self._expand_and_hyde_parallel(query)
        dense_queries = list(query_variants)
        if hyde_query:
            dense_queries.append(hyde_query)
        all_dense_hits = []
        all_bm25_hits = []

        t_dense_start = time.time()
        all_dense_hits.extend(
            self._dense_search(search_query, k=candidate_k)
            for search_query in dense_queries
        )
        t_dense = time.time() - t_dense_start

        t_bm25_start = time.time()
        all_bm25_hits.extend(
            self._bm25_search(search_query, k=candidate_k)
            for search_query in query_variants
        )
        t_bm25 = time.time() - t_bm25_start

        t_fuse_start = time.time()
        fused_results = self._rrf_fuse(all_dense_hits, all_bm25_hits, candidate_k, dense_weight, bm25_weight)

        effective_min_score = settings.min_score_threshold if min_score_threshold is None else min_score_threshold
        fused_results = [
            c for c in fused_results
            if (c.get("retrieval_score", 0.0) if isinstance(c, dict) else getattr(c, "retrieval_score", 0.0)) >= effective_min_score
        ]

        logger.info(
            "[RETRIEVAL_STAGE] after_score_filter=%d",
            len(fused_results),
        )

        candidates = [c for c in fused_results if not _cf_is_low_quality_chunk(c)]
        rerank_candidates = candidates[:20] if rerank else []
        t_fuse = time.time() - t_fuse_start
        t_rerank_start = time.time()
        if rerank:
            reranked, reranker_status = self._rerank(query, rerank_candidates)
        else:
            logger.info("[Rerank] skipped (rerank=False); filtering candidates in RRF order")
            reranked = candidates
            reranker_status = {
                "attempted": False,
                "succeeded": False,
                "error": None,
                "skipped_reason": "rerank=False",
            }
        t_rerank = time.time() - t_rerank_start

        t_filter_start = time.time()
        filtered_chunks = [
            c for c in reranked
            if _cf_passes_strict_filter(c, allow_references=allow_references)
        ]

        # Breakdown terstruktur: alasan tiap chunk ditolak
        bd = _cf_filter_breakdown(reranked, allow_references=allow_references)
        logger.info(
            "[RETRIEVAL_STAGE] fused=%d | candidates=%d | "
            "rerank_candidates=%d | reranked=%d | filtered=%d | "
            "blocked_ref=%d | blocked_title_author=%d | blocked_fragment=%d | "
            "blocked_low_quality=%d | blocked_not_informative=%d",
            len(fused_results),
            len(candidates),
            len(rerank_candidates),
            len(reranked),
            len(filtered_chunks),
            bd["reference"],
            bd["title_author"],
            bd["fragment"],
            bd["low_quality"],
            bd["not_informative"],
        )

        def apply_reference_budget(items):
            max_references = int(top_k * max_ref_ratio)
            selected = []
            references_used = 0
            removed = 0
            for item in items:
                if _cf_is_reference_chunk(item):
                    if not allow_references or references_used >= max_references:
                        removed += 1
                        continue
                    references_used += 1
                selected.append(item)
                if len(selected) == top_k:
                    break
            return selected, removed

        final_chunks, reference_budget_removed = apply_reference_budget(filtered_chunks)
        if final_chunks:
            should_abstain = False
            filter_stage = "strict"
        else:
            fallback_pool = [
                c for c in reranked
                if _cf_passes_minimum_filter(c, allow_references=allow_references)
            ]
            final_chunks, reference_budget_removed = apply_reference_budget(fallback_pool)
            should_abstain = not bool(final_chunks)
            filter_stage = "fallback" if final_chunks else "none"

            blocked = _cf_filter_breakdown(reranked, allow_references=allow_references)
            logger.warning(
                "[RETRIEVAL_FALLBACK] strict filter returned no chunks | "
                "fallback_pool=%d | passed_minimum=%d | "
                "blocked_ref=%d | blocked_title_author=%d | blocked_fragment=%d | "
                "blocked_low_quality=%d | blocked_not_informative=%d | "
                "abstain=%s",
                len(reranked),
                len(final_chunks),
                blocked["reference"],
                blocked["title_author"],
                blocked["fragment"],
                blocked["low_quality"],
                blocked["not_informative"],
                should_abstain,
            )
        t_filter = time.time() - t_filter_start

        # Per-final-chunk detail for debugging. This runs for EVERY query, so it must
        # not go to stdout — PROJECT_GOALS.md §4 requires structured, queryable
        # observability, and bare print() bypasses the logging pipeline entirely.
        # The same data is already available in trace["final_rows"] and
        # trace["filter_breakdown"].
        for c in final_chunks:
            cid = c["chunk_id"] if isinstance(c, dict) else getattr(c, "chunk_id", None)
            rerank_score = c.get("rerank_score") if isinstance(c, dict) else getattr(c, "rerank_score", None)
            text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
            text_len = len(text.split())
            starts_ok = not text.strip().lower().startswith(("and ", "or ", "but "))
            ends_ok = text.strip().endswith((".", ":", ";", "?", ")"))
            score_str = f"{rerank_score:.2f}" if isinstance(rerank_score, (int, float)) else "n/a"
            logger.debug(
                "  %s: score=%s, words=%d, complete=(%s&%s)",
                cid, score_str, text_len, starts_ok, ends_ok,
            )

        reranker_executed = reranker_status["succeeded"]
        retrieval_method = "hybrid_rrf_rerank" if reranker_executed else "hybrid_rrf"
        ranking_mode = "rrf_reranker" if reranker_executed else "rrf_only"
        formatted = self._format_output(
            final_chunks, retrieval_method, should_abstain,
            filter_stage=filter_stage, ranking_mode=ranking_mode,
        )

        t_search_total = time.time() - t_search_start
        t_fuse_and_rerank = t_fuse + t_rerank

        raw_hits = [
            hit
            for hit_list in all_dense_hits + all_bm25_hits
            for hit in hit_list
        ]
        ref_count = sum(1 for hit in raw_hits if self._is_reference_chunk(hit))

        logger.info(
            f"[RETRIEVAL_LATENCY] query='{query[:50]}' | "
            f"dense_ms={t_dense*1000:.1f} | bm25_ms={t_bm25*1000:.1f} | "
            f"fuse_ms={t_fuse*1000:.1f} | total_ms={t_search_total*1000:.1f} | "
            f"raw_refs={ref_count} | final={len(formatted)}"
        )

        def _ids(items):
            ids = []
            for it in items:
                ids.append(it.get("chunk_id") if isinstance(it, dict) else getattr(it, "chunk_id", None))
            return ids

        def _scores(items, source: str):
            rows: list[dict] = []
            for it in items[:50]:
                cid = it.get("chunk_id") if isinstance(it, dict) else getattr(it, "chunk_id", None)
                did = it.get("doc_id") if isinstance(it, dict) else getattr(it, "doc_id", None)
                text = it.get("text") if isinstance(it, dict) else getattr(it, "text", None)
                if source == "dense":
                    score = it.get("score") if isinstance(it, dict) else getattr(it, "score", None)
                elif source == "bm25":
                    score = it.get("score") if isinstance(it, dict) else getattr(it, "score", None)
                elif source == "rrf":
                    score = it.get("rrf_score", it.get("retrieval_score")) if isinstance(it, dict) else getattr(it, "rrf_score", None)
                elif source == "reranked":
                    score = it.get("rerank_score") if isinstance(it, dict) else getattr(it, "rerank_score", None)
                else:
                    score = None
                if source in ("dense", "bm25"):
                    score_map = it.get("source_scores", {}) if isinstance(it, dict) else getattr(it, "source_scores", None) or {}
                    score = score_map.get(source, score)
                meta_extra = {}
                if source in ("rrf", "filtered", "final"):
                    contrib = it.get("rrf_source_contributions") if isinstance(it, dict) else getattr(it, "rrf_source_contributions", None)
                    if contrib:
                        meta_extra = {"dense_contrib": contrib.get("dense"), "bm25_contrib": contrib.get("bm25")}
                rows.append({"chunk_id": cid, "doc_id": did, "text": (text or "")[:200], "score": float(score) if score is not None else None, **meta_extra})
            return rows

        trace = {
            "dense_ids": _ids(all_dense_hits[0] if all_dense_hits else []),
            "bm25_ids": _ids(all_bm25_hits[0] if all_bm25_hits else []),
            "dense_ids_by_query": [_ids(hits) for hits in all_dense_hits],
            "bm25_ids_by_query": [_ids(hits) for hits in all_bm25_hits],
            "rrf_ids": _ids(fused_results),
            "reranked_ids": _ids(reranked),
            "filtered_ids": _ids(filtered_chunks),
            "final_ids": _ids(final_chunks),
            "dense_rows": _scores(all_dense_hits[0] if all_dense_hits else [], "dense"),
            "bm25_rows": _scores(all_bm25_hits[0] if all_bm25_hits else [], "bm25"),
            "rrf_rows": _scores(fused_results, "rrf"),
            "reranked_rows": _scores(reranked, "reranked"),
            "filtered_rows": _scores(filtered_chunks, "rrf"),
            "final_rows": _scores(final_chunks, "rrf"),
            "rerank_input_ids": _ids(rerank_candidates),
            "rerank_input_count": len(rerank_candidates),
            "reranker_executed": reranker_executed,
            "reranker_attempted": reranker_status["attempted"],
            "reranker_succeeded": reranker_status["succeeded"],
            "reranker_error": reranker_status["error"],
            "reranker_skipped_reason": reranker_status["skipped_reason"],
            "ranking_mode": ranking_mode,
            "reranker_skip_reason": (
                None if reranker_executed else reranker_status["skipped_reason"]
            ),
            "candidate_k": candidate_k,
            "dense_query_count": len(dense_queries),
            "bm25_query_count": len(query_variants),
            "hyde_used": bool(hyde_query),
            "should_abstain": should_abstain,
            "filter_stage": filter_stage,
            "reference_budget_removed": reference_budget_removed,
            "latency_ms": {
                "dense": round(t_dense * 1000, 2),
                "bm25": round(t_bm25 * 1000, 2),
                "fusion": round(t_fuse * 1000, 2),
                "reranker": round(t_rerank * 1000, 2),
                "filtering": round(t_filter * 1000, 2),
                "fuse_and_rerank": round(t_fuse_and_rerank * 1000, 2),
                "total": round(t_search_total * 1000, 2),
            },
            "counts": {
                "dense": sum(map(len, all_dense_hits)),
                "bm25": sum(map(len, all_bm25_hits)),
                "fused": len(fused_results),
                "reranked": len(reranked),
                "strict_filtered": len(filtered_chunks),
                "final": len(final_chunks),
            },
            "filter_breakdown": bd,
        }
        return formatted, trace

    def _format_output(
        self,
        results: list,
        method: str,
        should_abstain: bool,
        filter_stage: str = "strict",
        ranking_mode: str = "rrf_only",
    ):
        formatted = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                cid  = r.get("chunk_id")
                txt  = r.get("text")
                did  = r.get("doc_id")
                rerank_score = r.get("rerank_score") if isinstance(r, dict) else getattr(r, "rerank_score", None)
                rrf_score = r.get("rrf_score") or r.get("retrieval_score", 0) if isinstance(r, dict) else getattr(r, "rrf_score", None) or getattr(r, "retrieval_score", 0)
                meta = r.get("metadata", {})
            else:
                cid  = getattr(r, "chunk_id", "unknown")
                txt  = getattr(r, "text", "")
                did  = getattr(r, "doc_id", "unknown")
                rerank_score = getattr(r, "rerank_score", None)
                rrf_score = getattr(r, "retrieval_score", None) or getattr(r, "score", 0)
                meta = getattr(r, "metadata", {})

            formatted.append({
                "chunk_id":         cid,
                "text":             txt,
                "doc_id":           did,
                "rerank_score":     float(rerank_score) if rerank_score is not None else None,
                "retrieval_score":  float(rrf_score),
                "final_score":      float(rerank_score) if rerank_score is not None else float(rrf_score),
                "retrieval_rank":   i + 1,
                "retrieval_method": method,
                "ranking_mode": ranking_mode,
                "metadata":         meta,
                "dense_score":      (r.get("source_scores", {}).get("dense", r.get("dense_score")) if isinstance(r, dict) else (getattr(r, "source_scores", None) or {}).get("dense", getattr(r, "score", None))),
                "bm25_score":       (r.get("source_scores", {}).get("bm25", r.get("bm25_score")) if isinstance(r, dict) else (getattr(r, "source_scores", None) or {}).get("bm25", None)),
                "rrf_score":        float(rrf_score),
                "should_abstain":   should_abstain,
                "filter_stage":     filter_stage,
            })
        return formatted

# --- CLI TEST ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store     = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)

    test_queries = [
        "What is Retrieval-Augmented Generation?",
        "How does semantic chunking work?",
        "What are attention mechanisms?",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("[PRODUCTION TEST] Running with TIERED SECTION BOOST + HARD REFERENCE FILTERING")
    logger.info("Target: references=0 in top results | methods/experiments ranked highest")
    logger.info("=" * 80)

    results_all = []

    for query in test_queries:
        t_start = time.time()
        results = retriever.search(query, top_k=10)
        latency_ms = round((time.time() - t_start) * 1000, 2)

        ref_in_top = sum(1 for r in results[:3] if retriever._is_reference_chunk(r))

        logger.info(f"\n[QUERY] {query}")
        logger.info(f"[TOTAL_LATENCY] {latency_ms}ms")
        logger.info(f"[REF_CHECK] references in top-3: {ref_in_top} (target: 0)")

        for r in results[:3]:
            section = (r.get("metadata") or {}).get("section", "unknown")
            boost   = MasterHybridRetriever._section_boost(r.get("metadata") or {})
            logger.info(
                f"  Rank {r['retrieval_rank']}: score={r['retrieval_score']:.4f} | "
                f"section={section} | boost={boost} | doc={r['doc_id']}"
            )
            logger.info(f"    Text: {r['text'][:100]}...")

        results_all.append({
            "query":       query,
            "latency_ms":  latency_ms,
            "top_chunks":  results[:3],
            "num_results": len(results),
            "refs_in_top3": ref_in_top,
        })
    logger.info("\n" + "=" * 80)
    avg_latency  = sum(r["latency_ms"] for r in results_all) / len(results_all)
    total_refs   = sum(r["refs_in_top3"] for r in results_all)
    logger.info(f"[SUMMARY] queries={len(results_all)} | avg_latency={avg_latency:.1f}ms | refs_in_top3={total_refs}")
    logger.info("=" * 80 + "\n")
    store.close()
