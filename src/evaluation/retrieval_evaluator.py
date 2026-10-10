"""Retrieval-only evaluator — tidak butuh LLM maupun RAGAS.

Menghitung metrik retrieval murni dari daftar retrieved chunk ids:
Recall@k, MRR@k, hit rate, gold rank, failure type.

Deterministic, no network, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation.gold_matching import best_gold_rank, is_any_gold_hit
from src.evaluation.gold_schema import _UNSET, normalize_gold_chunk_ids

FAILURE_SUCCESS = "success"
FAILURE_RANKING = "ranking_failure"
FAILURE_MISS = "retrieval_miss"
FAILURE_EVAL_DATA = "evaluation_data_error"


@dataclass
class QueryStageResult:
    query_id: str
    question: str
    gold_chunk_id: Optional[str]
    gold_chunk_ids: List[str]
    gold_doc_id: Optional[str]
    stage: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    gold_rank: Optional[int] = None
    hit_at_1: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: float = 0.0
    retrieval_failure_type: str = FAILURE_SUCCESS
    num_candidates: int = 0
    latency_ms: float = 0.0


def classify_failure(gold_rank_value: Optional[int]) -> str:
    if gold_rank_value is None:
        return FAILURE_MISS
    if gold_rank_value > 5:
        return FAILURE_RANKING
    return FAILURE_SUCCESS


def evaluate_query(
    *,
    query_id: str,
    question: str,
    gold_chunk_id: object = _UNSET,
    gold_chunk_ids: object = _UNSET,
    gold_doc_id: Optional[str] = None,
    retrieved_chunk_ids: List[str],
    retrieved_doc_ids: Optional[List[str]] = None,
    stage: str,
    latency_ms: float = 0.0,
) -> QueryStageResult:
    gold_ids = normalize_gold_chunk_ids(
        {
            **({"gold_chunk_id": gold_chunk_id} if gold_chunk_id is not _UNSET else {}),
            **({"gold_chunk_ids": gold_chunk_ids} if gold_chunk_ids is not _UNSET else {}),
        }
    )
    rank = best_gold_rank(gold_ids, retrieved_chunk_ids)
    row = QueryStageResult(
        query_id=query_id,
        question=question,
        gold_chunk_id=gold_ids[0],
        gold_chunk_ids=gold_ids,
        gold_doc_id=gold_doc_id,
        stage=stage,
        retrieved_chunk_ids=list(retrieved_chunk_ids),
        retrieved_doc_ids=list(retrieved_doc_ids or []),
        gold_rank=rank,
        hit_at_1=is_any_gold_hit(gold_ids, retrieved_chunk_ids[:1]),
        hit_at_5=is_any_gold_hit(gold_ids, retrieved_chunk_ids[:5]),
        hit_at_10=is_any_gold_hit(gold_ids, retrieved_chunk_ids[:10]),
        reciprocal_rank=(1.0 / rank) if rank and rank <= 10 else 0.0,
        retrieval_failure_type=classify_failure(rank),
        num_candidates=len(retrieved_chunk_ids),
        latency_ms=latency_ms,
    )
    return row


def aggregate_rows(rows: List[QueryStageResult]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hits1 = sum(1 for r in rows if r.hit_at_1)
    hits5 = sum(1 for r in rows if r.hit_at_5)
    hits10 = sum(1 for r in rows if r.hit_at_10)
    rr_sum = sum(r.reciprocal_rank for r in rows)
    gold_ranks = [r.gold_rank for r in rows if r.gold_rank is not None]
    latencies = [r.latency_ms for r in rows]
    failure_counts: Dict[str, int] = {}
    for r in rows:
        failure_counts[r.retrieval_failure_type] = failure_counts.get(r.retrieval_failure_type, 0) + 1
    return {
        "n": n,
        "recall@1": round(hits1 / n, 4),
        "recall@5": round(hits5 / n, 4),
        "recall@10": round(hits10 / n, 4),
        "mrr@10": round(rr_sum / n, 4),
        "avg_candidates": round(sum(r.num_candidates for r in rows) / n, 2),
        "avg_gold_rank_when_hit": round(sum(gold_ranks) / len(gold_ranks), 2) if gold_ranks else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "failure_counts": failure_counts,
    }


def doc_level_hit(gold_doc_id: Optional[str], retrieved_doc_ids: List[str], k: int = 10) -> Optional[bool]:
    """Document recall@k bila gold_doc_id tersedia."""
    if not gold_doc_id:
        return None
    target = str(gold_doc_id).strip().lower()
    return any(str(d).strip().lower() == target for d in retrieved_doc_ids[:k])
