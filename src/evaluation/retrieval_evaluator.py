from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    gold_chunk_id: str | None
    gold_chunk_ids: list[str]
    gold_doc_id: str | None
    stage: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    gold_rank: int | None = None
    hit_at_1: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: float = 0.0
    retrieval_failure_type: str = FAILURE_SUCCESS
    num_candidates: int = 0
    latency_ms: float = 0.0
    missing_gold_chunk_ids: list[str] = field(default_factory=list)


def classify_failure(gold_rank_value: int | None) -> str:
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
    gold_doc_id: str | None = None,
    should_abstain: bool = False,
    retrieved_chunk_ids: list[str],
    retrieved_doc_ids: list[str] | None = None,
    stage: str,
    latency_ms: float = 0.0,
    corpus_chunk_ids: set[str] | None = None,
) -> QueryStageResult:
    if should_abstain:
        gold_ids: list[str] = []
    else:
        gold_ids = normalize_gold_chunk_ids(
            {
                **({"gold_chunk_id": gold_chunk_id} if gold_chunk_id is not _UNSET else {}),
                **({"gold_chunk_ids": gold_chunk_ids} if gold_chunk_ids is not _UNSET else {}),
            }
        )

    missing = (
        [g for g in gold_ids if g not in corpus_chunk_ids] if corpus_chunk_ids is not None else []
    )
    if missing or not gold_ids:
        return QueryStageResult(
            query_id=query_id,
            question=question,
            gold_chunk_id=gold_ids[0] if gold_ids else None,
            gold_chunk_ids=gold_ids,
            gold_doc_id=gold_doc_id,
            stage=stage,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            retrieved_doc_ids=list(retrieved_doc_ids or []),
            retrieval_failure_type=FAILURE_EVAL_DATA,
            num_candidates=len(retrieved_chunk_ids),
            latency_ms=latency_ms,
            missing_gold_chunk_ids=missing,
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


def aggregate_rows(rows: list[QueryStageResult]) -> dict[str, Any]:
    scored = [r for r in rows if r.retrieval_failure_type != FAILURE_EVAL_DATA]
    excluded = len(rows) - len(scored)

    failure_counts: dict[str, int] = {}
    for r in rows:
        failure_counts[r.retrieval_failure_type] = (
            failure_counts.get(r.retrieval_failure_type, 0) + 1
        )

    n = len(scored)
    if n == 0:
        return {
            "n": 0,
            "excluded_evaluation_data_errors": excluded,
            "failure_counts": failure_counts,
        }
    hits1 = sum(1 for r in scored if r.hit_at_1)
    hits5 = sum(1 for r in scored if r.hit_at_5)
    hits10 = sum(1 for r in scored if r.hit_at_10)
    rr_sum = sum(r.reciprocal_rank for r in scored)
    gold_ranks = [r.gold_rank for r in scored if r.gold_rank is not None]
    latencies = [r.latency_ms for r in scored]
    return {
        "n": n,
        "excluded_evaluation_data_errors": excluded,
        "recall@1": round(hits1 / n, 4),
        "recall@5": round(hits5 / n, 4),
        "recall@10": round(hits10 / n, 4),
        "mrr@10": round(rr_sum / n, 4),
        "avg_candidates": round(sum(r.num_candidates for r in scored) / n, 2),
        "avg_gold_rank_when_hit": round(sum(gold_ranks) / len(gold_ranks), 2)
        if gold_ranks
        else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "failure_counts": failure_counts,
    }
