import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_evaluator import aggregate_rows, evaluate_query
from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids


def _row(query_id, gold, retrieved, latency_ms=0.0, stage="rrf_reranked"):
    return evaluate_query(
        query_id=query_id,
        question="q",
        gold_chunk_id=gold,
        retrieved_chunk_ids=retrieved,
        stage=stage,
        latency_ms=latency_ms,
    )


def test_hit_at_k_and_mrr():
    r = _row("q1", "gold", ["x", "gold", "y"])
    assert r.hit_at_1 is False
    assert r.hit_at_5 is True
    assert r.gold_rank == 2
    assert r.reciprocal_rank == 0.5
    assert r.retrieval_failure_type == "success"


def test_miss_and_ranking_failure_typing():
    miss = _row("q1", "gold", ["a", "b"])
    assert miss.retrieval_failure_type == "retrieval_miss"
    rank = _row("q1", "gold", ["a", "b", "c", "d", "e", "gold"])
    assert rank.retrieval_failure_type == "ranking_failure"


def test_mrr_at_ten_is_zero_when_gold_is_below_ten():
    rank = _row("q1", "gold", [f"x{i}" for i in range(10)] + ["gold"])
    assert rank.gold_rank == 11
    assert rank.hit_at_10 is False
    assert rank.reciprocal_rank == 0.0


def test_subchunk_counts_as_hit():
    r = _row("q1", "doc_rs_0001", ["doc_rs_0001_sub2"])
    assert r.hit_at_1 is True and r.gold_rank == 1


def test_aggregate_rates():
    rows = [
        _row("q1", "g", ["g"]),
        _row("q2", "g", ["x", "g"]),
        _row("q3", "g", ["x", "y", "z", "w", "v", "g"]),
        _row("q4", "g", ["x"]),
    ]
    agg = aggregate_rows(rows)
    assert agg["n"] == 4
    assert agg["recall@1"] == 0.25
    assert agg["recall@5"] == 0.5
    assert agg["recall@10"] == 0.75
    assert agg["failure_counts"]["retrieval_miss"] == 1
    assert abs(agg["mrr@10"] - ((1 + 0.5 + 1 / 6) / 4)) < 1e-3


def test_v1_and_v2_single_gold_metrics_match_for_canonical_dataset():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "data/evaluation/retrieval_eval_dataset_v2.json").read_text())
    records = dataset_records(payload)
    assert len(records) == 13

    v1_rows = []
    v2_rows = []
    for index, item in enumerate(records):
        gold_id = item["gold_chunk_ids"][0]
        retrieved = [f"distractor-{index}-{rank}" for rank in range(index % 12)]
        retrieved.insert(index % 12, gold_id)
        v1_rows.append(evaluate_query(
            query_id=item["query_id"],
            question=item["question"],
            gold_chunk_id=gold_id,
            gold_doc_id=item["gold_doc_id"],
            retrieved_chunk_ids=retrieved,
            stage="final_filtered",
        ))
        v2_rows.append(evaluate_query(
            query_id=item["query_id"],
            question=item["question"],
            gold_chunk_ids=item["gold_chunk_ids"],
            gold_doc_id=item["gold_doc_id"],
            retrieved_chunk_ids=retrieved,
            stage="final_filtered",
        ))

    assert v1_rows == v2_rows
    assert [row.gold_rank for row in v1_rows] == [row.gold_rank for row in v2_rows]
    assert [row.hit_at_10 for row in v1_rows] == [row.hit_at_10 for row in v2_rows]
    assert [row.reciprocal_rank for row in v1_rows] == [row.reciprocal_rank for row in v2_rows]
    assert aggregate_rows(v1_rows) == aggregate_rows(v2_rows)


def test_multi_gold_uses_best_rank_and_any_gold_hit():
    result = evaluate_query(
        query_id="multi",
        question="q",
        gold_chunk_ids=["B", "A"],
        retrieved_chunk_ids=["A", "X", "B", "Y"],
        stage="rrf_fused",
    )
    assert result.hit_at_1 is True
    assert result.gold_rank == 1
    assert result.reciprocal_rank == 1.0

    result = evaluate_query(
        query_id="multi",
        question="q",
        gold_chunk_ids=["A", "B"],
        retrieved_chunk_ids=["X", "Y", "B"],
        stage="rrf_fused",
    )
    assert result.hit_at_10 is True
    assert result.gold_rank == 3
    assert result.reciprocal_rank == 1 / 3

    result = evaluate_query(
        query_id="multi",
        question="q",
        gold_chunk_ids=["A", "B"],
        retrieved_chunk_ids=["X", "Y", "Z"],
        stage="rrf_fused",
    )
    assert result.hit_at_10 is False
    assert result.gold_rank is None
    assert result.reciprocal_rank == 0.0


def test_gold_schema_rejects_invalid_shapes():
    invalid_rows = [
        ({}, "missing gold_chunk_id"),
        ({"gold_chunk_ids": []}, "must not be empty"),
        ({"gold_chunk_ids": ["A", 1]}, "element must be"),
        ({"gold_chunk_ids": [None]}, "element must be"),
        ({"gold_chunk_ids": ["A", "A"]}, "unique"),
        ({"gold_chunk_ids": "A"}, "array of strings"),
        ({"gold_chunk_id": ["A"]}, "non-empty string"),
        ({"gold_chunk_id": None}, "non-empty string"),
        ({"gold_chunk_id": "A", "gold_chunk_ids": ["A"]}, "exactly one"),
    ]
    for row, message in invalid_rows:
        with pytest.raises(ValueError, match=message):
            normalize_gold_chunk_ids(row)


def test_evaluator_rejects_missing_or_null_gold_schema():
    with pytest.raises(ValueError, match="missing gold_chunk_id"):
        _row_without_gold()
    with pytest.raises(ValueError, match="array of strings"):
        evaluate_query(
            query_id="q",
            question="q",
            gold_chunk_ids=None,
            retrieved_chunk_ids=[],
            stage="rrf_fused",
        )


def _row_without_gold():
    return evaluate_query(
        query_id="q",
        question="q",
        retrieved_chunk_ids=[],
        stage="rrf_fused",
    )
