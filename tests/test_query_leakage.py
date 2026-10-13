"""Query-leakage gate — PROJECT_GOALS.md §10.2.

The arithmetic is tested with a fake embedder. The real one costs seconds to
load and needs the model on disk, which would make these tests depend on a
download; the property under test is the comparison, not the encoder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.query_leakage import (
    LEAKAGE_THRESHOLD,
    cosine_similarities,
    dataset_leakage,
    find_leaks,
)

CHUNKS = [
    {"chunk_id": "c1", "text": "Attention compares query and key representations."},
    {"chunk_id": "c2", "text": "Recall is the share of gold chunks retrieved."},
]


def fake_embedder(table: dict[str, tuple[float, ...]]):
    """Look text up in `table`; unknown text gets a distinct orthogonal unit."""

    def encode(texts):
        return [table.get(t, (0.0, 0.0)) for t in texts]

    return encode


UNIT_X = (1.0, 0.0)
UNIT_Y = (0.0, 1.0)


def test_identical_vectors_score_one():
    sims = cosine_similarities(["a", "b"], ["a", "b"], embedder=lambda t: [UNIT_X, UNIT_Y][: len(t)] if t == ["a"] else [UNIT_X])
    assert sims[0] == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    encode = fake_embedder({"a": UNIT_X, "b": UNIT_Y})
    assert cosine_similarities(["a"], ["b"], embedder=encode) == [pytest.approx(0.0)]


def test_opposite_vectors_score_minus_one():
    encode = fake_embedder({"a": (1.0, 0.0), "b": (-1.0, 0.0)})
    assert cosine_similarities(["a"], ["b"], embedder=encode) == [pytest.approx(-1.0)]


def test_unequal_normalised_by_magnitude():
    encode = fake_embedder({"a": (3.0, 0.0), "b": (0.0, 7.0)})
    assert cosine_similarities(["a"], ["b"], embedder=encode) == [pytest.approx(0.0)]


def test_zero_vector_does_not_produce_nan():
    """A zero embedding must score 0.0, not NaN.

    NaN compares False against every threshold, so it would silently pass
    every leak check instead of failing visibly.
    """
    encode = fake_embedder({"a": (0.0, 0.0), "b": UNIT_X})
    result = cosine_similarities(["a"], ["b"], embedder=encode)
    assert math.isfinite(result[0])
    assert result[0] == pytest.approx(0.0)


def test_parallel_lists_are_required():
    with pytest.raises(ValueError, match="parallel"):
        cosine_similarities(["a"], ["a", "b"], embedder=lambda t: [UNIT_X] * len(t))


def test_empty_input_returns_empty():
    assert cosine_similarities([], [], embedder=lambda t: []) == []


# --- the gate itself ---------------------------------------------------------


def test_question_restating_its_chunk_is_reported_as_leak():
    encode = fake_embedder({
        "What is attention?": UNIT_X,
        "Attention is a mechanism.": UNIT_X,
    })
    report = dataset_leakage(
        {"records": [{
            "query_id": "q1",
            "question": "What is attention?",
            "gold_chunk_id": "c1",
        }]},
        [{"chunk_id": "c1", "text": "Attention is a mechanism."}],
        embedder=encode,
    )
    assert report["leaked"] == 1
    assert report["detail"][0]["similarity"] == pytest.approx(1.0)
    assert report["detail"][0]["leaked"] is True


def test_unrelated_question_is_not_a_leak():
    encode = fake_embedder({"What is the capital of Peru?": UNIT_X, "chunk text": UNIT_Y})
    report = dataset_leakage(
        {"records": [{
            "query_id": "q1",
            "question": "What is the capital of Peru?",
            "gold_chunk_id": "c1",
        }]},
        [{"chunk_id": "c1", "text": "chunk text"}],
        embedder=encode,
    )
    assert report["leaked"] == 0
    assert report["max_similarity"] == pytest.approx(0.0)


def test_threshold_boundary_is_exclusive_at_exactly_09():
    """§10.2 says 'similarity > 0.9', so exactly 0.9 passes."""
    encode = fake_embedder({"q": (0.9, 0.4358898943540674)})
    report = dataset_leakage(
        {"records": [{"query_id": "q1", "question": "q", "gold_chunk_id": "c1"}]},
        [{"chunk_id": "c1", "text": "c"}],
        embedder=encode,
    )
    assert report["leaked"] == 0


def test_custom_threshold_is_honoured():
    encode = fake_embedder({"q": UNIT_X, "c": UNIT_Y})
    rows = [{"query_id": "q1", "question": "q", "gold_chunk_id": "c1"}]
    chunks = [{"chunk_id": "c1", "text": "c"}]
    assert dataset_leakage({"records": rows}, chunks, embedder=encode)["leaked"] == 0
    strict = dataset_leakage({"records": rows}, chunks, embedder=encode, threshold=-0.5)
    assert strict["leaked"] == 1


def test_abstain_rows_are_skipped_not_scored_as_leaks():
    encode = fake_embedder({"q": UNIT_X, "c": UNIT_X})
    report = dataset_leakage(
        {"records": [{
            "query_id": "adv", "question": "q", "should_abstain": True,
            "gold_chunk_id": "",
        }]},
        [{"chunk_id": "c1", "text": "c"}],
        embedder=encode,
    )
    assert report["scored"] == 0
    assert report["skipped"] == 1
    assert report["leaked"] == 0


def test_gold_id_absent_from_corpus_is_skipped():
    encode = fake_embedder({"q": UNIT_X})
    report = dataset_leakage(
        {"records": [{"query_id": "q1", "question": "q", "gold_chunk_id": "gone"}]},
        CHUNKS,
        embedder=encode,
    )
    assert report["scored"] == 0
    assert report["skipped"] == 1


def test_multihop_row_is_scored_against_its_first_gold_chunk():
    encode = fake_embedder({"q": UNIT_X, "first": UNIT_X})
    report = dataset_leakage(
        {"records": [{
            "query_id": "m1", "question": "q", "gold_chunk_ids": ["c1", "c2"],
        }]},
        CHUNKS,
        embedder=encode,
    )
    assert report["scored"] == 1
    assert report["detail"][0]["gold_chunk_id"] == "c1"


def test_report_records_the_threshold_and_model():
    report = dataset_leakage(
        {"records": [{"query_id": "q", "question": "q", "gold_chunk_id": "c1"}]},
        CHUNKS,
        embedder=fake_embedder({"q": UNIT_X, "c": UNIT_Y}),
    )
    assert report["threshold"] == LEAKAGE_THRESHOLD
    assert report["embed_model"]
    assert report["total_rows"] == 1


def test_empty_dataset_produces_a_report_not_an_error():
    report = dataset_leakage({"records": []}, CHUNKS, embedder=lambda t: [])
    assert report["scored"] == 0
    assert report["leaked"] == 0
    assert report["max_similarity"] is None


def test_find_leaks_accepts_a_bare_list():
    encode = fake_embedder({"q": UNIT_X, "c": UNIT_X})
    report = find_leaks(
        [{"query_id": "q1", "question": "q", "gold_chunk_id": "c1"}],
        {"c1": {"chunk_id": "c1", "text": "c"}},
        embedder=encode,
    )
    assert report["leaked"] == 1


def test_real_datasets_are_not_leaking():
    """The shipped datasets must pass §10.2 under the real encoder.

    Skipped when the embedding model cannot be loaded, so the arithmetic
    tests above still run offline. A gate that cannot run must not report
    success, which is why this skips rather than passing.
    """
    import json

    chunks_path = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
    if not chunks_path.exists():
        pytest.skip("corpus not present")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    payload = json.loads(
        (ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v4.json")
        .read_text(encoding="utf-8")
    )

    try:
        report = dataset_leakage(payload, chunks)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedder unavailable: {type(exc).__name__}: {str(exc)[:100]}")

    leaked = [d for d in report["detail"] if d["leaked"]]
    assert not leaked, (
        f"{len(leaked)} row(s) in retrieval_eval_dataset_v4.json exceed "
        f"§10.2: " + ", ".join(f"{d['query_id']}({d['similarity']})" for d in leaked[:5])
    )