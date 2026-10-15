import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controller.confidence_engine import ConfidenceEngine


def _engine():
    return ConfidenceEngine()


def test_score_type_is_rrf_when_no_rerank_scores():
    engine = _engine()
    chunks = [{"retrieval_score": 0.01, "rerank_score": None, "metadata": {"doc_id": "d"}} for _ in range(3)]
    result = engine.calculate_confidence(chunks)
    assert result["signals"]["score_type"] == "rrf"


def test_score_type_is_cross_encoder_when_rerank_present():
    engine = _engine()
    chunks = [{"retrieval_score": 0.01, "rerank_score": 2.5, "metadata": {"doc_id": "d"}} for _ in range(3)]
    result = engine.calculate_confidence(chunks)
    assert result["signals"]["score_type"] == "cross_encoder"


def test_same_numeric_score_normalizes_differently_by_score_space():
    engine = _engine()
    chunks_ce = [{"rerank_score": -5.0, "metadata": {"doc_id": "d"}} for _ in range(3)]
    chunks_rrf = [{"retrieval_score": -5.0, "metadata": {"doc_id": "d"}} for _ in range(3)]
    ce = engine.calculate_confidence(chunks_ce)
    rrf = engine.calculate_confidence(chunks_rrf)
    # cross-encoder: sigmoid(-5) ≈ 0.0067 (rendah). RRF: arr/max = 1.0.
    assert ce["signals"]["score_type"] == "cross_encoder"
    assert rrf["signals"]["score_type"] == "rrf"
    assert ce["confidence_score"] != rrf["confidence_score"]


def test_get_chunk_score_precedence():
    engine = _engine()
    chunk = {"rerank_score": 1.2, "retrieval_score": 0.3, "score": 9.9}
    score, kind = engine._select_chunk_score(chunk)
    assert (score, kind) == (1.2, "cross_encoder")
    chunk2 = {"retrieval_score": 0.3, "score": 9.9}
    assert engine._select_chunk_score(chunk2) == (0.3, "rrf")
    chunk3 = {"score": 9.9}
    assert engine._select_chunk_score(chunk3) == (9.9, "rrf")


def test_confidence_stays_within_bounds():
    engine = _engine()
    chunks = [{"rerank_score": float(i), "metadata": {"doc_id": f"d{i}"}} for i in range(10)]
    result = engine.calculate_confidence(chunks)
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_agreement_uses_top_level_doc_id():
    engine = _engine()
    chunks = [
        {"doc_id": "same-doc", "retrieval_score": 0.1, "metadata": {}},
        {"doc_id": "same-doc", "retrieval_score": 0.08, "metadata": {}},
        {"doc_id": "same-doc", "retrieval_score": 0.06, "metadata": {}},
    ]

    assert engine.calculate_confidence(chunks)["signals"]["agreement"] == 1.0


def test_invalid_scores_do_not_shift_valid_chunk_alignment():
    engine = _engine()
    chunks = [
        {"doc_id": "invalid-doc", "retrieval_score": float("nan"), "metadata": {}},
        {"doc_id": "same-doc", "retrieval_score": 0.1, "metadata": {}},
        {"doc_id": "same-doc", "retrieval_score": 0.08, "metadata": {}},
    ]

    assert engine.calculate_confidence(chunks)["signals"]["agreement"] == 1.0


def test_broad_question_with_mixed_sources_cannot_reach_generate():
    engine = _engine()
    chunks = [
        {
            "rerank_score": score,
            "doc_id": doc,
            "text": "This paper discusses performance measurements and evaluation results.",
        }
        for score, doc in ((5.0, "paper-a"), (4.0, "paper-b"), (3.0, "paper-c"))
    ]

    result = engine.calculate_confidence(
        chunks,
        answer="The paper discusses performance measurements and evaluation results.",
        contexts=[chunk["text"] for chunk in chunks],
        query="What does the paper say about performance?",
    )

    assert result["signals"]["broad_topic_with_mixed_sources"] is True
    assert result["decision"] == "PARTIAL_TRUST"


def test_lexically_unrelated_evidence_cannot_reach_generate():
    engine = _engine()
    chunks = [
        {
            "rerank_score": score,
            "doc_id": "same-paper",
            "text": "The training corpus contains examples of language model prompts.",
        }
        for score in (5.0, 4.0, 3.0)
    ]

    result = engine.calculate_confidence(
        chunks,
        query="How does the authors calculate inter rater reliability?",
    )

    assert result["signals"]["evidence_relevance"] < 0.25
    assert result["signals"]["weak_evidence"] is True
    assert result["decision"] != "GENERATE"
