"""Out-of-corpus abstention tests.

Verifies that questions outside the RAG/LLM domain (biology, medicine, physics,
economics, law, etc.) result in hard abstention (REJECT decision), not
hallucinated answers. This enforces the honest domain scope declared in
``docs/DOMAIN_SCOPE.md``.

The ConfidenceEngine already encodes the correct gates:
  * No chunks at all → confidence 0.0, decision REJECT
  * evidence_relevance < 0.25 → confidence capped below reject threshold → REJECT
  * Weak/mixed-source evidence → confidence capped → REJECT

These tests exercise those gates with out-of-domain queries and low-relevance
retrieved chunks, without needing Qdrant or Groq.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.controller.confidence_engine import ConfidenceEngine  # noqa: E402


# ── Out-of-domain questions (must abstain) ──────────────────────────────────

OUT_OF_CORPUS_QUERIES = [
    "What is the mechanism of CRISPR-Cas9 gene editing in human embryos?",  # biology
    "What was the closing price of Bitcoin in Indonesian rupiah on 15 Oct 2026?",  # finance
    "Who won the 2024 Nobel Prize in Physics?",  # general knowledge / physics
    "What is the standard treatment for stage III non-small-cell lung cancer?",  # medicine
    "How does quantum chromodynamics explain quark confinement?",  # physics
    "What is the GDP growth rate of Argentina in 2025?",  # economics
    "What are the elements of a valid contract under English law?",  # law
]


def _make_chunk(text: str, score: float = 0.05, doc_id: str = "doc_x") -> dict[str, Any]:
    return {
        "chunk_id": f"{doc_id}_c",
        "doc_id": doc_id,
        "text": text,
        "score": score,
        "retrieval_score": score,
        "rerank_score": score,
        "metadata": {"doc_id": doc_id, "source_domain": "rag_llm"},
    }


def _low_relevance_chunks(query: str, n: int = 3) -> list[dict[str, Any]]:
    """Return chunks whose text shares almost no terms with an out-of-domain query."""
    rag_texts = [
        "Retrieval-Augmented Generation combines a retriever with a language model.",
        "The Self-Instruct framework bootstraps instruction-following data.",
        "Llama 2 is a family of large language models trained on public data.",
        "Attention Is All You Need introduces the Transformer architecture.",
        "Rerankers improve retrieval precision by cross-encoding query-document pairs.",
    ]
    return [_make_chunk(t, score=0.03) for t in rag_texts[:n]]


class TestOutOfCorpusAbstention:
    """Hard abstention for out-of-domain questions."""

    def test_no_chunks_always_rejects(self):
        """Zero retrieved chunks → REJECT, never GENERATE."""
        engine = ConfidenceEngine()
        result = engine.calculate_confidence(chunks=[], query="What is CRISPR?")
        assert result["decision"] == "REJECT"
        assert result["confidence_score"] == 0.0
        assert result["signals"]["top_score"] == 0.0

    def test_all_out_of_domain_queries_reject(self):
        """Every out-of-corpus query with low-relevance chunks → REJECT."""
        engine = ConfidenceEngine()
        for query in OUT_OF_CORPUS_QUERIES:
            chunks = _low_relevance_chunks(query)
            result = engine.calculate_confidence(
                chunks=chunks, query=query, query_type="factual"
            )
            assert result["decision"] == "REJECT", (
                f"Query should be rejected (out of corpus) but got {result['decision']}: {query}"
            )
            assert result["confidence_score"] <= engine.reject_threshold, (
                f"Confidence {result['confidence_score']} must be below reject "
                f"threshold {engine.reject_threshold}: {query}"
            )
            # Weak-evidence gate must have fired.
            assert result["signals"]["weak_evidence"] is True, (
                f"weak_evidence must be True for out-of-domain query: {query}"
            )
            assert result["signals"]["evidence_relevance"] is not None
            assert result["signals"]["evidence_relevance"] < 0.25, (
                f"evidence_relevance {result['signals']['evidence_relevance']} "
                f"must be < 0.25 for out-of-domain: {query}"
            )

    def test_out_of_domain_never_generates(self):
        """No out-of-domain query may produce a GENERATE decision."""
        engine = ConfidenceEngine()
        decisions = []
        for query in OUT_OF_CORPUS_QUERIES:
            result = engine.calculate_confidence(
                chunks=_low_relevance_chunks(query), query=query
            )
            decisions.append(result["decision"])
        assert "GENERATE" not in decisions, (
            f"Out-of-domain queries must never GENERATE; got {decisions}"
        )
        assert all(d == "REJECT" for d in decisions)

    def test_in_domain_query_can_generate(self):
        """Sanity check: an in-domain RAG query with relevant chunks can GENERATE."""
        engine = ConfidenceEngine()
        chunks = [
            _make_chunk(
                "Retrieval-Augmented Generation combines a retriever with a "
                "language model to ground answers in external documents.",
                score=0.9,
                doc_id="doc_rag_survey",
            )
        ]
        result = engine.calculate_confidence(
            chunks=chunks,
            query="How does RAG combine retrieval with generation?",
            query_type="factual",
        )
        # High relevance + high score → should not be REJECT
        assert result["decision"] != "REJECT", (
            f"In-domain RAG query should not be rejected; got {result['decision']}"
        )
        assert result["signals"]["evidence_relevance"] is not None
        assert result["signals"]["evidence_relevance"] >= 0.25, (
            f"In-domain query should have evidence_relevance >= 0.25, "
            f"got {result['signals']['evidence_relevance']}"
        )

    def test_empty_chunks_abstain_reason(self):
        """Empty chunk set yields REJECT with no valid scores signal."""
        engine = ConfidenceEngine()
        result = engine.calculate_confidence(chunks=[], query="What is CRISPR?")
        assert result["signals"]["top_score"] == 0.0
        assert result["decision"] == "REJECT"
