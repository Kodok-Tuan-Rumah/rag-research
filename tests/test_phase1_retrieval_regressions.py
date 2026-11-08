from __future__ import annotations

import pytest

from src.retrieval.rerank_order import rank_reranked_chunks
from src.controller.confidence_engine import ConfidenceEngine


@pytest.mark.parametrize(
    ("query_id", "chunks", "expected_first"),
    [
        (
            "p1_answer_11",
            [
                {
                    "chunk_id": "paper-title-body",
                    "rerank_score": 0.0217,
                    "final_score": 0.2217,
                },
                {
                    "chunk_id": "answer-bearing-abstract",
                    "rerank_score": 0.0453,
                    "final_score": 0.1453,
                },
            ],
            "answer-bearing-abstract",
        ),
        (
            "p1_answer_13",
            [
                {
                    "chunk_id": "generic-body",
                    "rerank_score": -0.0026,
                    "final_score": 0.1974,
                },
                {
                    "chunk_id": "answer-bearing-retrieval",
                    "rerank_score": 0.0797,
                    "final_score": 0.0797,
                },
            ],
            "answer-bearing-retrieval",
        ),
    ],
)
def test_phase1_reranker_order_uses_cross_encoder_score(query_id, chunks, expected_first):
    ranked = rank_reranked_chunks(chunks)

    assert ranked[0]["chunk_id"] == expected_first, query_id


@pytest.mark.parametrize(
    ("query_id", "query", "query_type", "chunks"),
    [
        (
            "p1_answer_11",
            (
                "In Large Language Models Can Self-Improve, how many parameters "
                "did the experimental LLM have?"
            ),
            "reasoning",
            [
                {
                    "doc_id": "llama2",
                    "text": "Llama 2 models range from 7 to 70 billion parameters.",
                    "rerank_score": 0.0456,
                },
                {
                    "doc_id": "self_improve",
                    "text": (
                        "The experimental LLM has 540 billion parameters. "
                        "Large Language Models Can Self-Improve."
                    ),
                    "rerank_score": 0.0453,
                },
                {
                    "doc_id": "self_improve",
                    "text": "Large language models can improve reasoning on benchmark tasks.",
                    "rerank_score": 0.0427,
                },
            ],
        ),
        (
            "p1_answer_13",
            (
                "What component is directly substituted to generate relevant "
                "documents according to a query?"
            ),
            "general",
            [
                {
                    "doc_id": "rag_survey",
                    "text": (
                        "The retriever module is directly substituted to generate "
                        "relevant documents according to a query."
                    ),
                    "rerank_score": 0.0797,
                },
                {
                    "doc_id": "rag_survey",
                    "text": "Query expansion enriches a query with further context.",
                    "rerank_score": 0.0089,
                },
                {
                    "doc_id": "rag_survey",
                    "text": "Naive RAG has indexing, retrieval, and generation stages.",
                    "rerank_score": 0.001,
                },
            ],
        ),
    ],
)
def test_phase1_answer_cases_with_strong_evidence_remain_reviewable(
    query_id, query, query_type, chunks
):
    engine = object.__new__(ConfidenceEngine)
    engine.reject_threshold = 0.25
    engine.partial_threshold = 0.50
    result = engine.calculate_confidence(
        chunks,
        query=query,
        query_type=query_type,
    )

    assert result["signals"]["evidence_relevance"] >= 0.8, query_id
    assert result["decision"] == "PARTIAL_TRUST", query_id
