from __future__ import annotations

import asyncio

from src.retrieval.tools import RetrievalTools


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return []


def test_hybrid_tool_defaults_to_rrf_only_but_keeps_reranker_opt_in():
    retriever = FakeRetriever()
    tools = RetrievalTools(
        vector_store=None,
        retriever=retriever,
        confidence_engine=None,
        generator=None,
    )

    asyncio.run(tools.search_hybrid("question"))
    asyncio.run(tools.search_hybrid("question", rerank=True))

    assert retriever.calls[0][1]["rerank"] is False
    assert retriever.calls[1][1]["rerank"] is True


def test_assess_retrieval_quality_passes_query_type():
    class Confidence:
        def __init__(self):
            self.query_type = None

        def calculate_confidence(self, chunks, query_type=None):
            self.query_type = query_type
            return {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}}

    confidence = Confidence()
    tools = RetrievalTools(
        vector_store=None,
        retriever=FakeRetriever(),
        confidence_engine=confidence,
        generator=None,
    )

    result = asyncio.run(
        tools.assess_retrieval_quality([], "Why does this happen?", query_type="reasoning")
    )

    assert confidence.query_type == "reasoning"
    assert result["confidence"]["decision"] == "GENERATE"


def test_generate_answer_passes_retrieval_confidence_contract():
    class Generator:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {
                "status": "INSUFFICIENT_CONTEXT",
                "confidence_score": 0.0,
            })()

    generator = Generator()
    tools = RetrievalTools(
        vector_store=None,
        retriever=FakeRetriever(),
        confidence_engine=None,
        generator=generator,
    )

    asyncio.run(tools.generate_answer(
        "question",
        [],
        query_type="factual",
        retrieval_decision="REJECT",
        retrieval_confidence=0.2,
    ))

    assert generator.kwargs["query_type"] == "factual"
    assert generator.kwargs["retrieval_decision"] == "REJECT"
    assert generator.kwargs["retrieval_confidence"] == 0.2
