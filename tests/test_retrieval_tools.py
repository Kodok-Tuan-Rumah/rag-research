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
