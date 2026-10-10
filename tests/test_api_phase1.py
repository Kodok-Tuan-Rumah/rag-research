import asyncio

from src.api import main


class FakeRetriever:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeConfidenceEngine:
    def __init__(self, decisions):
        self.decisions = decisions

    def calculate_confidence(self, chunks):
        return self.decisions.pop(0)


class FakeGenerator:
    def generate(self, query, chunks):
        return type("Response", (), {
            "answer": "grounded answer",
            "retrieval_method": "hybrid_rrf_rerank",
            "model": "test-model",
            "status": "ANSWERED",
            "confidence_score": 0.77,
            "llm_confidence_score": 0.64,
            "retrieved_chunks": chunks,
            "context_chunk_ids": [chunks[0]["chunk_id"]] if chunks else [],
        })()


async def _run_inline(func, *args, **kwargs):
    """Keep endpoint tests deterministic; threadpool behavior is FastAPI-owned."""
    return func(*args, **kwargs)


def test_generate_request_default_top_k_comes_from_settings():
    request = main.GenerateRequest(query="test query")
    assert request.top_k == main.settings.default_top_k
    assert request.use_reranker is False


def test_api_defaults_to_rrf_only(monkeypatch):
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "ranking_mode": "rrf_only",
        "metadata": {},
    }
    retriever = FakeRetriever([[chunk]])
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert retriever.calls[0]["rerank"] is False
    assert response.retrieved_chunks[0].ranking_mode == "rrf_only"


def test_health_reports_actual_reranker_load_state(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        main,
        "store",
        SimpleNamespace(
            client=SimpleNamespace(count=lambda collection: SimpleNamespace(count=7)),
            collection_name="test",
        ),
    )
    monkeypatch.setattr(main, "retriever", SimpleNamespace(reranker=None))

    response = asyncio.run(main.health())

    assert response["reranker_model_loaded"] is False
    assert response["reranker_default_enabled"] is main.DEFAULT_USE_RERANKER
    assert response["reranker"] is False


def test_api_explicitly_opted_in_reranker_flag_is_forwarded(monkeypatch):
    retriever = FakeRetriever([
        [{"chunk_id": "c1", "doc_id": "d1", "text": "complete grounded context.", "retrieval_score": 0.2, "metadata": {}}],
    ])
    confidence = FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ])
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", confidence)
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    asyncio.run(
        main.generate_answer(
            main.GenerateRequest(query="test query", use_reranker=True)
        )
    )

    assert retriever.calls[0]["rerank"] is True


def test_api_retries_once_with_bm25_heavy_weights(monkeypatch):
    retriever = FakeRetriever([
        [],
        [{"chunk_id": "c1", "doc_id": "d1", "text": "complete grounded context.", "retrieval_score": 0.2, "metadata": {}}],
    ])
    confidence = FakeConfidenceEngine([
        {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ])
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", confidence)
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.retry_triggered is True
    assert response.failure_type == "none"
    assert len(retriever.calls) == 2
    assert retriever.calls[0]["rerank"] is False
    assert retriever.calls[1]["dense_weight"] == 0.3
    assert retriever.calls[1]["bm25_weight"] == 0.7
    assert len(events) == 1
    assert events[0]["query"] == "test query"
    assert events[0]["answer_status"] == "answered"
    assert events[0]["retry_triggered"] is True
    assert response.generation_chunk_ids == ["c1"]
    assert response.context_chunk_ids == ["c1"]
    assert response.generator_status == "ANSWERED"


def test_api_propagates_generator_abstention_and_context_provenance(monkeypatch):
    class AbstainingGenerator(FakeGenerator):
        def generate(self, query, chunks):
            return type("Response", (), {
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_method": "hybrid_rrf_rerank",
                "model": "test-model",
                "status": "INSUFFICIENT_CONTEXT",
                "confidence_score": 0.12,
                "llm_confidence_score": 0.1,
                "retrieved_chunks": chunks[:1],
                "context_chunk_ids": [],
            })()

    chunk = {
        "chunk_id": "c1", "doc_id": "d1", "text": "complete grounded context.",
        "retrieval_score": 0.2, "metadata": {},
    }
    monkeypatch.setattr(main, "retriever", FakeRetriever([[chunk]]))
    monkeypatch.setattr(main, "generator", AbstainingGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.generator_status == "INSUFFICIENT_CONTEXT"
    assert response.generator_confidence == 0.12
    assert response.llm_confidence == 0.1
    assert response.generation_chunk_ids == ["c1"]
    assert response.context_chunk_ids == []
    assert events[0]["answer_status"] == "abstained"
    assert events[0]["failure_type"] == "insufficient_context"


def test_generate_context_returns_only_chunk_ids_included_before_limit():
    from src.generation.generator import build_context_with_ids

    chunks = [
        {"chunk_id": "included", "doc_id": "d1", "text": "short context", "metadata": {}},
        {"chunk_id": "too-large", "doc_id": "d2", "text": "x" * 100, "metadata": {}},
        {"chunk_id": "never-reached", "doc_id": "d3", "text": "later context", "metadata": {}},
    ]

    context, chunk_ids = build_context_with_ids(chunks, max_chars=100)

    assert "included" in context
    assert "too-large" not in context
    assert chunk_ids == ["included"]


def test_generator_abstention_retains_context_provenance():
    from src.generation.generator import _make_abstain_response

    response = _make_abstain_response(
        query="test",
        chunks=[],
        model="test-model",
        reason="sanity rejection",
        context_used="[Source 1 | d1 | c1 | Section: body]\nEvidence.",
        context_chunk_ids=["c1"],
    )

    assert response.status == "INSUFFICIENT_CONTEXT"
    assert response.context_chunk_ids == ["c1"]
    assert "c1" in response.context_used


def test_api_converts_pipeline_exception_to_safe_abstain(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([RuntimeError("backend unavailable")]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.decision == "REJECT"
    assert response.failure_type == "pipeline_error"
    assert response.retry_triggered is False
    assert events[0]["answer_status"] == "abstained"
    assert events[0]["failure_type"] == "pipeline_error"
