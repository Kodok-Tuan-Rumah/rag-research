import asyncio

import pytest

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
        self.query_types = []

    def calculate_confidence(self, chunks, query_type=None):
        self.query_types.append(query_type)
        return self.decisions.pop(0)


class FakeGenerator:
    def generate(self, query, chunks, **kwargs):
        return type("Response", (), {
            "answer": "grounded answer",
            "retrieval_method": "hybrid_rrf_rerank",
            "model": "test-model",
            "status": "ANSWERED",
            "confidence_score": 0.77,
            "post_generation_confidence_score": 0.72,
            "post_generation_decision": "GENERATE",
            "llm_confidence_score": 0.64,
            "retrieved_chunks": chunks,
            "context_chunk_ids": [chunks[0]["chunk_id"]] if chunks else [],
        })()


async def _run_inline(func, *args, **kwargs):
    """Keep endpoint tests deterministic; threadpool behavior is FastAPI-owned."""
    return func(*args, **kwargs)


def test_generate_request_leaves_top_k_to_query_policy_when_omitted():
    request = main.GenerateRequest(query="test query")
    assert request.top_k is None
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


def test_api_passes_policy_query_type_to_confidence_engine(monkeypatch):
    confidence = FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ])
    monkeypatch.setattr(
        main,
        "retriever",
        FakeRetriever([[
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "complete grounded context.",
                "retrieval_score": 0.2,
                "metadata": {},
            }
        ]]),
    )
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", confidence)
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    asyncio.run(main.generate_answer(main.GenerateRequest(query="Who authored this paper?")))

    assert confidence.query_types == ["factual"]


@pytest.mark.parametrize(
    ("query", "expected_type", "expected_top_k", "expected_max_tokens", "expected_temperature"),
    [
        ("Who authored this paper?", "factual", 5, 256, 0.0),
        ("How does this method work?", "reasoning", 10, 768, 0.3),
        ("test query", "general", 8, 512, 0.2),
    ],
)
def test_api_applies_shared_query_policy_to_retrieval_and_generation(
    monkeypatch, query, expected_type, expected_top_k, expected_max_tokens, expected_temperature
):
    chunks = [{
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "metadata": {},
    }]
    retriever = FakeRetriever([chunks])

    class RecordingGenerator(FakeGenerator):
        def __init__(self):
            self.kwargs = None

        def generate(self, query, chunks, **kwargs):
            self.kwargs = kwargs
            return super().generate(query, chunks)

    generator = RecordingGenerator()
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", generator)
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    asyncio.run(main.generate_answer(main.GenerateRequest(query=query)))

    search = retriever.calls[0]
    assert search["top_k"] == expected_top_k
    assert search["allow_references"] == (expected_type == "factual")
    assert search["use_multi_query"] is main.settings.use_multi_query
    assert search["use_hyde"] is main.settings.use_hyde
    assert generator.kwargs["query_type"] == expected_type
    assert generator.kwargs["max_tokens"] == expected_max_tokens
    assert generator.kwargs["temperature"] == expected_temperature


def test_api_explicit_top_k_overrides_query_policy(monkeypatch):
    chunks = [{
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "metadata": {},
    }]
    retriever = FakeRetriever([chunks])
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

    asyncio.run(
        main.generate_answer(
            main.GenerateRequest(query="Who authored this paper?", top_k=7)
        )
    )

    assert retriever.calls[0]["top_k"] == 7


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


@pytest.mark.parametrize(
    ("decision", "confidence_score", "expected_chunk_count", "expected_prefix"),
    [
        ("PARTIAL_TRUST", 0.5, 3, "[Low confidence — verify independently] "),
        ("GENERATE", 0.8, 4, ""),
    ],
)
def test_api_generation_decision_contract(
    monkeypatch, decision, confidence_score, expected_chunk_count, expected_prefix
):
    chunks = [
        {
            "chunk_id": f"c{i}",
            "doc_id": f"d{i}",
            "text": f"grounded context chunk {i}.",
            "retrieval_score": 0.2 - i * 0.01,
            "metadata": {},
        }
        for i in range(1, 5)
    ]

    class RecordingGenerator(FakeGenerator):
        def __init__(self):
            self.calls = []

        def generate(self, query, chunks, **kwargs):
            self.calls.append((query, chunks, kwargs))
            return super().generate(query, chunks)

    recording_generator = RecordingGenerator()
    monkeypatch.setattr(main, "retriever", FakeRetriever([chunks]))
    monkeypatch.setattr(main, "generator", recording_generator)
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": confidence_score, "decision": decision, "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.decision == decision
    assert len(recording_generator.calls) == 1
    assert len(recording_generator.calls[0][1]) == expected_chunk_count
    assert recording_generator.calls[0][2]["retrieval_decision"] == decision
    assert recording_generator.calls[0][2]["retrieval_confidence"] == confidence_score
    assert recording_generator.calls[0][2]["max_tokens"] == 512
    assert recording_generator.calls[0][2]["temperature"] == 0.2
    assert response.answer.startswith(expected_prefix)
    assert len(response.retrieved_chunks) == expected_chunk_count
    assert response.retrieval_confidence == confidence_score
    assert response.post_generation_confidence == 0.72
    assert response.post_generation_decision == "GENERATE"


def test_api_rejects_after_bounded_retry_without_calling_generator(monkeypatch):
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "metadata": {},
    }

    class RecordingGenerator(FakeGenerator):
        def __init__(self):
            self.calls = []

        def generate(self, query, chunks, **kwargs):
            self.calls.append((query, chunks))
            return super().generate(query, chunks)

    recording_generator = RecordingGenerator()
    retriever = FakeRetriever([[chunk], [chunk]])
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", recording_generator)
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.2, "decision": "REJECT", "signals": {}},
            {"confidence_score": 0.2, "decision": "REJECT", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.decision == "REJECT"
    assert response.failure_type == "low_confidence"
    assert response.retry_triggered is True
    assert len(retriever.calls) == 2
    assert len(recording_generator.calls) == 0


def test_api_propagates_generator_abstention_and_context_provenance(monkeypatch):
    class AbstainingGenerator(FakeGenerator):
        def generate(self, query, chunks, **kwargs):
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
