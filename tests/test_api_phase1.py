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

    def search_with_trace(self, **kwargs):
        results = self.search(**kwargs)
        return {"results": results, "trace": {"latency_ms": {"reranker": 0.0}}}


class FakeConfidenceEngine:
    def __init__(self, decisions):
        self.decisions = decisions
        self.query_types = []

    def calculate_confidence(self, chunks, query_type=None, query=None):
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


class FakeCache:
    def __init__(self, data=None):
        self.data = data
        self.set_calls = []
        self.delete_calls = []

    def get(self, key):
        return self.data

    def set(self, key, value, ttl_seconds=300):
        self.set_calls.append((key, value, ttl_seconds))

    def delete(self, key):
        self.delete_calls.append(key)


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


# --- abstain_reason surfacing (PROJECT_GOALS.md §4) -------------------------


def _single_chunk():
    return [{
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "metadata": {},
    }]


def test_api_surfaces_generator_abstain_reason(monkeypatch):
    """The reason must reach the caller, not just the log.

    RAGResponse.reason was added but the API never forwarded it, so an HTTP
    client still could not distinguish a provider rejection from a model
    refusal or a sanity-check rejection.
    """
    reason = ("provider rejected model output: response_format=json_object "
              "validation failed")

    class RejectingGenerator(FakeGenerator):
        def generate(self, query, chunks, **kwargs):
            return type("Response", (), {
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_method": "abstained",
                "model": "test-model",
                "status": "INSUFFICIENT_CONTEXT",
                "confidence_score": 0.0,
                "reason": reason,
                "retrieved_chunks": chunks,
                "context_chunk_ids": [],
            })()

    monkeypatch.setattr(main, "retriever", FakeRetriever([_single_chunk()]))
    monkeypatch.setattr(main, "generator", RejectingGenerator())
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

    assert response.generator_status == "INSUFFICIENT_CONTEXT"
    assert response.failure_type == "insufficient_context"
    assert response.abstain_reason == reason


def test_api_abstain_reason_is_none_when_answered(monkeypatch):
    """A successful answer must not carry a stale abstain reason."""
    monkeypatch.setattr(main, "retriever", FakeRetriever([_single_chunk()]))
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

    assert response.generator_status == "ANSWERED"
    assert response.failure_type == "none"
    assert response.abstain_reason is None


def test_api_records_reason_for_retrieval_miss(monkeypatch):
    # The endpoint retries once when the decision is REJECT, so the retriever
    # stub must serve both attempts.
    monkeypatch.setattr(main, "retriever", FakeRetriever([[], []]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
            {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.failure_type == "retrieval_miss"
    assert response.abstain_reason
    assert "no usable chunks" in response.abstain_reason


def test_api_records_reason_for_pipeline_error(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([_single_chunk()]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
        ]),
    )

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_in_threadpool", _boom)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.failure_type == "pipeline_error"
    assert response.abstain_reason
    assert "boom" in response.abstain_reason


# --- §4 structured log schema on the API adapter ----------------------------

# PROJECT_GOALS.md §4 fixes these keys. The CLI (run_single_query.py) emitted
# them but the API did not, so the adapter serving real traffic was the one
# missing query_type / retrieval_scores / reranker_scores / chunks_used and
# latency_ms.reranker.
REQUIRED_LOG_FIELDS = {
    "query_id",
    "query",
    "query_type",
    "retrieval_scores",
    "reranker_scores",
    "chunks_used",
    "confidence_score",
    "failure_type",
    "answer_status",
    "latency_ms",
    "retry_triggered",
}


def test_api_query_event_carries_full_goals_log_schema(monkeypatch):
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "complete grounded context.",
        "retrieval_score": 0.2,
        "rerank_score": 0.35,
        "metadata": {},
    }
    monkeypatch.setattr(main, "retriever", FakeRetriever([[chunk]]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert len(events) == 1
    event = events[0]
    missing = REQUIRED_LOG_FIELDS - event.keys()
    assert not missing, f"missing §4 log fields: {sorted(missing)}"
    assert set(event["latency_ms"]) == {"retrieval", "reranker", "generation", "total"}
    assert event["query_type"] == "general"
    assert event["answer_status"] == "answered"
    assert event["chunks_used"] == ["c1"]
    assert event["retrieval_scores"] == [0.2]
    assert event["reranker_scores"] == [0.35]


def test_api_event_reports_retriever_reranker_latency(monkeypatch):
    """latency_ms.reranker must come from the retriever trace, not a hardcoded 0."""

    class TracingRetriever(FakeRetriever):
        def search_with_trace(self, **kwargs):
            results = self.search(**kwargs)
            return {"results": results, "trace": {"latency_ms": {"reranker": 12.5}}}

    monkeypatch.setattr(main, "retriever", TracingRetriever([_single_chunk()]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert events[0]["latency_ms"]["reranker"] == 12.5


def test_api_abstain_event_still_carries_full_log_schema(monkeypatch):
    """Abstains are the path §4 cares most about — they must not skip fields."""
    monkeypatch.setattr(main, "retriever", FakeRetriever([[], []]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(
        main,
        "confidence_engine",
        FakeConfidenceEngine([
            {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
            {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
        ]),
    )
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    event = events[0]
    missing = REQUIRED_LOG_FIELDS - event.keys()
    assert not missing, f"missing §4 log fields: {sorted(missing)}"
    assert set(event["latency_ms"]) == {"retrieval", "reranker", "generation", "total"}
    assert event["answer_status"] == "abstained"
    assert event["chunks_used"] == []
    assert event["retrieval_scores"] == []
    assert event["reranker_scores"] == []


def test_redis_cache_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REDIS_ENABLED", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    from src.api.cache import RedisCache

    assert RedisCache().client is None


def test_cache_hit_records_metrics_and_query_log(monkeypatch):
    cached = main.GenerateResponse(
        query="test query",
        answer="cached answer",
        retrieved_chunks=[],
        retrieval_method="cache",
        latency_retrieval_ms=12.0,
        latency_generation_ms=34.0,
        latency_total_ms=46.0,
        model="test-model",
        confidence=0.8,
        decision="GENERATE",
        generator_status="ANSWERED",
    ).model_dump()

    fake_cache = FakeCache(cached)
    retriever = FakeRetriever([])
    monkeypatch.setattr(main, "cache", fake_cache)
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([]))

    metrics = []
    events = []
    monkeypatch.setattr(main, "_record_metrics", lambda **event: metrics.append(event))
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.answer == "cached answer"
    assert response.latency_retrieval_ms == 0.0
    assert response.latency_generation_ms == 0.0
    assert retriever.calls == []
    assert metrics[0]["decision"] == "GENERATE"
    assert metrics[0]["failure_type"] == "none"
    assert events[0]["cache_hit"] is True
    assert events[0]["query_type"] == "cache_hit"
    assert events[0]["answer_status"] == "answered"
    assert fake_cache.set_calls == []


def test_abstention_is_not_cached(monkeypatch):
    class AbstainingGenerator(FakeGenerator):
        def generate(self, query, chunks, **kwargs):
            return type("Response", (), {
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_method": "abstained",
                "model": "test-model",
                "status": "INSUFFICIENT_CONTEXT",
                "confidence_score": 0.0,
                "reason": "sanity rejection",
                "retrieved_chunks": chunks,
                "context_chunk_ids": [],
            })()

    fake_cache = FakeCache()
    monkeypatch.setattr(main, "cache", fake_cache)
    monkeypatch.setattr(main, "retriever", FakeRetriever([_single_chunk()]))
    monkeypatch.setattr(main, "generator", AbstainingGenerator())
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

    assert response.generator_status == "INSUFFICIENT_CONTEXT"
    assert fake_cache.set_calls == []
    assert fake_cache.delete_calls == []


def test_successful_answer_is_cached(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr(main, "cache", fake_cache)
    monkeypatch.setattr(main, "retriever", FakeRetriever([_single_chunk()]))
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

    assert response.decision == "GENERATE"
    assert len(fake_cache.set_calls) == 1
    assert fake_cache.set_calls[0][2] == 600
    assert fake_cache.delete_calls == []
