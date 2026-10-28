import asyncio

from src.api import main


class FakeRetriever:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.outcomes.pop(0)

    def search_with_trace(self, **kwargs):
        results = self.search(**kwargs)
        return {"results": results, "trace": {"latency_ms": {"reranker": 0.0}}}


class FakeConfidenceEngine:
    def __init__(self, decisions):
        self.decisions = decisions

    def calculate_confidence(self, chunks, query_type=None, query=None):
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
    return func(*args, **kwargs)


def test_api_calibration_fields_present(monkeypatch):
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "ctx", "retrieval_score": 0.2, "metadata": {}}]
    monkeypatch.setattr(main, "retriever", FakeRetriever([chunks]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.calibrated_confidence is not None
    assert response.calibrated_decision is not None
    assert response.is_calibrated is False
    assert response.calibration_method == "heuristic_uncalibrated"


def test_api_calibration_abstain_zero(monkeypatch):
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "ctx", "retrieval_score": 0.2, "metadata": {}}]
    monkeypatch.setattr(main, "retriever", FakeRetriever([chunks]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.2, "decision": "REJECT", "signals": {}},
        {"confidence_score": 0.2, "decision": "REJECT", "signals": {}},
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **event: None)

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))
    assert response.decision == "REJECT"
    assert response.calibrated_confidence == 0.0
    assert response.calibrated_decision == "REJECT"
