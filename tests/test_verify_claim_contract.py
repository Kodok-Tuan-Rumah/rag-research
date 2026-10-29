import asyncio

import pytest

from src.api import main


class FakeRetriever:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def search_with_trace(self, **kwargs):
        outcome = self.outcomes.pop(0) if self.outcomes else []
        return {"results": outcome, "trace": {"latency_ms": {"reranker": 0.0}}}


class FakeConfidenceEngine:
    def __init__(self, decisions):
        self.decisions = decisions

    def calculate_confidence(self, chunks, query_type=None, query=None, **kwargs):
        return self.decisions.pop(0)


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _chunks():
    return [{"chunk_id": "c1", "doc_id": "d1", "text": "multi-head attention projects", "retrieval_score": 0.8, "metadata": {}}]


def test_verify_claim_abstains_when_low_confidence(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([_chunks()]))
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.1, "decision": "REJECT", "calibrated_decision": "REJECT",
         "calibrated_confidence": 0.1, "is_calibrated": False, "signals": {}},
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **e: None)

    resp = asyncio.run(main.verify_claim(main.VerifyClaimRequest(claim="some claim text here")))
    assert resp.decision == "abstain"
    assert resp.confidence_zone == "red"
    assert resp.failure_type == "insufficient_context"
    assert resp.calibrated_confidence == 0.1
    # Abstain mirror F-01: no generator dependency, no span attribution emitted.
    assert resp.span_level_attribution == []


def test_verify_claim_answers_when_high_confidence(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([_chunks()]))
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "calibrated_decision": "GENERATE",
         "calibrated_confidence": 0.85, "is_calibrated": True, "signals": {}},
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **e: None)

    resp = asyncio.run(main.verify_claim(main.VerifyClaimRequest(claim="some claim text here")))
    assert resp.decision == "answer"
    assert resp.confidence_zone == "green"
    assert len(resp.evidence) == 1
    assert resp.failure_type == "none"
    assert resp.calibrated_confidence == 0.85
    assert len(resp.span_level_attribution) == 1
    assert resp.span_level_attribution[0].chunk_id == "c1"


def test_verify_claims_batch_cap(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([_chunks()] * 3))
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([
        {"confidence_score": 0.8, "decision": "GENERATE", "calibrated_decision": "GENERATE",
         "calibrated_confidence": 0.85, "is_calibrated": True, "signals": {}} for _ in range(3)
    ]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(main, "_log_query_event", lambda **e: None)

    resp = asyncio.run(main.verify_claims(main.VerifyClaimsRequest(claims=["claim one text", "claim two text", "claim three text"])))
    assert resp.count == 3
    assert len(resp.results) == 3


def test_verify_claims_rejects_over_cap():
    with pytest.raises(Exception):
        main.VerifyClaimsRequest(claims=["c"] * 51)
