import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controller.confidence_engine import ConfidenceEngine
from src.controller.policy_engine import PolicyEngine
from scripts import run_single_query as runner
from scripts.run_single_query import should_skip_reranker
from src.generation.generator import (
    GroqGenerator,
    _make_abstain_response,
    check_answer_evidence,
    sanity_check_answer,
    should_abort_before_generation,
)


def test_sanity_check_rejects_verbatim_reference_dump():
    answer = "This is a long answer that repeats the chunk content exactly and should not be accepted as a real answer because it is a copy of the retrieved text. " * 4
    chunk = {"text": "This is a long answer that repeats the chunk content exactly and should not be accepted as a real answer because it is a copy of the retrieved text."}

    ok, reason = sanity_check_answer(answer, [chunk])

    assert ok is False
    assert "reference_dump" in reason.lower()


def test_answer_evidence_check_rejects_partial_topic_match():
    answer = (
        "HateBERT reported 94 percent accuracy on the benchmark and outperformed "
        "all baseline systems."
    )
    chunks = [{"text": "The study evaluates toxicity classification using HateBERT."}]

    ok, reason = check_answer_evidence(answer, chunks)

    assert ok is False
    assert reason.startswith("insufficient_evidence:")


def test_generator_abstains_with_reason_for_unsupported_specific_answer():
    chunk = {
        "chunk_id": "c1",
        "doc_id": "paper-a",
        "text": "The study evaluates toxicity classification using HateBERT.",
        "retrieval_score": 0.8,
        "metadata": {"section": "body"},
    }

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    '{"answer":"HateBERT reported 94 percent accuracy on the benchmark '
                    'and outperformed all baseline systems.","status":"ANSWERED",'
                    '"confidence_score":0.99,"supporting_sources":["[Source 1]"]}'
                )
            ))])

    class HighConfidence:
        def calculate_confidence(self, chunks, **kwargs):
            return {"confidence_score": 0.95, "decision": "GENERATE", "signals": {}}

    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model = "test-model"
    generator.client = Client()
    generator._confidence = HighConfidence()

    result = generator.generate(
        "What was HateBERT's benchmark accuracy?",
        [chunk],
        retrieval_decision="GENERATE",
        retrieval_confidence=0.95,
    )

    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.answer == "INSUFFICIENT_CONTEXT"
    assert result.reason.startswith("insufficient_evidence:")


@pytest.mark.parametrize(
    "query",
    [
        "What does the paper say about performance?",
        "Tell me about performance.",
        "What about model performance?",
    ],
    ids=["p1_abstain_04", "tell-me-about", "what-about"],
)
def test_broad_topic_query_with_mixed_relevant_context_abstains(query):
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "doc_id": doc_id,
            "text": text,
            "retrieval_score": score,
            "metadata": {"section": "body"},
        }
        for index, (doc_id, text, score) in enumerate([
            (
                "paper-a",
                "The paper describes performance evaluation of retrieval systems "
                "using recall and ranking metrics.",
                0.030,
            ),
            (
                "paper-b",
                "Model benchmarks report accuracy on knowledge and reasoning tasks "
                "across datasets.",
                0.020,
            ),
            (
                "paper-c",
                "The results section compares performance across several datasets "
                "and model configurations.",
                0.010,
            ),
        ], 1)
    ]

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    '{"answer":"Performance is evaluated with recall metrics across datasets.",'
                    '"status":"ANSWERED","confidence_score":0.99,'
                    '"supporting_sources":["[Source 1]"]}'
                )
            ))])

    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model = "test-model"
    generator.client = Client()
    generator._confidence = ConfidenceEngine()

    result = generator.generate(
        query,
        chunks,
        retrieval_decision="GENERATE",
        retrieval_confidence=0.9,
    )

    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.answer == "INSUFFICIENT_CONTEXT"
    assert "broad topic question" in result.reason


def test_confidence_engine_rejects_non_finite_scores():
    engine = ConfidenceEngine()
    chunks = [{"retrieval_score": float("nan")}, {"retrieval_score": 0.1}]

    result = engine.calculate_confidence(chunks)

    # Dokumentasi current behavior (Fix 3): chunk skor invalid di-skip,
    # confidence tetap dihitung dari chunk yang valid, tanpa crash.
    assert math.isfinite(result["confidence_score"])
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert result["decision"] in {"REJECT", "PARTIAL_TRUST", "GENERATE"}


def test_confidence_engine_rejects_batch_with_only_invalid_scores():
    engine = ConfidenceEngine()
    chunks = [{"retrieval_score": float("nan")}, {"retrieval_score": float("inf")}]

    result = engine.calculate_confidence(chunks)

    assert result["decision"] == "REJECT"
    assert result["confidence_score"] == 0.0


def test_confidence_engine_prefers_reranker_score_when_available():
    engine = ConfidenceEngine()
    chunk = {"retrieval_score": 0.01, "rerank_score": 2.5}

    assert engine._get_chunk_score(chunk) == 2.5


def test_confidence_decision_uses_configured_thresholds():
    engine = ConfidenceEngine()

    assert engine._decision_for_score(0.3499) == "REJECT"
    assert engine._decision_for_score(0.35) == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.5555) == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.65) == "GENERATE"
    assert engine._decision_for_score(0.39, query_type="factual") == "REJECT"
    assert engine._decision_for_score(0.4, query_type="factual") == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.3, query_type="reasoning") == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.6, query_type="reasoning") == "GENERATE"


def test_make_abstain_response_is_structured_and_safe():
    response = _make_abstain_response(
        query="What is the answer?",
        chunks=[{"chunk_id": "c1", "text": "context"}],
        model="test-model",
        reason="low confidence",
    )

    assert response.status == "INSUFFICIENT_CONTEXT"
    assert response.answer == "INSUFFICIENT_CONTEXT"
    assert response.confidence_score == 0.0
    assert response.supporting_sources == []


def test_should_abort_before_generation_only_for_truly_empty_or_very_low_confidence_context():
    assert should_abort_before_generation(confidence_score=0.2, decision="REJECT", has_chunks=True) is False
    assert should_abort_before_generation(confidence_score=0.05, decision="REJECT", has_chunks=True) is True
    assert should_abort_before_generation(confidence_score=0.2, decision="PARTIAL_TRUST", has_chunks=True) is False
    assert should_abort_before_generation(confidence_score=0.0, decision="GENERATE", has_chunks=True) is False
    assert should_abort_before_generation(confidence_score=0.0, decision="REJECT", has_chunks=False) is True


def test_generator_honors_retrieval_reject_before_llm_call():
    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            self.calls += 1
            raise AssertionError("LLM must not be called for retrieval REJECT")

    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model = "test-model"
    generator.client = Client()
    generator._confidence = ConfidenceEngine()

    result = generator.generate(
        "What is this?",
        [{"chunk_id": "c1", "text": "retrieved text", "retrieval_score": 0.2}],
        retrieval_decision="REJECT",
        retrieval_confidence=0.2,
    )

    assert generator.client.calls == 0
    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.retrieval_decision == "REJECT"
    assert result.retrieval_confidence_score == 0.2
    assert result.post_generation_confidence_score is None


def test_generator_limits_partial_trust_to_three_chunks_and_separates_confidence():
    chunks = [
        {
            "chunk_id": f"c{i}",
            "doc_id": f"d{i}",
            "text": f"Grounding evidence for source {i} supports this answer clearly.",
            "retrieval_score": 0.8 - i * 0.01,
            "metadata": {},
        }
        for i in range(1, 5)
    ]

    class Client:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"answer":"Evidence supports this answer.","status":"ANSWERED","confidence_score":0.91,"supporting_sources":["[Source 1]"]}')
                )]
            )

    class PostGenerationConfidence:
        def __init__(self):
            self.chunks = None

        def calculate_confidence(self, chunks, **kwargs):
            self.chunks = chunks
            return {
                "confidence_score": 0.58,
                "decision": "PARTIAL_TRUST",
                "signals": {},
            }

    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model = "test-model"
    generator.client = Client()
    generator._confidence = PostGenerationConfidence()

    result = generator.generate(
        "What is this?",
        chunks,
        retrieval_decision="PARTIAL_TRUST",
        retrieval_confidence=0.52,
    )

    assert len(generator.client.calls) == 1
    assert "[Source 4 |" not in generator.client.calls[0]["messages"][1]["content"]
    assert len(generator._confidence.chunks) == 3
    assert len(result.retrieved_chunks) == 3
    assert result.retrieval_confidence_score == 0.52
    assert result.retrieval_decision == "PARTIAL_TRUST"
    assert result.post_generation_confidence_score == 0.58
    assert result.post_generation_decision == "PARTIAL_TRUST"
    assert result.confidence_score == 0.58


def test_should_skip_reranker_for_small_or_weak_candidates():
    assert should_skip_reranker(candidate_count=2, top_gap=0.001, elapsed_ms=100.0) is True
    assert should_skip_reranker(candidate_count=6, top_gap=0.01, elapsed_ms=100.0) is True
    assert should_skip_reranker(candidate_count=6, top_gap=0.001, elapsed_ms=100.0) is False


def test_general_generation_policy_allows_json_response_budget():
    policy = PolicyEngine().generation_policy("general")

    assert policy["max_tokens"] >= 512


def test_factual_escalation_retries_once_and_emits_log_schema(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    context = "Attention lets a model focus on relevant tokens. " * 4
    initial_chunks = [{
        "chunk_id": "initial",
        "doc_id": "doc-1",
        "text": context,
        "score": 0.8,
        "retrieval_score": 0.8,
        "metadata": {"section": "body"},
    }]
    retry_chunks = [{
        **initial_chunks[0],
        "chunk_id": "retry",
    }]

    class FakeRetriever:
        def __init__(self):
            self.retry_calls = []
            self.chunks = []
            self.vector_store = SimpleNamespace(
                search=lambda query, k: initial_chunks
            )

        def dense_search(self, query, k):
            return initial_chunks

        def search(self, query, **kwargs):
            self.retry_calls.append(kwargs)
            return retry_chunks

    class FakeGenerator:
        def __init__(self):
            self.calls = 0
            self.responses = [
                SimpleNamespace(
                    status="INSUFFICIENT_CONTEXT",
                    answer="not enough context",
                    confidence_score=0.2,
                    supporting_sources=[],
                ),
                SimpleNamespace(
                    status="ANSWERED",
                    answer="Attention focuses on relevant tokens.",
                    confidence_score=0.8,
                    supporting_sources=["[Source 1]"],
                ),
            ]

        def generate(self, *args, **kwargs):
            self.calls += 1
            return self.responses.pop(0)

    class FakeConfidenceEngine:
        def calculate_confidence(self, chunks, **kwargs):
            return {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}}

    class FakePolicyEngine:
        def resolve(self, query):
            return {
                "query_type": "factual",
                "retrieval": {"top_k": 3, "use_hyde": False, "use_multi_query": False},
                "reference": {"allow_references": False, "max_ref_ratio": 0.0},
                "generation": {"max_tokens": 64, "temperature": 0.0},
            }

    retriever = FakeRetriever()
    generator = FakeGenerator()
    overlap_scores = iter([0.1, 0.8, 0.8])
    fake_store = SimpleNamespace(
        require_nonempty_index=lambda **kwargs: {
            "backend": "test",
            "collection": "test",
            "vector_count": 796,
            "endpoint": None,
            "storage_path": None,
            "probe_hit_count": 1,
            "probe_chunk_ids": ["initial"],
        }
    )
    monkeypatch.setattr(runner, "QdrantVectorStore", lambda: fake_store)
    monkeypatch.setattr(runner, "MasterHybridRetriever", lambda vector_store: retriever)
    monkeypatch.setattr(runner, "GroqGenerator", lambda model: generator)
    monkeypatch.setattr(runner, "Groq", lambda **kwargs: object())
    monkeypatch.setattr(runner, "ConfidenceEngine", FakeConfidenceEngine)
    monkeypatch.setattr(runner, "PolicyEngine", FakePolicyEngine)
    monkeypatch.setattr(
        runner,
        "compute_context_overlap",
        lambda answer, contexts: next(overlap_scores),
    )

    result = runner.run_single_query(
        query="What does attention do?",
        mode="baseline",
        save_output=False,
    )

    assert generator.calls == 2
    assert len(retriever.retry_calls) == 1
    assert retriever.retry_calls[0]["bm25_weight"] == 0.7
    assert retriever.retry_calls[0]["dense_weight"] == 0.3
    assert result["retry_triggered"] is True
    assert result["config"]["retries_used"] == 1

    required_log_fields = {
        "query_id",
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
    missing_fields = required_log_fields - result.keys()
    assert not missing_fields, f"Missing required log fields: {sorted(missing_fields)}"
    assert result["answer_status"] == "answered"
    assert result["chunks_used"] == ["retry"]
    assert result["retrieval_scores"] == [0.8]
    assert result["reranker_scores"] == [None]
    assert set(result["latency_ms"]) == {"retrieval", "reranker", "generation", "total"}


# --- sanity_check_answer: the 5 cases §7 Phase 1 names -----------------------

_LONG_CHUNK = (
    "Attention mechanisms let a model compare query representations with key representations, "
    "produce normalized weights, and use those weights to combine value representations into a "
    "context vector. This helps the model focus on relevant tokens when generating each output "
    "rather than compressing an entire sequence into one fixed representation. The mechanism is "
    "especially useful in encoder decoder and transformer architectures because it creates a "
    "direct path between distant positions in the input sequence."
)
_HIGH_OVERLAP_ANSWER = (
    "model Attention mechanisms let a model compare query representations with key representations, "
    "produce normalized weights, and use those weights to combine value representations into a "
    "context vector. This helps the model focus on relevant tokens when generating each output"
)
_NORMAL_ANSWER = (
    "An attention mechanism scores which input tokens matter for the current output step and "
    "uses those scores to build a context representation from the source sequence."
)

# PROJECT_GOALS.md §7 Phase 1 names exactly five manual cases. They lived only
# in generator.py's __main__, so the suite's one sanity test covered a single
# case and the other four were never exercised by CI.
SANITY_CHECK_CASES = [
    ("short answer <20 words", "Attention weights identify relevant input tokens.", [{"text": _LONG_CHUNK}], True),
    ("verbatim chunk copy >200 chars", _LONG_CHUNK, [{"text": _LONG_CHUNK}], False),
    ("word overlap >=0.95", _HIGH_OVERLAP_ANSWER, [{"text": _LONG_CHUNK}], False),
    ("normal grounded answer", _NORMAL_ANSWER, [{"text": _LONG_CHUNK}], True),
    ("empty answer", "", [{"text": _LONG_CHUNK}], True),
]


@pytest.mark.parametrize(
    ("name", "answer", "chunks", "expected"),
    SANITY_CHECK_CASES,
    ids=[case[0] for case in SANITY_CHECK_CASES],
)
def test_sanity_check_answer_manual_cases(name, answer, chunks, expected):
    ok, reason = sanity_check_answer(answer, chunks)

    assert ok is expected, f"{name}: expected {expected}, got {ok} ({reason})"
