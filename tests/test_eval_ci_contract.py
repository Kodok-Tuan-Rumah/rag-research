import json

import pytest

from scripts import eval_ci


def test_load_samples_requires_complete_ragas_fields(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{
        "query": "What is retrieval?",
        "should_abstain": True,
    }]))

    with pytest.raises(ValueError, match="RAGAS-ready"):
        eval_ci.load_samples(path)


def test_ci_gate_uses_threshold_metric_names(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{
        "question": "What is retrieval?",
        "answer": "Retrieval finds relevant evidence.",
        "contexts": ["Retrieval finds relevant evidence."],
        "ground_truth": "Retrieval finds relevant evidence.",
    }]))

    class FakeEvaluator:
        def __init__(self, include_answer_correctness):
            assert include_answer_correctness is False

        def evaluate(self, samples, exp_id):
            assert len(samples) == 1
            return type("Result", (), {
                "avg_faithfulness": 0.95,
                "avg_answer_relevancy": 0.95,
                "avg_context_recall": 0.95,
                "hallucination_rate": 0.0,
                "total_samples": 1,
            })()

    monkeypatch.setattr(eval_ci, "RAGASEvaluator", FakeEvaluator)
    output_path = tmp_path / "result.json"

    assert eval_ci.run_ci_eval(path, output_path) == 0
    payload = json.loads(output_path.read_text())
    assert payload["summary"]["faithfulness"] == 0.95
    assert payload["summary"]["answer_relevancy"] == 0.95
    assert payload["summary"]["context_recall"] == 0.95


def test_ci_gate_fails_when_metrics_miss_threshold(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{
        "question": "What is retrieval?",
        "answer": "Retrieval finds relevant evidence.",
        "contexts": ["Retrieval finds relevant evidence."],
        "ground_truth": "Retrieval finds relevant evidence.",
    }]))

    class LowScoreEvaluator:
        def __init__(self, include_answer_correctness):
            pass

        def evaluate(self, samples, exp_id):
            return type("Result", (), {
                "avg_faithfulness": 0.2,
                "avg_answer_relevancy": 0.2,
                "avg_context_recall": 0.2,
                "hallucination_rate": 0.5,
                "total_samples": 1,
            })()

    monkeypatch.setattr(eval_ci, "RAGASEvaluator", LowScoreEvaluator)

    assert eval_ci.run_ci_eval(path, tmp_path / "result.json") == 1


def test_ci_gate_rejects_non_finite_metrics(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{
        "question": "What is retrieval?",
        "answer": "Retrieval finds relevant evidence.",
        "contexts": ["Retrieval finds relevant evidence."],
        "ground_truth": "Retrieval finds relevant evidence.",
    }]))

    class NonFiniteEvaluator:
        def __init__(self, include_answer_correctness):
            pass

        def evaluate(self, samples, exp_id):
            return type("Result", (), {
                "avg_faithfulness": float("nan"),
                "avg_answer_relevancy": 0.95,
                "avg_context_recall": 0.95,
                "hallucination_rate": 0.0,
                "total_samples": 1,
            })()

    monkeypatch.setattr(eval_ci, "RAGASEvaluator", NonFiniteEvaluator)
    output_path = tmp_path / "result.json"

    assert eval_ci.run_ci_eval(path, output_path) == 1
    assert json.loads(output_path.read_text())["summary"]["faithfulness"] is None
