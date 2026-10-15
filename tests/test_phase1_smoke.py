from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase1_smoke import archive_existing_report, load_cases, summarize


def _passing_report() -> dict:
    records = []
    for index in range(20):
        should_abstain = index >= 15
        result = {
            "status": "INSUFFICIENT_CONTEXT" if should_abstain else "ANSWERED",
            "answer_status": "abstained" if should_abstain else "answered",
            "confidence_score": 0.8,
            "confidence_decision": "PARTIAL_TRUST",
            "abstain_reason": "No supporting evidence in the corpus." if should_abstain else None,
        }
        records.append({
            "query_id": f"q{index}",
            "should_abstain": should_abstain,
            "exception": None,
            "result": result,
            "runtime_checks": {
                "valid_status": True,
                "confidence_in_range": True,
                "retry_within_limit": True,
                "log_schema_complete": True,
                "log_latency_complete": True,
                "qdrant_count_matches": True,
                "dense_probe_nonempty": True,
            },
            "human_review": {
                "answer_correct": None if should_abstain else True,
                "grounded": None if should_abstain else True,
                "reviewer": None if should_abstain else "reviewer-1",
                "failure_categories": [],
            },
        })
    return {"records": records}


def test_phase1_acceptance_set_is_frozen_and_balanced():
    dataset = load_cases()

    assert dataset["dataset_version"] == "phase1-smoke-v1"
    assert len(dataset["records"]) == 20
    assert sum(row["should_abstain"] is False for row in dataset["records"]) == 15
    assert sum(row["should_abstain"] is True for row in dataset["records"]) == 5
    assert dataset["records"][-6]["query_id"] == "p1_answer_15_attention"


def test_phase1_review_gate_passes_only_with_runtime_and_quality_evidence():
    summary = summarize(_passing_report())

    assert summary["runtime_pass"] is True
    assert summary["correct_final_decisions"] == 20
    assert summary["confident_wrong_answers"] == 0
    assert summary["phase1_pass"] is True


def test_phase1_gate_requires_at_least_18_correct_decisions():
    report = _passing_report()
    for row in report["records"][:3]:
        row["human_review"]["answer_correct"] = False
        row["human_review"]["grounded"] = False

    summary = summarize(report)

    assert summary["correct_final_decisions"] == 17
    assert summary["quality_pass"] is False
    assert summary["phase1_pass"] is False


def test_phase1_gate_rejects_confident_wrong_answer_even_if_other_cases_pass():
    report = _passing_report()
    row = report["records"][0]
    row["human_review"]["answer_correct"] = False
    row["human_review"]["grounded"] = False
    row["result"]["confidence_decision"] = "GENERATE"
    for other in report["records"][1:3]:
        other["human_review"]["answer_correct"] = False
        other["human_review"]["grounded"] = False

    summary = summarize(report)

    assert summary["correct_final_decisions"] == 17
    assert summary["confident_wrong_answers"] == 1
    assert summary["phase1_pass"] is False


def test_phase1_gate_stays_pending_until_answerable_cases_are_reviewed():
    report = _passing_report()
    report["records"][2]["human_review"]["grounded"] = None

    summary = summarize(report)

    assert summary["human_review_complete"] is False
    assert summary["correct_final_decisions"] is None
    assert summary["phase1_pass"] is False


def test_p1_answer_02_ungrounded_generate_is_a_hard_acceptance_failure():
    report = _passing_report()
    row = report["records"][1]
    row["query_id"] = "p1_answer_02"
    row["human_review"].update({
        "answer_correct": True,
        "grounded": False,
        "reviewer": "reviewer-1",
        "failure_categories": ["unsupported_claim"],
    })
    row["result"]["confidence_decision"] = "GENERATE"

    summary = summarize(report)

    assert summary["confident_wrong_answers"] == 1
    assert summary["correct_final_decisions"] == 19
    assert summary["phase1_pass"] is False


def test_failed_review_requires_a_failure_category():
    report = _passing_report()
    report["records"][0]["human_review"].update({
        "answer_correct": True,
        "grounded": False,
    })

    summary = summarize(report)

    assert summary["human_review_complete"] is True
    assert summary["failure_reasons_complete"] is False
    assert summary["phase1_pass"] is False


def test_known_confident_wrong_is_reported_while_other_reviews_are_pending():
    report = _passing_report()
    report["records"][1]["human_review"].update({
        "answer_correct": True,
        "grounded": False,
        "reviewer": "reviewer-1",
        "failure_categories": ["unsupported_claim"],
    })
    report["records"][1]["result"]["confidence_decision"] = "GENERATE"
    report["records"][2]["human_review"]["answer_correct"] = None

    summary = summarize(report)

    assert summary["human_review_complete"] is False
    assert summary["confident_wrong_answers"] == 1


def test_phase1_gate_rejects_runtime_schema_or_exception_failures():
    report = _passing_report()
    report["records"][0]["exception"] = "RuntimeError: failed"

    summary = summarize(report)

    assert summary["runtime_pass"] is False
    assert summary["runtime_failed_query_ids"] == ["q0"]
    assert summary["phase1_pass"] is False


def test_archiving_preserves_the_previous_acceptance_report(tmp_path):
    report_path = tmp_path / "phase1_smoke.json"
    original = b'{"report":"previous acceptance evidence"}\n'
    report_path.write_bytes(original)

    archived_path = archive_existing_report(report_path)

    assert archived_path is not None
    assert archived_path.read_bytes() == original
    assert report_path.read_bytes() == original
