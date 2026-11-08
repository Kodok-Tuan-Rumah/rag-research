from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "results/metrics/phase1_smoke_v3.json"


def test_phase1_artifact_passes():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    assert summary.get("total") == 20
    assert summary.get("provider_errors") == 0
    assert summary.get("runtime_pass") is True


def test_p1_answer_11_and_13_are_answered_from_prompt_evidence():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    records = {row["query_id"]: row for row in report.get("records", [])}

    evidence_markers = {
        "p1_answer_11": ("540", "parameter"),
        "p1_answer_13": ("retriever module",),
    }
    answer_markers = {
        "p1_answer_11": ("540", "parameter"),
        "p1_answer_13": ("retriever module",),
    }

    for query_id, markers in evidence_markers.items():
        result = records[query_id]["result"]
        assert result["status"] == "ANSWERED", query_id
        assert result["answer_status"] == "answered", query_id
        assert result["confidence_decision"] != "REJECT", query_id
        assert (
            result["retrieval_confidence"]["signals"]["query_type"]
            == result["config"]["query_type"]
        ), query_id

        generations = result["generation_contexts"]
        assert generations, f"{query_id}: no prompt context was recorded"
        prompt_ids = {
            chunk_id
            for attempt in generations
            for chunk_id in attempt.get("prompt_chunk_ids", [])
        }
        assert prompt_ids, f"{query_id}: no prompt chunk IDs were recorded"

        context = " ".join(attempt.get("context", "") for attempt in generations).casefold()
        answer = result["answer"].casefold()
        assert all(marker in context for marker in markers), query_id
        assert all(marker in answer for marker in answer_markers[query_id]), query_id
