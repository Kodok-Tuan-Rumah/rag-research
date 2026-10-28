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
