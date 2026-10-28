"""The calibration dataset must not contain failed pipeline runs.

``raw_decision == "ERROR"`` is a sentinel for "the run broke", not an
observation about evidence. Fitting it teaches isotonic regression that a
supported query deserves zero confidence, which is the opposite of what the
calibration is supposed to learn. These tests pin both the artifact and the
guard in ``scripts/fit_calibration.py``.
"""

from __future__ import annotations

import json

import pytest

from scripts.fit_calibration import DatasetError, load_samples

DATASET = "data/evaluation/calibration_dataset.json"


def _load() -> dict:
    with open(DATASET, encoding="utf-8") as handle:
        return json.load(handle)


def test_dataset_has_no_error_rows():
    data = _load()
    broken = [
        row.get("query_id")
        for row in data["samples"]
        if str(row.get("raw_decision", "")).upper() == "ERROR"
        or row.get("raw_confidence") is None
    ]
    assert broken == [], f"ERROR rows must move to excluded_samples: {broken}"


def test_declared_counts_match_rows():
    data = _load()
    rows = data["samples"]
    assert data["total_samples"] == len(rows)
    assert data["supported_count"] == sum(1 for row in rows if row["is_supported"])
    assert data["unsupported_count"] == sum(1 for row in rows if not row["is_supported"])


def test_excluded_rows_keep_provenance():
    data = _load()
    excluded = data.get("excluded_samples", [])
    assert data.get("excluded_count") == len(excluded) > 0
    for row in excluded:
        assert row.get("exclusion_reason"), "every excluded row needs a reason"
        assert str(row.get("raw_decision", "")).upper() == "ERROR"


def test_loader_accepts_clean_dataset():
    _, samples = load_samples(DATASET)
    assert len(samples) == _load()["total_samples"]


def test_loader_refuses_error_rows(tmp_path):
    data = _load()
    broken = dict(data["excluded_samples"][0])
    broken.pop("exclusion_reason", None)
    data["samples"] = data["samples"] + [broken]
    data["total_samples"] = len(data["samples"])
    path = tmp_path / "with_error.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetError, match="ERROR row"):
        load_samples(path)


def test_loader_refuses_count_drift(tmp_path):
    data = _load()
    data["total_samples"] = len(data["samples"]) + 1
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetError, match="total_samples"):
        load_samples(path)
