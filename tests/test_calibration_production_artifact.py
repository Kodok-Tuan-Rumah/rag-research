"""Production calibration-artifact validation.

Loads the shipped calibration model (``models/calibration_isotonic.joblib``) and
computes Expected Calibration Error (ECE) / Brier against the FULL labeled
dataset — not merely checking that ``calibrate(0.5)`` returns a value in [0, 1].
This guards against a stale or mis-fitted artifact reaching production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller.calibration import CalibrationEngine, CalibrationSample

DATASET = Path("data/evaluation/calibration_dataset.json")
MODEL = Path("models/calibration_isotonic.joblib")


def _load_samples():
    assert DATASET.is_file(), "calibration dataset not found"
    assert MODEL.is_file(), "production calibration model not found"
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    return [
        CalibrationSample(
            raw_confidence=float(s["raw_confidence"]),
            is_supported=bool(s.get("is_supported")),
            query_id=str(s.get("query_id", "")),
            query=str(s.get("query", "")),
            decision=str(s.get("raw_decision", "")),
        )
        for s in dataset["samples"]
        if s.get("raw_confidence") is not None
    ]


def test_production_artifact_ece_against_full_dataset():
    """ECE of the shipped isotonic model on the full dataset must be < 0.10."""
    samples = _load_samples()
    assert len(samples) >= 200, f"need 200+ samples, got {len(samples)}"

    engine = CalibrationEngine.load(str(MODEL))
    assert engine.is_fitted, "shipped model must be fitted"

    X = [s.raw_confidence for s in samples]
    y = [1 if s.is_supported else 0 for s in samples]
    y_cal = engine.calibrate_many(X)

    ece = engine._ece(np.array(y_cal), np.array(y))
    brier = engine._brier(np.array(y_cal), np.array(y))

    assert ece < 0.10, (
        f"production artifact ECE {ece:.4f} >= 0.10 target "
        f"(n={len(samples)}, brier={brier:.4f})"
    )
    assert 0.0 <= brier <= 1.0


def test_production_artifact_metadata_present():
    """Shipped artifact must carry auditable metadata (dataset hash, n, date, version)."""
    engine = CalibrationEngine.load(str(MODEL))
    meta = engine._last_result
    assert meta is not None, "artifact missing calibration metadata"
    for key in ("n_samples", "holdout_n", "ece_after", "dataset_hash", "code_version", "fitted_at"):
        assert key in meta, f"metadata missing {key}"
    assert meta["n_samples"] >= 200
    assert meta["dataset_hash"]
    assert meta["code_version"]
