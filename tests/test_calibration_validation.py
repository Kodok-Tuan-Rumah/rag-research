"""Calibration validation test.

Verifies that the calibration model achieves ECE < 0.10 on the labeled dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller.calibration import CalibrationEngine, CalibrationSample


def test_calibration_ece_below_target():
    """Isotonic calibration ECE must be < 0.10 on the labeled dataset."""
    dataset_path = Path("data/evaluation/calibration_dataset.json")
    assert dataset_path.is_file(), "Calibration dataset not found"

    with open(dataset_path) as f:
        dataset = json.load(f)

    samples = [
        CalibrationSample(
            raw_confidence=s["raw_confidence"],
            is_supported=s["is_supported"],
            query_id=s["query_id"],
            query=s["query"],
            decision=s.get("raw_decision", ""),
        )
        for s in dataset["samples"]
        if s.get("raw_confidence") is not None
    ]

    assert len(samples) >= 200, f"Need 200+ samples, got {len(samples)}"

    engine = CalibrationEngine(method="isotonic")
    result = engine.fit(samples)

    assert result.ece_after < 0.10, (
        f"Isotonic ECE {result.ece_after} >= 0.10 target. "
        f"Before: {result.ece_before}, After: {result.ece_after}"
    )


def test_calibration_dataset_size():
    """Calibration dataset must have 200+ labeled samples."""
    dataset_path = Path("data/evaluation/calibration_dataset.json")
    with open(dataset_path) as f:
        dataset = json.load(f)
    assert dataset["total_samples"] >= 200, (
        f"Dataset has {dataset['total_samples']} samples, need 200+"
    )
    assert dataset["supported_count"] > 0
    assert dataset["unsupported_count"] > 0


def test_calibration_model_saved():
    """Calibration model must be saved and loadable."""
    model_path = Path("models/calibration_isotonic.joblib")
    assert model_path.is_file(), "Calibration model not saved"

    engine = CalibrationEngine.load(str(model_path))
    assert engine.is_fitted
    # Test calibration mapping
    cal = engine.calibrate(0.5)
    assert 0.0 <= cal <= 1.0