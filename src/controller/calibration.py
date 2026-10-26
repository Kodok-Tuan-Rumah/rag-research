"""Confidence calibration: isotonic regression + Platt scaling.

Maps raw heuristic confidence scores from ConfidenceEngine to calibrated
probabilities, so the decision zones (reject / partial / generate) reflect
true empirical accuracy rather than arbitrary thresholds.

Expected Calibration Error (ECE) target: < 0.10 on 200+ labeled samples.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger("rag.calibration")


@dataclass
class CalibrationSample:
    """One labeled calibration example."""
    raw_confidence: float      # ConfidenceEngine output (0..1)
    is_supported: bool         # Ground truth: answer supported by evidence?
    query_id: str              # Dataset identifier
    query: str = ""
    decision: str = ""         # REJECT / PARTIAL_TRUST / GENERATE


@dataclass
class CalibrationResult:
    """Result of fitting calibration on a dataset."""
    method: str                # "isotonic" or "platt"
    n_samples: int
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    model_path: str | None = None


class CalibrationEngine:
    """Fits and applies calibration to raw confidence scores.

    Supports two methods:
      - Isotonic regression (non-parametric, monotonic)
      - Platt scaling (parametric, logistic regression)

    After fitting, ``calibrate(raw_confidence)`` returns a calibrated
    probability in [0, 1] that the answer is supported by evidence.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._model: Any = None
        self._fitted = False
        self._calibration_samples: list[CalibrationSample] = []

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, samples: list[CalibrationSample], holdout_fraction: float = 0.2) -> CalibrationResult:
        """Fit calibration model on labeled samples.

        Splits into train/holdout, fits on train, reports ECE/Brier on holdout.
        """
        if len(samples) < 10:
            raise ValueError(
                f"Need at least 10 calibration samples, got {len(samples)}"
            )

        self._calibration_samples = samples
        X = np.array([s.raw_confidence for s in samples]).reshape(-1, 1)
        y = np.array([1 if s.is_supported else 0 for s in samples])

        # Split train/holdout
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, test_size=holdout_fraction, random_state=42, stratify=y
            if len(set(y.tolist())) > 1 else None,
        )

        # Metrics before calibration (raw confidence as probability)
        ece_before = self._ece(X_train.flatten(), y_train)
        brier_before = self._brier(X_train.flatten(), y_train)

        # Fit
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(X_train.flatten(), y_train)
        else:
            self._model = LogisticRegression()
            self._model.fit(X_train, y_train)

        self._fitted = True

        # Metrics after calibration on holdout
        y_hold_cal = self.calibrate_many(X_hold.flatten().tolist())
        ece_after = self._ece(np.array(y_hold_cal), y_hold)
        brier_after = self._brier(np.array(y_hold_cal), y_hold)

        return CalibrationResult(
            method=self.method,
            n_samples=len(samples),
            ece_before=round(float(ece_before), 4),
            ece_after=round(float(ece_after), 4),
            brier_before=round(float(brier_before), 4),
            brier_after=round(float(brier_after), 4),
        )

    def calibrate(self, raw_confidence: float) -> float:
        """Map a raw confidence score to a calibrated probability."""
        if not self._fitted:
            return raw_confidence
        if self.method == "isotonic":
            return float(self._model.predict([raw_confidence])[0])
        # Platt: logistic regression predict_proba
        proba = self._model.predict_proba([[raw_confidence]])[0]
        return float(proba[1])

    def calibrate_many(self, raw_confidences: list[float]) -> list[float]:
        """Calibrate multiple scores at once."""
        if not self._fitted:
            return list(raw_confidences)
        if self.method == "isotonic":
            return [float(v) for v in self._model.predict(raw_confidences)]
        proba = self._model.predict_proba(np.array(raw_confidences).reshape(-1, 1))
        return [float(p[1]) for p in proba]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(
            {
                "method": self.method,
                "model": self._model,
                "fitted": self._fitted,
                "samples": [
                    {
                        "raw_confidence": s.raw_confidence,
                        "is_supported": s.is_supported,
                        "query_id": s.query_id,
                        "query": s.query,
                        "decision": s.decision,
                    }
                    for s in self._calibration_samples
                ],
            },
            path,
        )
        logger.info(f"Calibration model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationEngine":
        import joblib
        payload = joblib.load(path)
        engine = cls(method=payload["method"])
        engine._model = payload["model"]
        engine._fitted = payload["fitted"]
        engine._calibration_samples = [
            CalibrationSample(
                raw_confidence=s["raw_confidence"],
                is_supported=s["is_supported"],
                query_id=s["query_id"],
                query=s.get("query", ""),
                decision=s.get("decision", ""),
            )
            for s in payload.get("samples", [])
        ]
        logger.info(f"Calibration model loaded from {path}")
        return engine

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        if len(probabilities) == 0:
            return 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece)

    @staticmethod
    def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Brier score."""
        if len(probabilities) == 0:
            return 0.0
        return float(np.mean((probabilities - labels) ** 2))

    @property
    def is_fitted(self) -> bool:
        return self._fitted
