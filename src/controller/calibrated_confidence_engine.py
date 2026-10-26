"""Calibrated ConfidenceEngine wrapper.

Applies calibrated probability (from CalibrationEngine) to raw heuristic
confidence scores, then maps calibrated probability to decision zones
(REJECT / PARTIAL_TRUST / GENERATE) using calibrated thresholds.

Calibrated thresholds:
  - REJECT if calibrated_prob < calibrated_reject_threshold (default 0.35)
  - PARTIAL_TRUST if calibrated_prob < calibrated_partial_threshold (default 0.65)
  - GENERATE otherwise

These thresholds mirror the original heuristic values but now apply to
calibrated probabilities instead of raw heuristic scores.

Public calibration API:
  - ``fit_calibration(samples)`` — fit isotonic/Platt on labeled data
  - ``save_calibration(path)`` / ``load_calibration(path)`` — persist/load
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.controller.calibration import CalibrationEngine, CalibrationSample
from src.controller.confidence_engine import ConfidenceEngine

logger = logging.getLogger("rag.calibrated_confidence_engine")


class CalibratedConfidenceEngine:
    """Wrapper around ConfidenceEngine that applies calibration."""

    def __init__(
        self,
        base_engine: ConfidenceEngine | None = None,
        calibration_engine: CalibrationEngine | None = None,
        calibrated_reject_threshold: float = 0.35,
        calibrated_partial_threshold: float = 0.65,
        calibration_method: str | None = None,
    ):
        from src.api.config import settings

        self.base_engine = base_engine or ConfidenceEngine()

        # Wire calibration_method from settings if not explicitly passed
        if calibration_method is None:
            calibration_method = (
                settings.calibration_method
                if hasattr(settings, "calibration_method")
                else "isotonic"
            )

        self.calibration_engine = calibration_engine or CalibrationEngine(
            method=calibration_method
        )
        self.calibrated_reject_threshold = calibrated_reject_threshold
        self.calibrated_partial_threshold = calibrated_partial_threshold

        # Allow settings override
        self.calibrated_reject_threshold = (
            settings.calibrated_reject_threshold
            if hasattr(settings, "calibrated_reject_threshold")
            else calibrated_reject_threshold
        )
        self.calibrated_partial_threshold = (
            settings.calibrated_partial_threshold
            if hasattr(settings, "calibrated_partial_threshold")
            else calibrated_partial_threshold
        )

    # ------------------------------------------------------------------ #
    # Calibration fitting
    # ------------------------------------------------------------------ #
    def fit_calibration(self, samples: list[CalibrationSample], holdout_fraction: float = 0.2):
        """Fit the calibration engine on labeled samples."""
        result = self.calibration_engine.fit(samples, holdout_fraction=holdout_fraction)
        logger.info(
            f"Calibration fitted: method={result.method}, "
            f"n={result.n_samples}, ECE {result.ece_before} -> {result.ece_after}, "
            f"Brier {result.brier_before} -> {result.brier_after}"
        )
        return result

    def load_calibration(self, path: str) -> None:
        """Load a previously fitted calibration model."""
        self.calibration_engine = CalibrationEngine.load(path)

    def save_calibration(self, path: str) -> None:
        """Save the fitted calibration model."""
        self.calibration_engine.save(path)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_engine.is_fitted

    # ------------------------------------------------------------------ #
    # Core: calculate calibrated confidence
    # ------------------------------------------------------------------ #
    def calculate_confidence(
        self,
        chunks: list[Any],
        answer: str | None = None,
        contexts: list[str] | None = None,
        query_type: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Calculate confidence with calibration applied.

        Returns the same dict structure as ConfidenceEngine.calculate_confidence,
        plus calibrated fields.
        """
        # Get raw confidence from base engine
        raw_result = self.base_engine.calculate_confidence(
            chunks=chunks,
            answer=answer,
            contexts=contexts,
            query_type=query_type,
            query=query,
        )

        raw_confidence = raw_result["confidence_score"]

        # Apply calibration
        if self.is_calibrated:
            calibrated_prob = self.calibration_engine.calibrate(raw_confidence)
            calibration_method = self.calibration_engine.method
        else:
            calibrated_prob = raw_confidence
            calibration_method = "heuristic_uncalibrated"

        # Map calibrated probability to decision zones
        calibrated_decision = self._decision_for_calibrated_prob(calibrated_prob)

        # Build result
        result = dict(raw_result)
        result.update(
            {
                "raw_confidence": round(float(raw_confidence), 4),
                "calibrated_confidence": round(float(calibrated_prob), 4),
                "calibrated_decision": calibrated_decision,
                "is_calibrated": self.is_calibrated,
                "calibration_method": calibration_method,
                # Use calibrated decision as the public decision
                "decision": calibrated_decision,
            }
        )

        logger.info(
            f"[CALIBRATED_CONFIDENCE] raw={raw_confidence:.4f} | "
            f"calibrated={calibrated_prob:.4f} | "
            f"raw_decision={raw_result['decision']} | "
            f"calibrated_decision={calibrated_decision} | "
            f"calibrated={self.is_calibrated}"
        )

        return result

    def _decision_for_calibrated_prob(self, prob: float) -> str:
        """Map calibrated probability to decision zone."""
        if prob < self.calibrated_reject_threshold:
            return "REJECT"
        if prob < self.calibrated_partial_threshold:
            return "PARTIAL_TRUST"
        return "GENERATE"

    # ------------------------------------------------------------------ #
    # Delegate attribute access to base engine for compatibility
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_engine, name)
