from __future__ import annotations

import logging
from typing import Any

from src.controller.calibration import CalibrationEngine, CalibrationSample
from src.controller.confidence_engine import ConfidenceEngine

logger = logging.getLogger("rag.calibrated_confidence_engine")


class CalibratedConfidenceEngine:
    def __init__(
        self,
        base_engine: ConfidenceEngine | None = None,
        calibration_engine: CalibrationEngine | None = None,
        calibrated_reject_threshold: float | None = None,
        calibrated_partial_threshold: float | None = None,
        calibration_method: str | None = None,
    ):
        self.base_engine = base_engine or ConfidenceEngine()

        from src.api.config import settings

        if calibration_method is None:
            calibration_method = (
                settings.calibration_method
                if hasattr(settings, "calibration_method")
                else "isotonic"
            )

        self.calibration_engine = calibration_engine or CalibrationEngine(method=calibration_method)

        if calibrated_reject_threshold is None:
            calibrated_reject_threshold = (
                settings.calibrated_reject_threshold
                if hasattr(settings, "calibrated_reject_threshold")
                else 0.25
            )
        if calibrated_partial_threshold is None:
            calibrated_partial_threshold = (
                settings.calibrated_partial_threshold
                if hasattr(settings, "calibrated_partial_threshold")
                else 0.65
            )

        self.calibrated_reject_threshold = float(calibrated_reject_threshold)
        self.calibrated_partial_threshold = float(calibrated_partial_threshold)

    # Calibration fitting
    def fit_calibration(self, samples: list[CalibrationSample], holdout_fraction: float = 0.2):
        result = self.calibration_engine.fit(samples, holdout_fraction=holdout_fraction)
        logger.info(
            f"Calibration fitted: method={result.method}, "
            f"n={result.n_samples}, ECE {result.ece_before} -> {result.ece_after}, "
            f"Brier {result.brier_before} -> {result.brier_after}"
        )
        return result

    def load_calibration(self, path: str) -> bool:
        try:
            self.calibration_engine = CalibrationEngine.load(path)
        except Exception as exc:
            logger.warning(
                "[CALIBRATION] model unavailable at %s (%s: %s); using uncalibrated engine",
                path,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            method = getattr(self.calibration_engine, "method", "isotonic")
            self.calibration_engine = CalibrationEngine(method=method)
            return False
        logger.info("[CALIBRATION] model loaded from %s", path)
        return True

    def save_calibration(self, path: str) -> None:
        self.calibration_engine.save(path)

    def load_from_settings(self, path: str | None = None) -> CalibratedConfidenceEngine:
        if isinstance(self, str) and path is None:
            path = self
            self = None

        if isinstance(self, type):
            instance = self()
            instance.load_from_settings(path=path)
            return instance

        if self is None:
            self = CalibratedConfidenceEngine()

        from src.api.config import settings

        if path is None:
            path = getattr(settings, "calibration_model_path", None)
        if path:
            self.load_calibration(path)
        return self

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_engine.is_fitted

    # Core: calculate calibrated confidence
    def calculate_confidence(
        self,
        chunks: list[Any],
        answer: str | None = None,
        contexts: list[str] | None = None,
        query_type: str | None = None,
        query: str | None = None,
        use_retrieval_score: bool = False,
    ) -> dict[str, Any]:
        raw_result = self.base_engine.calculate_confidence(
            chunks=chunks,
            answer=answer,
            contexts=contexts,
            query_type=query_type,
            query=query,
            use_retrieval_score=use_retrieval_score,
        )

        raw_confidence = raw_result["confidence_score"]

        # Apply calibration
        if self.is_calibrated:
            calibrated_prob = self.calibration_engine.calibrate(raw_confidence)
            calibration_method = self.calibration_engine.method
        else:
            calibrated_prob = raw_confidence
            calibration_method = "heuristic_uncalibrated"
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
        if prob < self.calibrated_reject_threshold:
            return "REJECT"
        if prob < self.calibrated_partial_threshold:
            return "PARTIAL_TRUST"
        return "GENERATE"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_engine, name)
