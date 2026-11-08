from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

logger = logging.getLogger("rag.calibration")

try:
    _CODE_VERSION = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()[:12]
except Exception:
    _CODE_VERSION = "unknown"


def _hash_payload(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class CalibrationSample:
    raw_confidence: float  # ConfidenceEngine output (0..1)
    is_supported: bool  # Ground truth: answer supported by evidence?
    query_id: str  # Dataset identifier
    query: str = ""
    decision: str = ""  # REJECT / PARTIAL_TRUST / GENERATE


@dataclass
class CalibrationResult:
    method: str  # "isotonic" or "platt"
    n_samples: int  # total samples the final model was fit on
    holdout_n: int  # size of holdout used for honest evaluation
    ece_before: float  # raw ECE on train split (before calibration)
    ece_after: float  # holdout ECE after calibration (honest eval)
    brier_before: float  # raw Brier on train split
    brier_after: float  # holdout Brier after calibration
    ece_ci_low: float | None = None  # bootstrap CI lower bound (holdout)
    ece_ci_high: float | None = None  # bootstrap CI upper bound (holdout)
    cv_ece: float | None = None  # 5-fold cross-validation ECE
    dataset_hash: str | None = None
    code_version: str | None = None
    fitted_at: str | None = None  # ISO-8601 timestamp
    model_path: str | None = None


class CalibrationEngine:
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
    def fit(
        self,
        samples: list[CalibrationSample],
        holdout_fraction: float = 0.2,
        cv_folds: int = 5,
        dataset_path: str | Path | None = None,
    ) -> CalibrationResult:
        if len(samples) < 10:
            raise ValueError(f"Need at least 10 calibration samples, got {len(samples)}")

        self._calibration_samples = samples
        X = np.array([s.raw_confidence for s in samples]).reshape(-1, 1)
        y = np.array([1 if s.is_supported else 0 for s in samples])

        stratify = y if len(set(y.tolist())) > 1 else None
        X_train, X_hold, y_train, y_hold = train_test_split(
            X,
            y,
            test_size=holdout_fraction,
            random_state=42,
            stratify=stratify,
        )

        holdout_n = len(y_hold)

        # Metrics before calibration (raw confidence as probability) on train.
        ece_before = self._ece(X_train.flatten(), y_train)
        brier_before = self._brier(X_train.flatten(), y_train)

        # ---- Candidate model on train only (for holdout evaluation) ----
        candidate = self._fresh_model()
        candidate.fit(X_train, y_train)
        y_hold_cal = self._predict(candidate, X_hold)
        ece_holdout = self._ece(np.array(y_hold_cal), y_hold)
        brier_holdout = self._brier(np.array(y_hold_cal), y_hold)

        # ---- Bootstrap CI on holdout ECE ----
        ece_ci_low, ece_ci_high = self._bootstrap_ece_ci(candidate, X_hold, y_hold)

        # ---- 5-fold cross-validation ECE (more stable than single holdout) ----
        cv_ece = self._crossval_ece(X.flatten(), y, n_splits=cv_folds)

        # ---- Refit FINAL production model on ALL samples ----
        self._model = self._fresh_model()
        self._model.fit(X, y)
        self._fitted = True

        # Metadata
        dataset_hash = None
        if dataset_path is not None:
            try:
                dataset_hash = _hash_payload(
                    json.loads(Path(dataset_path).read_text(encoding="utf-8"))
                )
            except Exception:  # pragma: no cover - best effort
                dataset_hash = None

        result = CalibrationResult(
            method=self.method,
            n_samples=len(samples),
            holdout_n=holdout_n,
            ece_before=round(float(ece_before), 4),
            ece_after=round(float(ece_holdout), 4),
            ece_ci_low=round(float(ece_ci_low), 4) if ece_ci_low is not None else None,
            ece_ci_high=round(float(ece_ci_high), 4) if ece_ci_high is not None else None,
            cv_ece=round(float(cv_ece), 4) if cv_ece is not None else None,
            brier_before=round(float(brier_before), 4),
            brier_after=round(float(brier_holdout), 4),
            dataset_hash=dataset_hash,
            code_version=_CODE_VERSION,
            fitted_at=_dt.datetime.now(_dt.UTC).isoformat(),
        )
        self._last_result = result
        return result

    def calibrate(self, raw_confidence: float) -> float:
        if not self._fitted:
            return raw_confidence
        if self.method == "isotonic":
            return float(self._model.predict([raw_confidence])[0])
        proba = self._model.predict_proba([[raw_confidence]])[0]
        return float(proba[1])

    def calibrate_many(self, raw_confidences: list[float]) -> list[float]:
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

        payload = {
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
        }
        if getattr(self, "_last_result", None) is not None:
            payload["metadata"] = asdict(self._last_result)
        joblib.dump(payload, path)
        logger.info(f"Calibration model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationEngine:
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
        if payload.get("metadata") is not None:
            engine._last_result = payload["metadata"]
        logger.info(f"Calibration model loaded from {path}")
        return engine

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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
        if len(probabilities) == 0:
            return 0.0
        return float(np.mean((probabilities - labels) ** 2))

    # ------------------------------------------------------------------ #
    # Internal helpers for honest evaluation (P1-2)
    # ------------------------------------------------------------------ #
    def _fresh_model(self) -> Any:
        if self.method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip")
        return LogisticRegression()

    @staticmethod
    def _predict(model: Any, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(-1)
        if isinstance(model, IsotonicRegression):
            return model.predict(x_flat)
        return model.predict_proba(x)[:, 1]

    def _bootstrap_ece_ci(
        self, model: Any, x_hold: np.ndarray, y_hold: np.ndarray, n_boot: int = 2000, seed: int = 7
    ) -> tuple[float | None, float | None]:
        if len(x_hold) == 0:
            return None, None
        rng = np.random.RandomState(seed)
        n = len(x_hold)
        y_cal = self._predict(model, x_hold)
        eces = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            eces.append(self._ece(y_cal[idx], y_hold[idx]))
        eces = np.array(eces)
        return float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5))

    def _crossval_ece(
        self, x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42
    ) -> float | None:
        if len(x) < n_splits * 2:
            return None
        x2 = x.reshape(-1, 1)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        eces = []
        for train_idx, test_idx in kf.split(x2):
            model = self._fresh_model()
            model.fit(x2[train_idx], y[train_idx])
            y_cal = self._predict(model, x2[test_idx])
            eces.append(self._ece(np.array(y_cal), y[test_idx]))
        return float(np.mean(eces))

    @property
    def is_fitted(self) -> bool:
        return self._fitted
