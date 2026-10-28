#!/usr/bin/env python3
"""Fit and persist the confidence-calibration models from a labeled dataset.

The ``joblib`` artifacts under ``models/`` are build outputs, not tracked
inputs. The Dockerfile runs this script during the image build so that a fresh
clone cannot start with an uncalibrated or stale model, and so the artifact can
always be traced back to one dataset revision.

Failed pipeline runs must not reach the fitter. A row marked ``ERROR`` carries a
sentinel ``raw_confidence`` (0.0) that means "the run broke", not "the evidence
was weak"; fitting it would teach isotonic regression that a supported query is
worth zero confidence. ``load_samples`` therefore refuses such rows instead of
silently dropping them.

Usage (from rag-research/):

    PYTHONPATH=. python scripts/fit_calibration.py \
        --dataset data/evaluation/calibration_dataset.json --out models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controller.calibration import CalibrationEngine, CalibrationSample  # noqa: E402

METHODS = ("isotonic", "platt")


class DatasetError(ValueError):
    """The dataset cannot be fitted as-is."""


def load_samples(path: str | Path) -> tuple[dict, list[CalibrationSample]]:
    """Read a calibration dataset and return its metadata and usable samples.

    Raises ``DatasetError`` when a row is an ERROR sentinel or when the declared
    counts disagree with the rows actually present. Both conditions mean the
    dataset is not fit for calibration, and fitting it anyway would produce a
    model that looks valid but is trained on corrupted observations.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("samples", [])
    if not rows:
        raise DatasetError(f"{path}: no samples")

    broken = [
        str(row.get("query_id", "?"))
        for row in rows
        if str(row.get("raw_decision", "")).upper() == "ERROR"
        or row.get("raw_confidence") is None
    ]
    if broken:
        raise DatasetError(
            f"{path}: {len(broken)} ERROR row(s) cannot be fitted "
            f"({', '.join(broken[:5])}{'...' if len(broken) > 5 else ''}). "
            "Re-run those queries or move them to excluded_samples."
        )

    declared = data.get("total_samples")
    if declared is not None and declared != len(rows):
        raise DatasetError(
            f"{path}: total_samples={declared} but {len(rows)} rows present"
        )
    supported = sum(1 for row in rows if bool(row.get("is_supported")))
    if data.get("supported_count") is not None and data["supported_count"] != supported:
        raise DatasetError(
            f"{path}: supported_count={data['supported_count']} but {supported} supported rows"
        )

    samples = [
        CalibrationSample(
            raw_confidence=float(row["raw_confidence"]),
            is_supported=bool(row.get("is_supported")),
            query_id=str(row.get("query_id", "")),
            query=str(row.get("query", "")),
            decision=str(row.get("raw_decision", "")),
        )
        for row in rows
    ]
    return data, samples


def fit_all(samples: list[CalibrationSample]) -> dict:
    results = {}
    for method in METHODS:
        engine = CalibrationEngine(method=method)
        result = engine.fit(samples)
        results[method] = {
            "n_samples": result.n_samples,
            "ece_before": result.ece_before,
            "ece_after": result.ece_after,
            "brier_before": result.brier_before,
            "brier_after": result.brier_after,
        }
        results[method]["_engine"] = engine
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/evaluation/calibration_dataset.json",
        help="labeled calibration dataset (JSON)",
    )
    parser.add_argument("--out", default="models", help="output directory")
    args = parser.parse_args(argv)

    try:
        data, samples = load_samples(args.dataset)
    except DatasetError as exc:
        print(f"refusing to fit: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = fit_all(samples)
    report = {
        "dataset": str(args.dataset),
        "dataset_name": data.get("dataset_name"),
        "n_samples": len(samples),
        "methods": {},
    }
    for method, metrics in results.items():
        engine = metrics.pop("_engine")
        target = out / f"calibration_{method}.joblib"
        engine.save(target)
        report["methods"][method] = metrics
        print(
            f"{method}: n={metrics['n_samples']} "
            f"ECE {metrics['ece_before']} -> {metrics['ece_after']} | "
            f"Brier {metrics['brier_before']} -> {metrics['brier_after']} "
            f"-> {target}"
        )

    (out / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
