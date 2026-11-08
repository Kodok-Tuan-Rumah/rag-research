from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controller.calibration import CalibrationEngine, CalibrationSample  # noqa: E402

METHODS = ("isotonic", "platt")


class DatasetError(ValueError):
    """The dataset cannot be fitted as-is."""


def load_samples(path: str | Path) -> tuple[dict, list[CalibrationSample]]:
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
        raise DatasetError(f"{path}: total_samples={declared} but {len(rows)} rows present")
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


def fit_all(samples: list[CalibrationSample], dataset_path: str | Path = None) -> dict:
    results = {}
    for method in METHODS:
        engine = CalibrationEngine(method=method)
        result = engine.fit(samples, dataset_path=dataset_path)
        results[method] = {
            "n_samples": result.n_samples,
            "holdout_n": result.holdout_n,
            "ece_before": result.ece_before,
            "ece_after": result.ece_after,
            "ece_ci_low": result.ece_ci_low,
            "ece_ci_high": result.ece_ci_high,
            "cv_ece": result.cv_ece,
            "brier_before": result.brier_before,
            "brier_after": result.brier_after,
            "dataset_hash": result.dataset_hash,
            "code_version": result.code_version,
            "fitted_at": result.fitted_at,
            "_engine": engine,
        }
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

    results = fit_all(samples, dataset_path=args.dataset)
    report = {
        "dataset": str(args.dataset),
        "dataset_name": data.get("dataset_name"),
        "n_samples": len(samples),
        "fit_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "code_version": None,
        "methods": {},
    }
    for method, metrics in results.items():
        engine = metrics.pop("_engine")
        target = out / f"calibration_{method}.joblib"
        engine.save(target)
        report["methods"][method] = metrics
        # Carry code_version up to report top level (same for all methods).
        if report["code_version"] is None and metrics.get("code_version"):
            report["code_version"] = metrics["code_version"]
        print(
            f"{method}: n={metrics['n_samples']} holdout={metrics['holdout_n']} "
            f"ECE {metrics['ece_before']} -> {metrics['ece_after']} "
            f"(CI {metrics['ece_ci_low']}-{metrics['ece_ci_high']}, "
            f"CV {metrics['cv_ece']}) | "
            f"Brier {metrics['brier_before']} -> {metrics['brier_after']} "
            f"-> {target}"
        )

    (out / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
