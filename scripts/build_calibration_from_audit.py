#!/usr/bin/env python3
"""Build a calibration dataset from persisted audit_report_claims.

Reads audit_report_claims (decision, calibrated_confidence) and emits a
calibration_dataset.json consumable by scripts/fit_calibration.py.

Usage (from rag-research/):
    PYTHONPATH=. python scripts/build_calibration_from_audit.py \
        --out data/evaluation/calibration_dataset_from_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import psycopg2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"psycopg2 required: {exc}")


def fetch_rows(dsn: str) -> list[dict]:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT claim, decision, calibrated_confidence, confidence_zone "
                "FROM audit_report_claims ORDER BY claim_index"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def build_dataset(rows: list[dict]) -> dict:
    samples = []
    for i, r in enumerate(rows):
        decision = (r.get("decision") or "").lower()
        is_supported = decision == "answer"
        conf = r.get("calibrated_confidence")
        if conf is None:
            continue
        samples.append({
            "query_id": f"audit_{i}",
            "query": r.get("claim") or "",
            "raw_confidence": float(conf),
            "raw_decision": (r.get("decision") or "").upper() or "REJECT",
            "is_supported": is_supported,
        })
    supported = sum(1 for s in samples if s["is_supported"])
    return {
        "dataset_name": "audit_reports_calibration",
        "total_samples": len(samples),
        "supported_count": supported,
        "samples": samples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/evaluation/calibration_dataset_from_audit.json")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--min-samples", type=int, default=50,
                        help="minimum audit samples required before fitting (default 50)")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("DATABASE_URL") or (
        f"host={os.environ.get('POSTGRES_HOST','localhost')} "
        f"port={os.environ.get('POSTGRES_PORT','5432')} "
        f"dbname={os.environ.get('POSTGRES_DB','arxiv_db')} "
        f"user={os.environ.get('POSTGRES_USER','arxiv')} "
        f"password={os.environ.get('POSTGRES_PASSWORD','')}"
    )

    try:
        rows = fetch_rows(dsn)
    except Exception as exc:
        print(f"db fetch failed: {exc}", file=__import__("sys").stderr)
        return 1

    if not rows:
        print("no audit rows found", file=__import__("sys").stderr)
        return 1

    dataset = build_dataset(rows)
    if len(dataset["samples"]) < args.min_samples:
        print(
            f"insufficient audit samples: {len(dataset['samples'])} < {args.min_samples}; "
            "skipping (Opsi A: wait for real data)",
            file=__import__("sys").stderr,
        )
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(dataset['samples'])} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY