#!/usr/bin/env python3
"""Export a small Phase 1 sample for independent, gold-blind review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "results/metrics/phase1_smoke_v3.json"
DEFAULT_OUTPUT = ROOT / "results/metrics/phase1_blind_review_v3.json"
DEFAULT_IDS = [
    "p1_answer_01", "p1_answer_05", "p1_answer_09", "p1_answer_10",
    "p1_answer_15_attention",
]


def export(report_path: Path, output_path: Path, query_ids: list[str]) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_id = {row.get("query_id"): row for row in report.get("records", [])}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"Report is missing selected rows: {', '.join(missing)}")

    rows = []
    for query_id in query_ids:
        source = by_id[query_id]
        result = source.get("result") or {}
        rows.append({
            "query_id": query_id,
            "question": source.get("query"),
            "should_abstain": source.get("should_abstain"),
            "system_status": result.get("status"),
            "system_answer": result.get("answer"),
            "retrieval_confidence": result.get("retrieval_confidence"),
            "generation_attempts": result.get("generation_attempts"),
            "retry_triggered": result.get("retry_triggered"),
            "abstain_reason": result.get("abstain_reason"),
            "generation_contexts": result.get("generation_contexts", []),
            "retrieved_evidence": result.get("debug_full_chunks", []),
            "review": {"answer_correct": None, "grounded": None, "notes": None},
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({
            "review_type": "phase1_independent_blind_review",
            "source_report": str(report_path),
            "gold_and_prior_reviews_excluded": True,
            "review_guidance": (
                "Judge answer correctness/completeness against the paper context and whether "
                "each material answer claim is supported by the exact retrieved/generation context. "
                "The expected answer and prior review labels are intentionally omitted."
            ),
            "records": rows,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-ids", nargs="+", default=DEFAULT_IDS)
    args = parser.parse_args()
    count = export(args.report, args.output, args.query_ids)
    print(f"Exported {count} gold-blind cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
