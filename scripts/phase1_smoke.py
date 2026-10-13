#!/usr/bin/env python3
"""Phase 1 smoke harness: run_single_query --mode full on 20 queries.

Closes PROJECT_GOALS.md §7 Phase 1 exit criterion:
    "run_single_query.py --mode full completes without exception
     on 20 random queries"

The corpus only has 14 unique eval questions, so we sample with replacement
(seeded for reproducibility). Generation is skipped via SMOKE_STOP_BEFORE_GEN=1
to keep the run cheap and focused on retrieval + control flow. Set the flag
to 0 to exercise the full Groq path too.

Usage:
    python scripts/phase1_smoke.py [--count 20] [--seed 1] [--mode full]
    python scripts/phase1_smoke.py --with-generation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_single_query as runner

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "metrics" / "phase1_smoke.json"

VALID_STATUSES = {"ANSWERED", "INSUFFICIENT_CONTEXT", "REJECT", "FAILED"}

DATASET_PATHS = [
    REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v2.json",
    REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json",
    REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v3.json",
]


def load_unique_questions() -> list[str]:
    seen: set[str] = set()
    questions: list[str] = []
    for path in DATASET_PATHS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("records", data) if isinstance(data, dict) else data
        for rec in records:
            q = rec.get("question") or rec.get("query")
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
    return questions


def sample_questions(pool: list[str], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return rng.choices(pool, k=count)


def run_smoke(
    queries: list[str],
    mode: str,
    stop_before_gen: bool,
) -> dict:
    if stop_before_gen:
        os.environ["SMOKE_STOP_BEFORE_GEN"] = "1"
    else:
        os.environ.pop("SMOKE_STOP_BEFORE_GEN", None)

    results: list[dict] = []
    passed = 0
    failed = 0

    for i, query in enumerate(queries, 1):
        entry: dict = {
            "index": i,
            "query": query,
            "status": None,
            "answer_status": None,
            "failure_type": None,
            "error": None,
        }
        try:
            result = runner.run_single_query(
                query=query,
                mode=mode,
                save_output=False,
            )
            entry["status"] = result.get("status")
            entry["answer_status"] = result.get("answer_status")
            entry["failure_type"] = result.get("failure_type")

            if entry["status"] in VALID_STATUSES:
                passed += 1
            else:
                failed += 1
                entry["error"] = f"unexpected status: {entry['status']}"
        except Exception as exc:
            failed += 1
            entry["status"] = "CRASHED"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()

        results.append(entry)
        status_marker = "PASS" if entry["status"] in VALID_STATUSES else "FAIL"
        print(f"  [{i:>2}/{len(queries)}] {status_marker} | {query[:60]}")

    return {
        "summary": {
            "total": len(queries),
            "passed": passed,
            "failed": failed,
            "mode": mode,
            "stop_before_gen": stop_before_gen,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["baseline", "full"], default="full")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pool = load_unique_questions()
    if not pool:
        print("[phase1_smoke] ERROR: no questions found in eval datasets")
        return 1

    queries = sample_questions(pool, args.count, args.seed)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 SMOKE — {args.count} queries | mode={args.mode}")
    print(f"  Pool: {len(pool)} unique questions | seed={args.seed}")
    print(f"  Stop before generation: {not args.with_generation}")
    print(f"{'='*60}\n")

    report = run_smoke(queries, args.mode, stop_before_gen=not args.with_generation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  RESULT: {report['summary']['passed']}/{report['summary']['total']} passed")
    print(f"  Report: {args.output}")
    print(f"{'='*60}\n")

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
