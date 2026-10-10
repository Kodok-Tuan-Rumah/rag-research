#!/usr/bin/env python3
"""
K1 / Task 11a — MEASURE generation flakiness. Observation only.

Does not modify runtime behaviour, retrieval parameters, thresholds, models, or
gold labels. It calls the existing production path and records what comes back.

Why this exists
---------------
`src/generation/generator.py` sends `response_format={"type": "json_object"}`
to `openai/gpt-oss-20b`, which is a *reasoning* model. With production-sized
context (~2500 chars) the model occasionally emits output that fails Groq's
JSON validation, producing HTTP 400 `json_validate_failed`. That is raised as
RuntimeError at generator.py:394 rather than being handled as a graceful
abstain.

Observed manually: the same query returned ANSWERED and INSUFFICIENT_CONTEXT on
consecutive runs. The rate was not established. This script establishes it, and
identifies which of the four terminal paths fired.

Terminal paths in generator.py
------------------------------
  1. ANSWERED                       — normal
  2. ABSTAIN_EMPTY_OUTPUT           — generator.py:400, `raw_content.strip()` empty
  3. ABSTAIN_SANITY_CHECK           — generator.py:422, sanity_check_answer() failed
  4. ABSTAIN_LLM_STATUS             — generator.py:432, model returned status != ANSWERED
  5. ERROR_JSON_VALIDATE_FAILED     — generator.py:394, Groq 400 json_validate_failed
  6. ERROR_OTHER                    — any other exception

Usage
-----
    python3 scripts/diagnose_generation_flakiness.py --iterations 15
    python3 scripts/diagnose_generation_flakiness.py --iterations 15 \
        --out-prefix results/metrics/k1_generation_flakiness

Requires GROQ_API_KEY. Costs one generation call per iteration.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pathlib
import statistics
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERY = "What is Hybrid Search in RAG?"


class LogCapture(logging.Handler):
    """Collect log messages emitted during a single iteration."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:
            pass

    def find(self, needle: str) -> str | None:
        for r in self.records:
            if needle in r:
                return r
        return None


def classify(capture: LogCapture, response, exc: BaseException | None) -> tuple[str, str]:
    """Map an iteration outcome to one of the terminal paths."""
    if exc is not None:
        text = f"{type(exc).__name__}: {exc}"
        if "json_validate_failed" in text or "Failed to validate JSON" in text:
            return "ERROR_JSON_VALIDATE_FAILED", text[:300]
        return "ERROR_OTHER", text[:300]

    if response is None:
        return "ERROR_OTHER", "no response object returned"

    if response.status == "ANSWERED":
        return "ANSWERED", ""

    marker = capture.find("[GENERATION_EMPTY]")
    if marker:
        return "ABSTAIN_EMPTY_OUTPUT", marker[:300]

    marker = capture.find("[SANITY_CHECK]")
    if marker:
        return "ABSTAIN_SANITY_CHECK", marker[:300]

    marker = capture.find("[ABSTAIN]")
    if marker:
        return "ABSTAIN_LLM_STATUS", marker[:300]

    return "ABSTAIN_LLM_STATUS", "status=INSUFFICIENT_CONTEXT (no [ABSTAIN] log captured)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true",
                    help="opt in to reranking (default False, matching runtime)")
    ap.add_argument("--out-prefix", default="results/metrics/k1_generation_flakiness")
    args = ap.parse_args()

    import src._compat  # noqa: F401  (must precede qdrant_client on Python 3.14)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    from src.generation.generator import GroqGenerator
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.controller.policy_engine import PolicyEngine

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)
    generator = GroqGenerator()
    policy = PolicyEngine()

    plan = policy.initial_plan(args.query)
    chunks = retriever.search(args.query, top_k=args.top_k, rerank=args.rerank)

    print(f"query        : {args.query!r}")
    print(f"top_k        : {args.top_k}  rerank={args.rerank}")
    print(f"query_type   : {plan.query_type}")
    print(f"chunks       : retrieved={len(chunks)}")
    print(f"gen policy   : max_tokens={plan.generation_hint['max_tokens']} "
          f"temperature={plan.generation_hint['temperature']}")
    print(f"iterations   : {args.iterations}\n")

    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)

    rows = []
    for i in range(args.iterations):
        capture = LogCapture()
        root.addHandler(capture)
        exc = None
        response = None
        t0 = time.time()
        try:
            response = generator.generate(
                args.query,
                chunks,
                max_tokens=int(plan.generation_hint["max_tokens"]),
                temperature=float(plan.generation_hint["temperature"]),
            )
        except BaseException as e:  # noqa: BLE001 - we are classifying failures
            exc = e
        finally:
            root.removeHandler(capture)
        elapsed = (time.time() - t0) * 1000

        outcome, detail = classify(capture, response, exc)
        row = {
            "iteration": i,
            "outcome": outcome,
            "status": getattr(response, "status", None),
            "detail": detail,
            "latency_ms": round(elapsed, 1),
            "llm_confidence_score": getattr(response, "llm_confidence_score", None),
            "num_chunks_used": getattr(response, "num_chunks_used", None),
            "answer_head": (getattr(response, "answer", "") or "")[:90],
        }
        rows.append(row)
        flag = "  " if outcome == "ANSWERED" else "!!"
        print(f"  {flag} iter {i:2d} | {outcome:28s} | {row['latency_ms']:7.1f}ms | {detail[:70]}")

    root.setLevel(previous_level)
    store.close()

    # ---- aggregate -------------------------------------------------------
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    n = len(rows)
    answered = counts.get("ANSWERED", 0)
    failed = n - answered

    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "iterations": n,
        "answered": answered,
        "failed": failed,
        "answer_rate": round(answered / n, 4),
        "flaky_or_fail_rate": round(failed / n, 4),
        "outcome_counts": counts,
        "latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[max(0, int(0.95 * n) - 1)], 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
    }

    out_prefix = REPO_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measurement": "K1 Task 11a — generation flakiness (observation only)",
        "config": {
            "query": args.query,
            "top_k": args.top_k,
            "rerank": args.rerank,
            "query_type": str(plan.query_type),
            "generation_max_tokens": plan.generation_hint["max_tokens"],
            "generation_temperature": plan.generation_hint["temperature"],
            "model": generator.model,
            "chunks_retrieved": len(chunks),
        },
        "summary": summary,
        "iterations_detail": rows,
    }
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# K1 Task 11a — Generation Flakiness Measurement",
        "",
        "> Observation only. No runtime parameter, model, threshold, or gold label was changed.",
        "> One generation API call per iteration.",
        "",
        "## Config",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Query | `{args.query}` |",
        f"| top_k / rerank | {args.top_k} / {args.rerank} |",
        f"| query_type | {plan.query_type} |",
        f"| max_tokens / temperature | {plan.generation_hint['max_tokens']} / {plan.generation_hint['temperature']} |",
        f"| Model | `{generator.model}` |",
        f"| Chunks retrieved | {len(chunks)} |",
        f"| Iterations | {n} |",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Answered | {answered}/{n} |",
        f"| Failed / abstained | {failed}/{n} |",
        f"| **Answer rate** | **{summary['answer_rate']:.4f}** |",
        f"| **Flaky-or-fail rate** | **{summary['flaky_or_fail_rate']:.4f}** |",
        f"| Latency median | {summary['latency_ms']['median']:.1f} ms |",
        f"| Latency p95 | {summary['latency_ms']['p95']:.1f} ms |",
        "",
        "## Outcome distribution",
        "",
        "| Terminal path | Count |",
        "|---|---:|",
    ]
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if counts.get("ERROR_JSON_VALIDATE_FAILED", 0) > 0:
        lines += [
            "`ERROR_JSON_VALIDATE_FAILED` occurred. `generator.py:387` sends "
            "`response_format={\"type\": \"json_object\"}` to a reasoning model; with "
            "production-sized context the model occasionally emits output that fails "
            "Groq's JSON validation (HTTP 400). `generator.py:394` raises this as "
            "`RuntimeError` rather than handling it as a graceful abstain.",
            "",
        ]
    if summary["flaky_or_fail_rate"] > 0:
        lines += [
            f"**Non-determinism observed**: {failed} of {n} identical calls did not answer. "
            "`PROJECT_GOALS.md` §10.4 requires metric variance <= 3%; observed variance on "
            "this single query exceeds that by a wide margin. A repeat measurement is "
            "required before drawing a final rate, because a small sample makes the "
            "estimate unstable.",
            "",
        ]
    lines += [
        "## Scope limits",
        "",
        "- Single query, single configuration. This is **not** a system-wide rate.",
        "- Deterministic components (retrieval, RRF, filtering) were shown stable in earlier "
        "phases; the variance here sits in the LLM call, not in retrieval.",
        "- `PROJECT_GOALS.md` §2.3 sets a False Abstain Rate target of <= 10%. A single-query "
        "sample cannot be compared against that target directly.",
        "",
        "## Next",
        "",
        "Fix is deliberately NOT proposed yet. Required first:",
        "1. Confirm which terminal path dominates across more iterations.",
        "2. Decide the contract — should a JSON-validation failure abstain gracefully, "
        "or retry once without `response_format`?",
        "3. Add a regression test before changing `generator.py`.",
    ]
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n=== SUMMARY ===")
    print(f"  answer rate          : {summary['answer_rate']:.4f}  ({answered}/{n})")
    print(f"  flaky-or-fail rate   : {summary['flaky_or_fail_rate']:.4f}  ({failed}/{n})")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v}")
    print(f"\n  wrote {out_prefix.with_suffix('.json')}")
    print(f"  wrote {out_prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())