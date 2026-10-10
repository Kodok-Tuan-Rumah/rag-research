#!/usr/bin/env python3
"""
K1 Task 11a/11c — MEASURE generation stability. Observation only.

Does not modify runtime behaviour, retrieval parameters, thresholds, models, or
gold labels. It calls the existing production path and records what comes back.

Why this exists
---------------
`src/generation/generator.py` sends `response_format={"type": "json_object"}`
to `openai/gpt-oss-20b`, which is a *reasoning* model. With production-sized
context the model occasionally emits output that fails Groq's JSON validation,
producing HTTP 400 `json_validate_failed`. That is raised as RuntimeError at
generator.py:394 rather than handled as a graceful abstain.

Task 11a observed the same query returning ANSWERED and INSUFFICIENT_CONTEXT on
consecutive runs, and a 0/15 vs 10/10 split between ranking modes on one query.
Task 11c sweeps several queries across both ranking modes to establish whether
that is systemic or specific to one query.

Terminal paths in generator.py
------------------------------
  1. ANSWERED                    — normal
  2. ABSTAIN_EMPTY_OUTPUT        — generator.py:400, `raw_content.strip()` empty
  3. ABSTAIN_SANITY_CHECK        — generator.py:422, sanity_check_answer() failed
  4. ABSTAIN_LLM_STATUS          — generator.py:432, model returned status != ANSWERED
  5. ERROR_JSON_VALIDATE_FAILED  — generator.py:394, Groq 400 json_validate_failed
  6. ERROR_OTHER                 — any other exception

Usage
-----
    # single query (Task 11a behaviour, unchanged)
    python3 scripts/diagnose_generation_flakiness.py --iterations 15
    python3 scripts/diagnose_generation_flakiness.py --iterations 10 --rerank

    # sweep across queries and ranking modes (Task 11c)
    python3 scripts/diagnose_generation_flakiness.py \
        --dataset data/evaluation/retrieval_eval_dataset.json \
        --query-ids q01,q03,q15 --iterations 1 --modes rrf_only,rerank \
        --out-prefix results/metrics/k1_generation_triage

Requires GROQ_API_KEY. Costs one generation call per (query, mode, iteration).
"""
from __future__ import annotations

import argparse
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
DEFAULT_DATASET = "data/evaluation/retrieval_eval_dataset.json"


class LogCapture(logging.Handler):
    """Collect log messages emitted during a single generation call."""

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
    """Map one generation call to one of the terminal paths."""
    if exc is not None:
        text = f"{type(exc).__name__}: {exc}"
        if "json_validate_failed" in text or "Failed to validate JSON" in text \
                or "Failed to generate JSON" in text:
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


def measure(retriever, generator, policy, root, previous_level,
            query: str, rerank: bool, top_k: int, iterations: int,
            verbose: bool = True) -> dict:
    """Run `iterations` generation calls for one (query, ranking mode) pair."""
    plan = policy.initial_plan(query)
    # top_k follows the query policy unless explicitly overridden; this mirrors
    # what the API does when the caller omits top_k.
    effective_top_k = plan.top_k_hint if top_k is None else top_k
    chunks = retriever.search(query, top_k=effective_top_k, rerank=rerank)

    if verbose:
        print(f"\n--- query={query!r} rerank={rerank} ---")
        print(f"    query_type={plan.query_type}  top_k={effective_top_k}  "
              f"chunks={len(chunks)}  max_tokens={plan.generation_hint['max_tokens']}  "
              f"temperature={plan.generation_hint['temperature']}")

    root.setLevel(logging.DEBUG)
    rows = []
    for i in range(iterations):
        capture = LogCapture()
        root.addHandler(capture)
        exc = None
        response = None
        t0 = time.time()
        try:
            response = generator.generate(
                query,
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
        rows.append({
            "iteration": i,
            "outcome": outcome,
            "status": getattr(response, "status", None),
            "detail": detail,
            "latency_ms": round(elapsed, 1),
            "llm_confidence_score": getattr(response, "llm_confidence_score", None),
            "num_chunks_used": getattr(response, "num_chunks_used", None),
            "answer_head": (getattr(response, "answer", "") or "")[:120],
        })
        if verbose:
            flag = "  " if outcome == "ANSWERED" else "!!"
            print(f"    {flag} iter {i:2d} | {outcome:28s} | {rows[-1]['latency_ms']:7.1f}ms")

    root.setLevel(previous_level)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    n = len(rows)
    answered = counts.get("ANSWERED", 0)
    latencies = [r["latency_ms"] for r in rows]
    return {
        "query": query,
        "rerank": rerank,
        "query_type": str(plan.query_type),
        "top_k": effective_top_k,
        "max_tokens": plan.generation_hint["max_tokens"],
        "temperature": plan.generation_hint["temperature"],
        "chunks_retrieved": len(chunks),
        "iterations": n,
        "answered": answered,
        "answer_rate": round(answered / n, 4) if n else None,
        "outcome_counts": counts,
        "latency_ms": {
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "iterations_detail": rows,
    }


def load_queries(dataset_path: pathlib.Path, query_ids: str | None) -> list[dict]:
    from src.evaluation.gold_schema import dataset_records

    records = dataset_records(json.loads(dataset_path.read_text(encoding="utf-8")))
    if query_ids:
        wanted = [q.strip() for q in query_ids.split(",") if q.strip()]
        index = {r["query_id"]: r for r in records}
        missing = [q for q in wanted if q not in index]
        if missing:
            raise SystemExit(f"query ids not in dataset: {missing}")
        return [index[q] for q in wanted]
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--top-k", type=int, default=None,
                    help="override top_k; default follows PolicyEngine per query")
    ap.add_argument("--rerank", action="store_true",
                    help="single-query mode only: opt in to reranking")
    ap.add_argument("--dataset", default=None, help="enable sweep mode")
    ap.add_argument("--query-ids", default=None, help="comma-separated, e.g. q01,q03,q15")
    ap.add_argument("--modes", default="rrf_only,rerank",
                    help="sweep mode: subset of rrf_only,rerank")
    ap.add_argument("--out-prefix", default="results/metrics/k1_generation_flakiness")
    args = ap.parse_args()

    import src._compat  # noqa: F401  (must precede qdrant_client on Python 3.14)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    from src.controller.policy_engine import PolicyEngine
    from src.generation.generator import GroqGenerator
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.retrieval.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)
    generator = GroqGenerator()
    policy = PolicyEngine()
    root = logging.getLogger()
    previous_level = root.level

    sweep_mode = args.dataset is not None or args.query_ids is not None
    results = []

    try:
        if sweep_mode:
            queries = load_queries(REPO_ROOT / args.dataset, args.query_ids)
            modes = [m.strip() for m in args.modes.split(",") if m.strip()]
            print(f"sweep mode | queries={len(queries)} | modes={modes} | "
                  f"iterations={args.iterations} | model={generator.model}")
            for row in queries:
                for mode in modes:
                    results.append(measure(
                        retriever, generator, policy, root, previous_level,
                        query=row["question"],
                        rerank=(mode == "rerank"),
                        top_k=args.top_k,
                        iterations=args.iterations,
                    ))
        else:
            results.append(measure(
                retriever, generator, policy, root, previous_level,
                query=args.query,
                rerank=args.rerank,
                top_k=5 if args.top_k is None else args.top_k,
                iterations=args.iterations,
            ))
    finally:
        store.close()

    out_prefix = REPO_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    total_calls = sum(r["iterations"] for r in results)
    total_answered = sum(r["answered"] for r in results)
    payload = {
        "measurement": "K1 Task 11a/11c — generation stability (observation only)",
        "mode": "sweep" if sweep_mode else "single",
        "model": generator.model,
        "calls": total_calls,
        "calls_answered": total_answered,
        "overall_answer_rate": round(total_answered / total_calls, 4) if total_calls else None,
        "results": results,
    }
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {'K1 Task 11c — Generation Stability Triage' if sweep_mode else 'K1 Task 11a — Generation Flakiness Measurement'}",
        "",
        "> Observation only. No retrieval parameter, model, threshold, or gold label was changed.",
        f"> Model `{generator.model}`. {total_calls} generation call(s).",
        "",
    ]
    if sweep_mode:
        lines += [
            "## Result matrix",
            "",
            "| Query | query_type | top_k | max_tokens | RRF-only | RRF+rerank | Delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        by_key = {(r["query"], r["rerank"]): r for r in results}
        seen_q = []
        for r in results:
            if r["query"] not in seen_q:
                seen_q.append(r["query"])
        for q in seen_q:
            a = by_key.get((q, False))
            b = by_key.get((q, True))
            ar = f"{a['answer_rate']:.2f}" if a and a["answer_rate"] is not None else "-"
            br = f"{b['answer_rate']:.2f}" if b and b["answer_rate"] is not None else "-"
            delta = "-"
            if a and b and a["answer_rate"] is not None and b["answer_rate"] is not None:
                d = b["answer_rate"] - a["answer_rate"]
                delta = f"{d:+.2f}"
            meta = a or b
            lines.append(
                f"| `{meta['query'][:44]}` | {meta['query_type']} | {meta['top_k']} | "
                f"{meta['max_tokens']} | {ar} | {br} | {delta} |"
            )
        lines += ["", "## Per-cell detail", "",
                  "| Query | Rerank | Answered | Outcomes | Median ms |",
                  "|---|---|---:|---|---:|"]
        for r in results:
            counts = ", ".join(f"{k}×{v}" for k, v in sorted(r["outcome_counts"].items()))
            lines.append(
                f"| `{r['query'][:40]}` | {r['rerank']} | {r['answered']}/{r['iterations']} | "
                f"{counts} | {r['latency_ms']['median']} |"
            )
    else:
        r = results[0]
        lines += [
            "## Result", "",
            "| Metric | Value |", "|---|---:|",
            f"| Answered | {r['answered']}/{r['iterations']} |",
            f"| **Answer rate** | **{r['answer_rate']:.4f}** |",
            f"| query_type | {r['query_type']} |",
            f"| top_k / chunks | {r['top_k']} / {r['chunks_retrieved']} |",
            f"| max_tokens / temperature | {r['max_tokens']} / {r['temperature']} |",
            f"| Latency median | {r['latency_ms']['median']} ms |",
            "", "## Outcome distribution", "",
            "| Terminal path | Count |", "|---|---:|",
        ]
        for k, v in sorted(r["outcome_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")

    lines += [
        "", "## Overall", "",
        f"- Calls: **{total_calls}**",
        f"- Answered: **{total_answered}**",
        f"- **Overall answer rate: {payload['overall_answer_rate']:.4f}**" if total_calls else "- no calls -",
        "", "## Scope limits", "",
        "- Sample is small by design. It is a triage signal, not a system-wide rate.",
        "- Says nothing about Recall@10 or MRR@10; those are frozen separately in "
        "`k1_baseline_freeze.md`.",
        "- Any comparison with the README historical baseline must state the dataset "
        "version and ranking mode, because neither is implicit.",
        "",
        "## Reproduce", "", "```bash",
        "python3 scripts/diagnose_generation_flakiness.py \\",
        f"    --dataset {args.dataset or 'data/evaluation/retrieval_eval_dataset.json'} \\",
        f"    --query-ids {args.query_ids or '(all)'} --iterations {args.iterations} \\",
        f"    --out-prefix {args.out_prefix}",
        "```",
    ]
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n=== OVERALL ===")
    print(f"  calls={total_calls}  answered={total_answered}  "
          f"answer_rate={payload['overall_answer_rate']}")
    if sweep_mode:
        for r in results:
            print(f"    {r['query'][:44]:46s} rerank={str(r['rerank']):5s} "
                  f"{r['answered']}/{r['iterations']}  {r['outcome_counts']}")
    print(f"\n  wrote {out_prefix.with_suffix('.json')}")
    print(f"  wrote {out_prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())