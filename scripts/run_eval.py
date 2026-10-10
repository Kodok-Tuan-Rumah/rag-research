"""Answer-quality evaluation — RAGAS over the canonical labeled dataset.

Replaces the legacy hardcoded `data/processed/ground_truth_qa.json` path. That
file does not exist in this repo, so `run_eval.py` could not run at all and the
Phase 2 exit criterion "run_eval.py runs end to end" was unmet regardless of
RAGAS being importable. The legacy script also read `item["ground_truth"]` with
an `item["gold_answer"]` fallback, neither of which matched any shipped schema.

Datasets now come from data/evaluation/ and are read through
dataset_records(), the same normalizer scripts/run_retrieval_eval.py uses, so
v1 / v2 / v3 all load. v2 carries a verified `ground_truth` answer string,
which RAGAS needs for Context Recall and Answer Correctness; v1 and v3 carry
`gold_answer` instead and are accepted via the same fallback.

Retrieval is held identical to run_retrieval_eval.py (top_k=10, rerank=True) so
generation results sit on the same evidence chain as the frozen retrieval
baseline in results/metrics/k1_baseline_freeze.json.

Two measurements are kept apart on purpose:

  - pipeline behaviour: per-query answered/abstained, abstain reason, answer
    rate. This is the coverage number Phase 2 needs and that retrieval-only
    evaluation cannot produce.
  - answer quality: RAGAS, computed only over queries that actually answered.

An abstained query returns the literal string "INSUFFICIENT_CONTEXT" as its
answer. Scoring that with RAGAS would measure the abstain machinery rather than
the generator, and would inflate the hallucination rate. Abstains are counted,
not scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src._compat  # noqa: F401  must precede qdrant_client / ragas on 3.14

from dotenv import load_dotenv

load_dotenv()

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGSample
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.generation.generator import GroqGenerator
from src.evaluation.gold_schema import dataset_records
from src.api.config import settings

DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v2.json"
OUTPUT_RAGAS = REPO_ROOT / "results" / "metrics" / "ragas_results.json"
OUTPUT_REPORT = REPO_ROOT / "results" / "metrics" / "generation_run_report.json"

ABSTAIN_ANSWER = "INSUFFICIENT_CONTEXT"


def _chunk_text(chunk) -> str:
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return getattr(chunk, "text", "")


def _is_abstain(resp) -> bool:
    """An abstain is a structured decision, not a failure to produce output."""
    return (
        getattr(resp, "retrieval_method", "") == "abstained"
        or getattr(resp, "status", "") == ABSTAIN_ANSWER
        or getattr(resp, "answer", "") == ABSTAIN_ANSWER
    )


def _ground_truth(item: dict) -> str | None:
    """v2 ships `ground_truth`; v1/v3 ship `gold_answer`. Both are verified."""
    for field in ("ground_truth", "gold_answer"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _score_existing_report(
    report_path: pathlib.Path | None,
    out_ragas: pathlib.Path,
    out_report: pathlib.Path,
    judge_model: str | None,
) -> dict:
    """Score a pipeline run that was already produced.

    Opens no Qdrant collection and makes no generation call, so it can be run
    at any time without risking the answers already paid for. The cost is one
    judge call per sample per metric, which is where a rate limit bites.
    """
    if report_path is None:
        raise SystemExit("--stage ragas requires --from-report <pipeline report>")
    report_path = pathlib.Path(report_path)
    if not report_path.exists():
        raise SystemExit(f"report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    judge_model = judge_model or report.get("config", {}).get(
        "judge", settings.groq_model_fast
    ).replace("+json_mode", "")

    samples: list[RAGSample] = []
    skipped = {"abstained": 0, "errors": 0, "no_ground_truth": 0, "no_contexts": 0}
    for row in report.get("queries", []):
        if row.get("status") == "ERROR":
            skipped["errors"] += 1
            continue
        if not row.get("answered"):
            skipped["abstained"] += 1
            continue
        if not row.get("ground_truth"):
            skipped["no_ground_truth"] += 1
            continue
        contexts = [c for c in (row.get("contexts") or []) if c]
        if not contexts:
            # Reports written before the two-stage split stored only
            # context_chunk_ids, which cannot rebuild a RAG sample.
            skipped["no_contexts"] += 1
            continue
        samples.append(RAGSample(
            question=row["question"],
            answer=row["answer"],
            contexts=contexts,
            ground_truth=row["ground_truth"],
        ))

    print(f"[RAGAS] Scoring {len(samples)} sample(s) from {report_path.name}")
    print(f"  Judge     : {judge_model} + json_mode")
    print(f"  Pipeline  : {json.dumps(report.get('pipeline', {}), ensure_ascii=False)}")
    print(f"  Excluded  : {json.dumps(skipped, ensure_ascii=False)}")

    report["scoring"] = {
        "samples_submitted": len(samples),
        **{f"excluded_{k}": v for k, v in skipped.items()},
    }
    report["stage"] = "ragas"

    if not samples:
        print("[RAGAS] Nothing to score. Every query abstained, errored, or has "
              "no reference answer. Re-run --stage pipeline to regenerate a "
              "report that carries contexts.")
        report["ragas"] = None
        Path(out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    evaluator = RAGASEvaluator(llm_model=judge_model, include_answer_correctness=True)
    result = evaluator.evaluate(
        samples,
        exp_id=report.get("exp_id")
        or f"{pathlib.Path(report['dataset']).stem}_k{report['config'].get('top_k')}"
           f"_{judge_model}",
        config=report.get("config"),
    )
    evaluator.save_result(result, str(out_ragas))

    report["ragas"] = asdict(result)
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(
    dataset_path: Path,
    out_ragas: Path,
    out_report: Path,
    limit: int | None = None,
    query_ids: list[str] | None = None,
    top_k: int = 10,
    rerank: bool = True,
    judge_model: str | None = None,
    stage: str = "both",
    from_report: pathlib.Path | None = None,
) -> dict:
    if stage == "ragas":
        return _score_existing_report(
            from_report, out_ragas, out_report, judge_model
        )

    raw = dataset_path.read_text(encoding="utf-8")
    dataset_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    records = dataset_records(json.loads(raw))

    if query_ids:
        wanted = set(query_ids)
        records = [r for r in records if r.get("query_id") in wanted]
        missing = wanted - {r.get("query_id") for r in records}
        if missing:
            raise ValueError(f"query_ids not present in dataset: {sorted(missing)}")
    if limit is not None:
        records = records[:limit]

    judge_model = judge_model or settings.groq_model_fast

    print(f"[Started] Answer-quality evaluation ({len(records)} samples)...")
    print(f"  Stage     : {stage}")
    print(f"  Dataset   : {dataset_path.name}  sha256={dataset_sha[:12]}")
    print(f"  Generator : {settings.groq_model}")
    print(f"  Judge     : {judge_model} + json_mode")
    print(f"  Retrieval : Hybrid RRF + rerank={rerank}, top_k={top_k}")

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model=settings.groq_model)
    evaluator = RAGASEvaluator(
        llm_model=judge_model,
        include_answer_correctness=True,
    )

    rows: list[dict] = []
    samples: list[RAGSample] = []
    total = len(records)

    try:
        for i, item in enumerate(records, start=1):
            qid = item.get("query_id", f"idx{i:02d}")
            question = item["question"]
            reference = _ground_truth(item)
            print(f"[{i}/{total}] {qid} {question[:60]}...")

            t0 = time.perf_counter()
            try:
                chunks = retriever.search(question, top_k=top_k, rerank=rerank)
                resp = generator.generate(question, chunks)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                print(f"  [Warning] {type(exc).__name__}: {exc}")
                rows.append({
                    "query_id": qid,
                    "question": question,
                    "answered": False,
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "latency_ms": elapsed,
                    "has_ground_truth": reference is not None,
                })
                time.sleep(3)
                continue

            abstained = _is_abstain(resp)
            contexts = [_chunk_text(c) for c in (chunks or [])]
            contexts = [c for c in contexts if c]

            rows.append({
                "query_id": qid,
                "question": question,
                "answered": not abstained,
                "status": getattr(resp, "status", "UNKNOWN"),
                "reason": getattr(resp, "reason", None),
                # Persisted so `--stage ragas` can score this run later without
                # re-running retrieval or generation. RAGAS needs the chunk
                # text; context_chunk_ids alone is not enough to rebuild a
                # sample.
                "ground_truth": reference,
                "answer": None if abstained else resp.answer,
                "contexts": contexts,
                "num_chunks_retrieved": getattr(resp, "num_chunks_retrieved", 0),
                "num_chunks_used": getattr(resp, "num_chunks_used", 0),
                "context_chunk_ids": getattr(resp, "context_chunk_ids", []),
                "confidence_score": getattr(resp, "confidence_score", 0.0),
                "llm_confidence_score": getattr(resp, "llm_confidence_score", None),
                "retrieval_decision": getattr(resp, "retrieval_decision", None),
                "post_generation_decision": getattr(resp, "post_generation_decision", None),
                "retrieval_ms": getattr(resp, "latency_context_build_ms", None),
                "generation_ms": getattr(resp, "latency_generation_ms", None),
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "num_contexts_for_ragas": len(contexts),
                "has_ground_truth": reference is not None,
            })

            if abstained:
                print(f"  -> ABSTAIN ({getattr(resp, 'reason', None) or 'no reason recorded'})")
            elif reference is None:
                # RAGAS with_reference metrics are skipped wholesale when any
                # sample lacks a reference, so a missing ground truth silently
                # removes ContextRecall and AnswerCorrectness from the whole run.
                # Keep such samples out of the scored set instead.
                print("  -> skipped from scoring: no ground_truth in dataset row")
            else:
                samples.append(RAGSample(
                    question=question,
                    answer=resp.answer,
                    contexts=contexts,
                    ground_truth=reference,
                ))
                print("  -> answered, queued for RAGAS")

            time.sleep(0.5)
    finally:
        store.close()

    answered = [r for r in rows if r["answered"]]
    abstained_rows = [r for r in rows if not r["answered"] and r["status"] != "ERROR"]
    errors = [r for r in rows if r["status"] == "ERROR"]
    abstain_reasons = Counter(
        (r.get("reason") or "no reason recorded").split("|")[0].strip()
        for r in abstained_rows
    )

    report = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "stage": stage,
        "config": {
            "generator": settings.groq_model,
            "judge": f"{judge_model}+json_mode",
            "retrieval": f"hybrid_rerank={rerank}",
            "top_k": top_k,
            "min_top1_score": settings.retrieval_min_top1_score,
            "embed_model": settings.embed_model,
        },
        "pipeline": {
            "total_queries": len(rows),
            "answered": len(answered),
            "abstained": len(abstained_rows),
            "errors": len(errors),
            "answer_rate": round(len(answered) / len(rows), 4) if rows else 0.0,
            "abstain_reasons": dict(abstain_reasons),
        },
        "scoring": {
            "samples_submitted": len(samples),
            "excluded_abstained": len(abstained_rows),
            "excluded_errors": len(errors),
            "excluded_no_ground_truth": len(answered) - len(samples),
        },
        "queries": rows,
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] pipeline report -> {out_report}")

    pipeline = report["pipeline"]
    print(f"\n[Pipeline] answered {pipeline['answered']}/{pipeline['total_queries']} "
          f"(answer_rate={pipeline['answer_rate']}), abstained {pipeline['abstained']}, "
          f"errors {pipeline['errors']}")
    if abstain_reasons:
        for reason, count in abstain_reasons.most_common():
            print(f"    {count:>3}x  {reason}")

    if not samples:
        print("[RAGAS] No scored samples: every query abstained or errored. "
              "Answer quality is not measurable for this run.")
        report["ragas"] = None
        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    if stage == "pipeline":
        # Stop here. Generation costs one Groq call per query; scoring costs one
        # judge call per sample per metric, which is several times more. Keeping
        # them apart means a rate limit during scoring cannot destroy a
        # completed pipeline run, and the pipeline can be re-scored later from
        # this file without spending a single generation call again.
        print(f"\n[RAGAS] Skipped (--stage pipeline). {len(samples)} sample(s) "
              f"saved in {out_report}.")
        print(f"       Score them later with:")
        print(f"         python scripts/run_eval.py --stage ragas "
              f"--from-report {out_report}")
        report["ragas"] = None
        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    print(f"\n[RAGAS] Evaluating {len(samples)} samples...")
    result = evaluator.evaluate(
        samples,
        exp_id=f"{dataset_path.stem}_k{top_k}_rerank{int(rerank)}_{judge_model}",
        config=report["config"],
    )
    evaluator.save_result(result, str(out_ragas))
    print("[OK] Done! Results saved.")

    report["ragas"] = asdict(result)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out-ragas", default=str(OUTPUT_RAGAS))
    parser.add_argument("--out-report", default=str(OUTPUT_REPORT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-ids", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--stage", choices=("both", "pipeline", "ragas"), default="both",
        help="both: generate then score. pipeline: generate only, one Groq call "
             "per query, cheap and safe to re-run. ragas: score an existing "
             "report, no Qdrant and no generation calls.",
    )
    parser.add_argument(
        "--from-report", default=None,
        help="pipeline report to score; required by --stage ragas",
    )
    args = parser.parse_args()

    run(
        Path(args.dataset),
        Path(args.out_ragas),
        Path(args.out_report),
        limit=args.limit,
        query_ids=args.query_ids,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        judge_model=args.judge_model,
        stage=args.stage,
        from_report=Path(args.from_report) if args.from_report else None,
    )


if __name__ == "__main__":
    main()