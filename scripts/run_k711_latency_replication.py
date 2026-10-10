"""Replicate K7.9 latency comparisons with paired, interleaved measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data/evaluation/retrieval_eval_dataset_v2.json"
CHUNKS_PATH = ROOT / "data/processed/chunks_semantic_v2.json"
DOCUMENTS_PATH = ROOT / "data/processed/documents.json"
QDRANT_PATH = ROOT / "data/qdrant_storage"
K79_PATH = ROOT / "results/metrics/rrf_vs_reranker5_decision_20261005T141527Z.json"
MODES = ("rrf_only", "reranker5", "reranker20")
COUNTS = {"rrf_only": 0, "reranker5": 5, "reranker20": 20}
MAX_LENGTH = 326
TOP_K = 10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def round_ms(seconds: float | None) -> float | None:
    return round(seconds * 1000.0, 3) if seconds is not None else None


def rank_of(ids: list[str], gold_ids: list[str]) -> int | None:
    gold = set(gold_ids)
    return next((index for index, chunk_id in enumerate(ids, 1) if chunk_id in gold), None)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
        "median_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": round(percentile(values, 0.95), 3) if values else None,
        "min_ms": round(min(values), 3) if values else None,
        "max_ms": round(max(values), 3) if values else None,
        "stdev_ms": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0 if values else None,
    }


def interleaved_modes(repetition: int, query_index: int) -> list[str]:
    offset = (repetition + query_index) % len(MODES)
    return [*MODES[offset:], *MODES[:offset]]


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K7.11 — Paired Latency Replication",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Status: **{report['status']}**",
        f"- Repetitions per query/mode: {report['configuration']['repetitions']}",
        f"- Interleaving: `{report['configuration']['interleaving']}`",
        f"- Models: `{report['configuration']['embedding_model']}` / `{report['configuration']['reranker_model']}`",
        f"- Reranker max length: `{report['configuration']['max_length']}`; device: `{report['configuration']['device']}`",
        "",
        "## Latency summary",
        "",
        "| Mode | Total median ms | Total p95 ms | Total min–max ms | Reranker median ms | Reranker p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        summary = report.get("summary", {}).get(mode, {})
        total = summary.get("total", {})
        rerank = summary.get("reranker", {})
        lines.append(
            f"| {mode} | {total.get('median_ms')} | {total.get('p95_ms')} | "
            f"{total.get('min_ms')}–{total.get('max_ms')} | "
            f"{rerank.get('median_ms')} | {rerank.get('p95_ms')} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas",
            "",
            "| Comparison | Total median delta ms | Total mean delta ms | Reranker median delta ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, delta in report.get("paired_deltas", {}).items():
        lines.append(
            f"| {key} | {delta['total']['median_ms']} | {delta['total']['mean_ms']} | "
            f"{delta['reranker']['median_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Per-query/per-repetition latency",
            "",
            "| Rep | Query | Mode | Order | Total ms | Dense ms | BM25 ms | Fusion ms | Reranker ms | Filtering ms |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["measurements"]:
        latency = row["latency_ms"]
        lines.append(
            f"| {row['repetition']} | {row['query_id']} | {row['mode']} | {row['order_in_pair']} | "
            f"{latency.get('total')} | {latency.get('dense')} | {latency.get('bm25')} | "
            f"{latency.get('fusion')} | {latency.get('reranker')} | {latency.get('filtering')} |"
        )
    lines.extend(
        [
            "",
            "Measurements are serialized query-by-query. Each query's three modes rotate order across repetitions; "
            "the first retrieval and model warm-up are excluded. The report records stage latency from the retriever trace.",
            "",
        ]
    )
    return "\n".join(lines)


def update_summary(report: dict[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for row in report["measurements"]:
        grouped[row["mode"]].append(row)

    summaries: dict[str, Any] = {}
    for mode, rows in grouped.items():
        summaries[mode] = {
            stage: summarize(
                [float(row["latency_ms"][stage]) for row in rows if row["latency_ms"].get(stage) is not None]
            )
            for stage in ("total", "dense", "bm25", "fusion", "reranker", "filtering")
        }
    report["summary"] = summaries

    lookup = {
        (row["repetition"], row["query_id"], row["mode"]): row
        for row in report["measurements"]
    }
    comparisons = {
        "reranker5_minus_rrf_only": ("reranker5", "rrf_only"),
        "reranker20_minus_reranker5": ("reranker20", "reranker5"),
    }
    paired: dict[str, Any] = {}
    for label, (left_mode, right_mode) in comparisons.items():
        deltas = {stage: [] for stage in ("total", "reranker")}
        for rep, query_id in sorted({(row["repetition"], row["query_id"]) for row in report["measurements"]}):
            left = lookup.get((rep, query_id, left_mode))
            right = lookup.get((rep, query_id, right_mode))
            if left is None or right is None:
                continue
            for stage in deltas:
                left_value = left["latency_ms"].get(stage)
                right_value = right["latency_ms"].get(stage)
                if left_value is not None and right_value is not None:
                    deltas[stage].append(float(left_value) - float(right_value))
        paired[label] = {stage: summarize(values) for stage, values in deltas.items()}
    report["paired_deltas"] = paired

    quality: dict[str, Any] = {}
    for mode, rows in grouped.items():
        hits = [row["hit_at_10"] for row in rows]
        reciprocal_ranks = [
            1.0 / row["final_gold_rank"] if row["final_gold_rank"] is not None and row["final_gold_rank"] <= TOP_K else 0.0
            for row in rows
        ]
        quality[mode] = {
            "measurement_count": len(rows),
            "hit_at_10_rate": round(sum(hits) / len(hits), 4) if hits else None,
            "mrr_at_10": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None,
        }
    report["quality_sanity"] = quality


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    update_summary(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(make_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to estimate variability")
    if args.max_length != MAX_LENGTH:
        parser.error(f"This controlled replication requires --max-length {MAX_LENGTH}")
    if os.getenv("QDRANT_URL"):
        parser.error("QDRANT_URL is set; refusing to run against a remote Qdrant instance")
    if not QDRANT_PATH.exists() or not (QDRANT_PATH / "meta.json").exists():
        parser.error(f"Local Qdrant storage is unavailable at {QDRANT_PATH}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"K7.11-{stamp}"
    json_path = args.out_json or ROOT / f"results/metrics/k711_latency_replication_{stamp}.json"
    markdown_path = args.out_md or ROOT / f"results/metrics/k711_latency_replication_{stamp}.md"
    if json_path.exists() or markdown_path.exists():
        parser.error("Output path already exists; choose a new path to preserve prior artifacts")

    dataset_envelope = read_json(DATASET_PATH)
    dataset = dataset_envelope["records"]
    dataset_ids = [row["query_id"] for row in dataset]
    if len(set(dataset_ids)) != len(dataset_ids):
        parser.error("Dataset query_id values are not unique")
    k79 = read_json(K79_PATH)
    if [row["query_id"] for row in k79["modes"]["rrf_only"]["per_query_results"]] != dataset_ids:
        parser.error("Current dataset query IDs/order differ from the K7.9 artifact")
    expected = {
        "dataset": k79["identity"]["dataset_file_sha256"],
        "documents": k79["identity"]["documents_sha256"],
        "chunks": k79["identity"]["chunks_sha256"],
    }
    actual = {
        "dataset": sha256(DATASET_PATH),
        "documents": sha256(DOCUMENTS_PATH),
        "chunks": sha256(CHUNKS_PATH),
    }
    if actual != expected:
        parser.error("Dataset or corpus hashes differ from the K7.9 reference; refusing incomparable run")

    from src.api.config import settings
    from src.retrieval.model_registry import ModelRegistry

    expected_embedding = k79["common_configuration"]["embedding_model"]
    expected_reranker = k79["common_configuration"]["reranker_model"]
    if settings.embed_model != expected_embedding or settings.reranker_model != expected_reranker:
        parser.error("Configured embedding/reranker model differs from the K7.9 reference")
    if settings.candidate_k != k79["common_configuration"]["candidate_k_setting"]:
        parser.error("Configured candidate_k differs from the K7.9 reference")

    registry = ModelRegistry.get()
    registry.embedder.to("cpu")
    registry.reranker.model.to("cpu")
    registry.reranker.max_seq_length = MAX_LENGTH
    device = str(getattr(registry.reranker, "device", "cpu"))
    if "cpu" not in device.lower():
        parser.error(f"Reranker is not on CPU: {device}")

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store, bm25_chunks_path=str(CHUNKS_PATH))
    if retriever.reranker is None:
        store.close()
        parser.error("Reranker failed to load")
    retriever.reranker.max_seq_length = MAX_LENGTH

    report: dict[str, Any] = {
        "run_label": "K7.11_PAIRED_INTERLEAVED_LATENCY_REPLICATION",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_run_id": k79.get("run_id"),
        "identity": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "working_tree": (
                "dirty"
                if subprocess.run(
                    ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
                ).stdout.strip()
                else "clean"
            ),
            "dataset_content_sha256": dataset_envelope.get("dataset_sha256"),
            "dataset_file_sha256": actual["dataset"],
            "documents_sha256": actual["documents"],
            "chunks_sha256": actual["chunks"],
        },
        "configuration": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "query_ids": dataset_ids,
            "query_count": len(dataset),
            "embedding_model": settings.embed_model,
            "reranker_model": settings.reranker_model,
            "max_length": MAX_LENGTH,
            "device": device,
            "candidate_k_setting": settings.candidate_k,
            "effective_candidate_pool": retriever._choose_candidate_k(TOP_K),
            "top_k": TOP_K,
            "rrf_k": retriever.rrf_k,
            "dense_weight": 0.5,
            "bm25_weight": 0.5,
            "reranker_counts": COUNTS,
            "repetitions": args.repetitions,
            "interleaving": "per query, rotate [rrf_only, reranker5, reranker20] order by repetition+query index",
            "warmup": "one q01 reranker20 retrieval discarded before timed repetitions",
            "generation": False,
            "ragas": False,
            "read_only_qdrant": True,
        },
        "warmup": {},
        "measurements": [],
        "summary": {},
        "paired_deltas": {},
    }
    write_report(report, json_path, markdown_path)

    original_rerank = retriever._rerank
    captured: dict[str, list[str]] = {}

    def capped_rerank(query: str, candidates: list, text_of=None):
        cap = active_mode_count[0]
        actual_candidates = candidates[:cap]
        captured["ids"] = [
            str(c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", ""))
            for c in actual_candidates
        ]
        return original_rerank(query, actual_candidates, text_of=text_of)

    active_mode_count = [20]
    reranker = retriever.reranker
    warmup_query = dataset[0]
    warmup_start = time.perf_counter()
    retriever._rerank = capped_rerank
    active_mode_count[0] = 20
    warmup = retriever.search_with_trace(
        warmup_query["question"], top_k=TOP_K, dense_weight=0.5, bm25_weight=0.5, rerank=True,
        use_multi_query=False, use_hyde=False,
    )
    retriever._rerank = original_rerank
    report["warmup"] = {
        "query_id": warmup_query["query_id"],
        "elapsed_ms_excluded": round_ms(time.perf_counter() - warmup_start),
        "reranker_input_count": len(captured.get("ids", [])),
        "results_discarded": True,
    }
    _ = warmup
    write_report(report, json_path, markdown_path)

    base_trace_by_query: dict[str, list[str]] = {}
    measurement_started = time.perf_counter()
    try:
        for repetition in range(args.repetitions):
            for query_index, item in enumerate(dataset):
                for order_in_pair, mode in enumerate(interleaved_modes(repetition, query_index), 1):
                    cap = COUNTS[mode]
                    captured["ids"] = []
                    retriever._rerank = capped_rerank if cap else original_rerank
                    active_mode_count[0] = cap or 20
                    started = time.perf_counter()
                    result = retriever.search_with_trace(
                        item["question"],
                        top_k=TOP_K,
                        dense_weight=0.5,
                        bm25_weight=0.5,
                        rerank=bool(cap),
                        use_multi_query=False,
                        use_hyde=False,
                    )
                    wall_ms = round_ms(time.perf_counter() - started)
                    trace = result["trace"]
                    rrf_ids = list(trace.get("rrf_ids") or [])
                    previous_rrf = base_trace_by_query.setdefault(item["query_id"], rrf_ids)
                    stable_rrf = previous_rrf == rrf_ids
                    if not stable_rrf:
                        report.setdefault("warnings", []).append(
                            f"{item['query_id']} RRF order differed between paired conditions/repetitions"
                        )

                    actual_input_ids = captured.get("ids", []) if cap else []
                    final_ids = list(trace.get("final_ids") or [])
                    gold_ids = sorted(str(x) for x in item.get("gold_chunk_ids", []))
                    reranker_status = {
                        "attempted": trace.get("reranker_attempted"),
                        "succeeded": trace.get("reranker_succeeded"),
                        "executed": trace.get("reranker_executed"),
                        "error": trace.get("reranker_error"),
                    }
                    if cap and reranker_status["succeeded"] is not True:
                        raise RuntimeError(
                            f"{item['query_id']} {mode}: reranker did not succeed: {reranker_status}"
                        )
                    if cap and len(actual_input_ids) > cap:
                        raise RuntimeError(f"{item['query_id']} {mode}: reranker input exceeded cap {cap}")

                    latency = trace.get("latency_ms") or {}
                    row = {
                        "repetition": repetition + 1,
                        "query_index": query_index + 1,
                        "order_in_pair": order_in_pair,
                        "mode": mode,
                        "candidate_count": cap,
                        "query_id": item["query_id"],
                        "gold_chunk_ids": gold_ids,
                        "rrf_candidate_ids": rrf_ids,
                        "reranker_input_ids": actual_input_ids,
                        "reranker_input_count": len(actual_input_ids),
                        "rrf_gold_rank": rank_of(rrf_ids, gold_ids),
                        "final_gold_rank": rank_of(final_ids, gold_ids),
                        "hit_at_10": rank_of(final_ids, gold_ids) is not None and rank_of(final_ids, gold_ids) <= TOP_K,
                        "reranker_status": reranker_status,
                        "ranking_mode": trace.get("ranking_mode"),
                        "filter_stage": trace.get("filter_stage"),
                        "latency_ms": {
                            name: round(float(latency[name]), 3) if latency.get(name) is not None else None
                            for name in ("dense", "bm25", "fusion", "reranker", "filtering", "total")
                        },
                        "wall_elapsed_ms": wall_ms,
                        "rrf_order_stable": stable_rrf,
                    }
                    report["measurements"].append(row)
                    write_report(report, json_path, markdown_path)
                    print(
                        f"[{len(report['measurements'])}/{args.repetitions * len(dataset) * len(MODES)}] "
                        f"rep={row['repetition']} query={row['query_id']} mode={mode} "
                        f"order={order_in_pair} total_ms={row['latency_ms']['total']} "
                        f"reranker_ms={row['latency_ms']['reranker']}",
                        flush=True,
                    )
                    retriever._rerank = original_rerank

        expected_measurements = args.repetitions * len(dataset) * len(MODES)
        if len(report["measurements"]) != expected_measurements:
            raise RuntimeError(
                f"Expected {expected_measurements} measurements, got {len(report['measurements'])}"
            )
        if not all(row["rrf_order_stable"] for row in report["measurements"]):
            report["quality_comparison_validity"] = "RRF candidate order varied; compare latency, interpret quality cautiously"
        else:
            report["quality_comparison_validity"] = "RRF candidate order identical across modes/repetitions"
        report["status"] = "completed"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["measurement_wall_elapsed_ms"] = round_ms(time.perf_counter() - measurement_started)
        write_report(report, json_path, markdown_path)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report, json_path, markdown_path)
        raise
    finally:
        retriever._rerank = original_rerank
        store.close()

    print(f"[saved] {json_path}")
    print(f"[saved] {markdown_path}")
    print(f"[complete] {len(report['measurements'])} measurements; elapsed_ms={report['measurement_wall_elapsed_ms']}")


if __name__ == "__main__":
    main()
