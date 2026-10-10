"""Phase D2: retrieval forensic rank movement + BM25-vs-RRF + reranker attribution.

Uses search_with_trace stage rows (enriched trace with scores) for every gold
query and writes reproducible artifacts:
- gold_rank_movement.json/.md
- retrieval_recall_k_diag.json/.md  (Recall@1/5/10/20/50 + candidate_recall@K)
- per_query_regression_recovery.json/.md  (+counts/mean/median)
- q15_rank_forensic.json/.md
- stage_latency_diag.json/.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src._compat  # noqa: F401
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.evaluation.gold_matching import chunk_ids_match, gold_rank

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
OUT_DIR = ROOT / "results" / "metrics"

STAGES = ["dense", "bm25", "rrf", "reranker"]
KS = [1, 5, 10, 20, 50]


def _rank_row(rows: list[dict], gold_id: str) -> tuple[int | None, dict | None]:
    for i, row in enumerate(rows, start=1):
        cid = str(row.get("chunk_id") or "")
        if chunk_ids_match(gold_id, cid):
            return i, row
    return None, None


def _recall_from(rows_lists: list[list[dict]], gold_id: str, current_rows: list[dict] | None = None) -> dict:
    # recall@K per stage after building rows for one stage
    return {}


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    # accumulation for recall
    recall = {s: {k: [] for k in KS} for s in STAGES}
    recall["final"] = {k: [] for k in KS}
    candidate_recall = {f"candidate@{k}": [] for k in KS}

    rank_movement_rows = []
    per_query_rows = []
    per_query_stage_scores = []

    # latent order for statistical eval: list pipeline input identical fold
    for item in dataset:
        out = retriever.search_with_trace(item["question"], top_k=10, rerank=True)
        trace = out["trace"]
        rrf_rows = trace.get("rrf_rows", [])
        reranked_rows = trace.get("reranked_rows", [])
        dense_rows = trace.get("dense_rows", [])
        bm25_rows = trace.get("bm25_rows", [])

        def find(rows):
            for i, r in enumerate(rows, 1):
                if chunk_ids_match(item["gold_chunk_id"], r.get("chunk_id", "")):
                    return i, r
            return None, None

        dense_rank, dense_row = find(dense_rows)
        bm25_rank, bm25_row = find(bm25_rows)
        rrf_rank, rrf_row = find(rrf_rows)
        rerank_rank, rerank_row = find(reranked_rows)
        final_ids = [str(r.get("chunk_id") or "") for r in trace.get("final_rows", [])]
        final_included = any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in final_ids)

        for stage, rank in [("dense", dense_rank), ("bm25", bm25_rank), ("rrf", rrf_rank), ("reranker", rerank_rank), ("final", (gold_rank(item["gold_chunk_id"], final_ids) or None))]:
            for k in KS:
                recall[stage][k].append(1 if rank is not None and rank <= k else 0)

        # candidate recall: does gold appear in the rrf candidates pool (candidate_k=40 typically)
        for k in KS:
            candidate_recall[f"candidate@{k}"].append(1 if rrf_rank is not None and rrf_rank <= k else 0)

        cls = "NO_FAILURE"
        if not final_included:
            if dense_rank is None and bm25_rank is None and rrf_rank is None:
                cls = "RETRIEVAL_MISS"
            elif rrf_rank is None:
                cls = "FUSION_REGRESSION"
            elif rerank_rank is None and rrf_rank is not None:
                cls = "RERANKER_REGRESSION"
            elif rerank_rank is not None and rerank_rank > 10:
                cls = "TOP_K_CUTOFF"
            elif rrf_rank is not None and rerank_rank is not None and rrf_rank > bm25_rank if bm25_rank else False:
                cls = "RERANKER_REGRESSION"
            else:
                cls = "TOP_K_CUTOFF"
        elif not final_included:
            cls = "CROSS_MISS"
        # NO_FAILURE stays default when final_included True and final_ids has gold

        row = {
            "query_id": item["query_id"],
            "gold_chunk_id": item["gold_chunk_id"],
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
            "rrf_rank": rrf_rank,
            "reranker_rank": rerank_rank,
            "final_status": "in" if final_included else "out",
            "rank_delta_rrf": None if None in (dense_rank, bm25_rank) else (rrf_rank - bm25_rank if rrf_rank and bm25_rank else None),
            "rank_delta_reranker": None if None in (rrf_rank, rerank_rank) else (rerank_rank - rrf_rank if rrf_rank and rerank_rank else None),
            "failure_stage": cls,
            "dense_score": None if dense_row is None else dense_row.get("score"),
            "bm25_score": None if bm25_row is None else bm25_row.get("score"),
            "rrf_score": None if rrf_row is None else rrf_row.get("score"),
            "reranker_score": None if rerank_row is None else rerank_row.get("score"),
            "final_rank": gold_rank(item["gold_chunk_id"], final_ids),
            "latency_trace": trace.get("latency_ms", {}),
        }
        rank_movement_rows.append(row)
        per_query_rows.append(row)

    store.close()

    snapshot = {stage: {} for stage in STAGES + ["final"]}
    for stage in STAGES + ["final"]:
        for k in KS:
            vals = recall[stage][k]
            snapshot[stage][f"recall@{k}"] = round(sum(vals) / max(len(vals), 1), 4)
    for k in KS:
        vals = candidate_recall[f"candidate@{k}"]
        snapshot.setdefault("candidate", {})[f"recall@{k}"] = round(sum(vals) / max(len(vals), 1), 4)

    # Persists rank movement
    (OUT_DIR / "gold_rank_movement.json").write_text(json.dumps(rank_movement_rows, indent=2, ensure_ascii=False))
    lines = ["# Gold Rank Movement", "", "| query_id | gold_chunk_id | dense_rank | bm25_rank | rrf_rank | reranker_rank | final_status | Δrrf | Δrerank | failure_stage |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rank_movement_rows:
        lines.append(f"| {r['query_id']} | {r['gold_chunk_id']} | {r['dense_rank']} | {r['bm25_rank']} | {r['rrf_rank']} | {r['reranker_rank']} | {r['final_status']} | {r['rank_delta_rrf']} | {r['rank_delta_reranker']} | {r['failure_stage']} |")
    (OUT_DIR / "gold_rank_movement.md").write_text("\n".join(lines) + "\n")

    (OUT_DIR / "retrieval_recall_k_diag.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    md = ["# Retrieval Recall@K Diagnostic", "", "```json", json.dumps(snapshot, indent=2, ensure_ascii=False), "```", ""]
    (OUT_DIR / "retrieval_recall_k_diag.md").write_text("\n".join(md))

    # Per-query stages & regression counts
    import statistics

    deltas_rrf = []
    deltas_reranker = []
    for r in rank_movement_rows:
        if r['rank_delta_rrf'] is not None:
            deltas_rrf.append(r['rank_delta_rrf'])
        if r['rank_delta_reranker'] is not None:
            deltas_reranker.append(r['rank_delta_reranker'])

    summary = {
        "count_improved": sum(1 for d in deltas_reranker if d < 0),
        "count_regressed": sum(1 for d in deltas_reranker if d > 0),
        "count_unchanged": sum(1 for d in deltas_reranker if d == 0),
        "mean_rank_delta": None if not deltas_reranker else round(statistics.mean(deltas_reranker), 4),
        "median_rank_delta": None if not deltas_reranker else statistics.median(deltas_reranker),
    }
    summary['total'] = len(deltas_reranker)

    (OUT_DIR / "per_query_regression_recovery.json").write_text(json.dumps({'rows': per_query_rows, 'summary': summary}, indent=2, ensure_ascii=False))
    md = ["# Per-query Regression / Recovery", "", "```json", json.dumps(summary, indent=2), "```", "", "| query_id | gold_chunk_id | dense_rank | bm25_rank | rrf_rank | reranker_rank | final_status | failure_stage | Δrrf | Δrerank |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_query_rows:
        md.append(f"| {r['query_id']} | {r['gold_chunk_id']} | {r['dense_rank']} | {r['bm25_rank']} | {r['rrf_rank']} | {r['reranker_rank']} | {r['final_status']} | {r['failure_stage']} | {r['rank_delta_rrf']} | {r['rank_delta_reranker']} |")
    (OUT_DIR / "per_query_regression_recovery.md").write_text("\n".join(md) + "\n")

    # q15 forensic artifact (reuse stage scores)
    q15 = next((r for r in rank_movement_rows if r['query_id'] == 'q15'), None)
    if q15:
        (OUT_DIR / "q15_rank_forensic.json").write_text(json.dumps(q15, indent=2, ensure_ascii=False))
        (OUT_DIR / "q15_rank_forensic.md").write_text("# q15 Rank Forensic\n\n```json\n" + json.dumps(q15, indent=2, ensure_ascii=False) + "\n```\n")

    # Latency diag for stages from trace latency_ms entries
    lat_entries = []
    for r in rank_movement_rows:
        lat = r.get('latency_trace', {})
        lat_entries.append({k: v for k, v in lat.items() if isinstance(v, (int, float))})

    def percentile(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        idx = int(q / 100 * len(s))
        if idx >= len(s):
            idx = len(s) - 1
        return s[idx]

    latency_summary = {}
    for key in ['dense', 'bm25', 'fusion', 'reranker', 'fuse_and_rerank', 'total']:
        vals = [x[key] for x in lat_entries if key in x]
        if vals:
            ss = sorted(vals)
            midx = len(ss) // 2
            pidx = max(0, int(0.95 * len(ss)) - 1) if len(ss) > 1 else 0
            latency_summary[key] = {
                'p50_ms': round(ss[midx], 4) if vals else None,
                'p95_ms': round(ss[pidx], 4) if vals else None,
                'min_ms': min(vals),
                'max_ms': max(vals),
                'mean_ms': round(sum(vals)/len(vals), 4),
            }
    if latency_summary:
        (OUT_DIR / "stage_latency_diag.json").write_text(json.dumps(latency_summary, indent=2, ensure_ascii=False))
        (OUT_DIR / "stage_latency_diag.md").write_text("# Stage Latency Diagnostic\n\n```json\n" + json.dumps(latency_summary, indent=2, ensure_ascii=False) + "\n```\n")

    print("[done] phase-d2 artifacts written")


if __name__ == "__main__":
    main()
