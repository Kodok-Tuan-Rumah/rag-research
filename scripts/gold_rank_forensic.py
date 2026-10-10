"""Phase D1: ranking forensic and cutoff diagnostics.

Produces:
  results/metrics/q15_rank_forensic.json/.md
  results/metrics/gold_rank_movement.json/.md
  results/metrics/retrieval_recall_k_diag.json/.md
  results/metrics/per_query_regression_recovery.json/.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src._compat  # noqa: F401
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever, rrf_fuse
from src.evaluation.gold_matching import chunk_ids_match, gold_rank
from src.retrieval.chunk_filters import is_low_quality_chunk as _cf_is_low_quality

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"

OUT_Q15_JSON = ROOT / "results" / "metrics" / "q15_rank_forensic.json"
OUT_Q15_MD = ROOT / "results" / "metrics" / "q15_rank_forensic.md"
OUT_MOV_JSON = ROOT / "results" / "metrics" / "gold_rank_movement.json"
OUT_MOV_MD = ROOT / "results" / "metrics" / "gold_rank_movement.md"
OUT_RECALL_JSON = ROOT / "results" / "metrics" / "retrieval_recall_k_diag.json"
OUT_RECALL_MD = ROOT / "results" / "metrics" / "retrieval_recall_k_diag.md"
OUT_PER_JSON = ROOT / "results" / "metrics" / "per_query_regression_recovery.json"
OUT_PER_MD = ROOT / "results" / "metrics" / "per_query_regression_recovery.md"

TOP_KS = [5, 10, 20, 50]


def rank_of(gold_id: str, hits: list[tuple[str, float]]) -> tuple[int | None, float | None]:
    for i, (cid, score) in enumerate(hits, start=1):
        if chunk_ids_match(gold_id, cid):
            return i, score
    return None, None


def hits_match(gold_rank_value: int | None, k: int) -> bool:
    return gold_rank_value is not None and gold_rank_value <= k


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    per_query = []
    rank_movement_rows = []
    recall_rows = {stage: {k: [] for k in TOP_KS} for stage in ["dense", "bm25", "rrf", "reranked"]}
    recall_rows["dense"]["mrr10"] = []
    recall_rows["bm25"]["mrr10"] = []
    recall_rows["rrf"]["mrr10"] = []
    recall_rows["reranked"]["mrr10"] = []

    per_query_rows = []

    for item in dataset:
        qid = item["query_id"]
        gold = item["gold_chunk_id"]
        question = item["question"]

        # gunakan representasi sesungguhnya seperti yang diproses _run_search():
        dense_hits = retriever.dense_search(question, k=40)
        bm25_hits = retriever.bm25_search(question, k=40)

        # candidate list seperti di _run_search/fused (pakai _rrf_fuse real)
        fused_results = retriever._rrf_fuse([dense_hits], [bm25_hits], candidate_k=40, dense_weight=0.5, bm25_weight=0.5)
        fused_filtered = [
            c for c in fused_results
            if (c.get("retrieval_score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)) >= 0.0
        ]

        dense_pairs = [
            (
                h.chunk_id,
                float(h.score),
            )
            for h in dense_hits
        ]
        bm25_pairs = [
            (
                h["chunk_id"] if isinstance(h, dict) else getattr(h, "chunk_id", ""),
                float(h["score"] if isinstance(h, dict) else getattr(h, "score", 0.0) or 0.0),
            )
            for h in bm25_hits
        ]

        dense_rank, dense_score = rank_of(gold, dense_pairs)
        bm25_rank, bm25_score = rank_of(gold, bm25_pairs)

        # candidatorder mengikuti logic _run_search (score sort) untuk rerank candidates
        candidates = [c for c in fused_filtered if not _cf_is_low_quality(c)]
        rerank_candidates = sorted(
            candidates,
            key=lambda x: x.get("retrieval_score", 0.0) if isinstance(x, dict) else getattr(x, "score", 0.0),
            reverse=True,
        )[:20]

        rrf_pairs_for_stage = []
        for h in fused_filtered:
            score = h.get("retrieval_score") if isinstance(h, dict) else getattr(h, "score", 0.0)
            cid = h.get("chunk_id") if isinstance(h, dict) else getattr(h, "chunk_id", "")
            rrf_pairs_for_stage.append((cid, float(score) if score is not None else 0.0))
        rrf_rank, rrf_score = rank_of(gold, rrf_pairs_for_stage)

        reranked = retriever.rerank(question, rerank_candidates)
        reranked_hits = []
        for hit in reranked:
            score = hit.get("rerank_score") if isinstance(hit, dict) else getattr(hit, "rerank_score", 0.0)
            cid = hit.get("chunk_id") if isinstance(hit, dict) else getattr(hit, "chunk_id", "")
            reranked_hits.append((cid, float(score) if score is not None else 0.0))
        rerank_rank, rerank_score = rank_of(gold, reranked_hits)

        # Recall@K accumulation
        for stage, rankval in [("dense", dense_rank), ("bm25", bm25_rank), ("rrf", rrf_rank), ("reranked", rerank_rank)]:
            for k in TOP_KS:
                recall_rows[stage][k].append(1 if hits_match(rankval, k) else 0)
            mrr = 0.0 if rankval is None else 1.0 / rankval
            recall_rows[stage]["mrr10"].append(mrr if rankval is not None and rankval <= 10 else 0.0)

        # Determine failure stage / classification
        in_rrf = rrf_rank is not None
        in_reranked = rerank_rank is not None
        final_ok = in_reranked and rerank_rank <= 10

        if not in_rrf and not in_reranked:
            classification = "RETRIEVAL_MISS"
        elif in_rrf and not in_reranked:
            classification = "RERANKER_DEMOTION"
        elif in_reranked and not final_ok:
            classification = "TOP_K_CUTOFF"
        elif dense_rank is not None and dense_rank > 10 and rrf_rank is not None and rrf_rank <= 10:
            classification = "RERANKER_RECOVERY"
        elif bm25_rank is not None and rrf_rank is not None and rrf_rank > bm25_rank + 3:
            classification = "FUSION_REGRESSION"
        elif rerank_rank is not None and rrf_rank is not None and rerank_rank > rrf_rank + 3:
            classification = "RERANKER_REGRESSION"
        elif rerank_rank is not None and rrf_rank is not None and rerank_rank <= 10:
            classification = "NO_FAILURE"
        else:
            classification = "NO_FAILURE"

        if in_reranked and rerank_rank <= 10 and rrf_rank is not None and rrf_rank > rerank_rank:
            classification = "RERANKER_RECOVERY"

        per_query_rows.append(
            {
                "query_id": qid,
                "gold_chunk_id": gold,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "rrf_rank": rrf_rank,
                "reranker_rank": rerank_rank,
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "rrf_score": rrf_score,
                "reranker_score": rerank_score,
                "final_status": "in" if final_ok else "out",
                "rank_delta_rrf": (rrf_rank - bm25_rank) if (rrf_rank is not None and bm25_rank is not None) else None,
                "rank_delta_reranker": (rerank_rank - rrf_rank) if (rerank_rank is not None and rrf_rank is not None) else None,
                "failure_stage": classification,
            }
        )

        rank_movement_rows.append(
            {
                "query_id": qid,
                "gold_chunk_id": gold,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "rrf_rank": rrf_rank,
                "reranker_rank": rerank_rank,
                "final_status": "in" if final_ok else "out",
                "rank_delta_rrf": (rrf_rank - bm25_rank) if (rrf_rank is not None and bm25_rank is not None) else None,
                "rank_delta_reranker": (rerank_rank - rrf_rank) if (rerank_rank is not None and rrf_rank is not None) else None,
                "failure_stage": classification,
            }
        )

        per_query.append(
            {
                "query_id": qid,
                "question": question,
                "gold_chunk_id": gold,
                "dense_rank": dense_rank,
                "dense_score": dense_score,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_score,
                "rrf_rank": rrf_rank,
                "rrf_score": rrf_score,
                "reranker_rank": rerank_rank,
                "reranker_score": rerank_score,
                "final_included": final_ok,
                "classification": classification,
                "rank_movement": {
                    "dense_to_rrf": (rrf_rank - dense_rank) if (rrf_rank is not None and dense_rank is not None) else None,
                    "rrf_to_reranked": (rerank_rank - rrf_rank) if (rerank_rank is not None and rrf_rank is not None) else None,
                },
                "top10_reranked": [{"rank": i + 1, "chunk_id": cid, "score": score} for i, (cid, score) in enumerate(reranked_hits[:10])],
            }
        )

    store.close()

    # q15-specific
    q15_row = next((r for r in per_query if r["query_id"] == "q15"), None)
    OUT_Q15_JSON.write_text(json.dumps(q15_row, indent=2, ensure_ascii=False))
    OUT_Q15_MD.write_text("# q15 Rank Forensic\n\n```json\n" + json.dumps(q15_row, indent=2, ensure_ascii=False) + "\n```\n")

    OUT_MOV_JSON.write_text(json.dumps(rank_movement_rows, indent=2, ensure_ascii=False))
    mov_lines = ["# Gold Rank Movement", "", "| query_id | gold_chunk_id | dense_rank | bm25_rank | rrf_rank | reranker_rank | final_status | Δrrf(vs bm25) | Δreranker(vs rrf) | failure_stage |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rank_movement_rows:
        mov_lines.append(f"| {row['query_id']} | {row['gold_chunk_id']} | {row['dense_rank']} | {row['bm25_rank']} | {row['rrf_rank']} | {row['reranker_rank']} | {row['final_status']} | {row['rank_delta_rrf']} | {row['rank_delta_reranker']} | {row['failure_stage']} |")
    OUT_MOV_MD.write_text("\n".join(mov_lines) + "\n")

    recall_summary: dict[str, dict[str, float]] = {}
    for stage in ["dense", "bm25", "rrf", "reranked"]:
        recall_summary[stage] = {}
        for k in TOP_KS:
            vals = recall_rows[stage][k]
            recall_summary[stage][f"recall@{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        mrr = recall_rows[stage]["mrr10"]
        recall_summary[stage]["mrr@10"] = round(sum(mrr) / len(mrr), 4) if mrr else 0.0
    OUT_RECALL_JSON.write_text(json.dumps(recall_summary, indent=2, ensure_ascii=False))
    OUT_RECALL_MD.write_text("# Retrieval Recall@K Diagnostic\n\n" + json.dumps(recall_summary, indent=2, ensure_ascii=False) + "\n")

    OUT_PER_JSON.write_text(json.dumps(per_query_rows, indent=2, ensure_ascii=False))
    per_lines = ["# Per-query Regression / Recovery", "", "| query_id | dense_rank | bm25_rank | rrf_rank | reranker_rank | final_status | failure_stage |", "|---|---|---|---|---|---|---|"]
    for row in per_query_rows:
        per_lines.append(f"| {row['query_id']} | {row['dense_rank']} | {row['bm25_rank']} | {row['rrf_rank']} | {row['reranker_rank']} | {row['final_status']} | {row['failure_stage']} |")
    OUT_PER_MD.write_text("\n".join(per_lines) + "\n")

    print("[saved] all forensic artifacts")


if __name__ == "__main__":
    main()
