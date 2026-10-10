"""Build RRF contribution ledger per gold query."""

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
OUT = ROOT / "results" / "metrics" / "rrf_contribution_ledger.json"


def main() -> None:
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    rows = []
    for item in json.loads(DATASET.read_text()):
        out = retriever.search_with_trace(item["question"], top_k=10, rerank=True)
        trace = out["trace"]
        rrf_rows = trace.get("rrf_rows", [])
        gold_row = None
        gold_pos = None
        for i, r in enumerate(rrf_rows, start=1):
            if chunk_ids_match(item["gold_chunk_id"], r.get("chunk_id", "")):
                gold_row = r; gold_pos = i; break

        row = {
            "query_id": item["query_id"],
            "gold_chunk_id": item["gold_chunk_id"],
            "gold_doc_id": item.get("gold_doc_id"),
            "dense_rank": None,
            "bm25_rank": None,
            "dense_contribution": None if gold_row is None else gold_row.get("dense_contrib"),
            "bm25_contribution": None if gold_row is None else gold_row.get("bm25_contrib"),
            "dense_weight": None,
            "bm25_weight": None,
            "section": None if gold_row is None else (gold_row.get("metadata") or {}).get("section"),
            "section_boost": None,
            "final_rrf_score": None if gold_row is None else gold_row.get("score"),
            "tie_group": None,
            "insertion_order": None,
            "rrf_rank": gold_pos,
            "rerank_input_count": trace.get("rerank_input_count"),
            "gold_in_rerank_input": any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("rerank_input_ids", [])),
            "reranker_executed": trace.get("reranker_executed"),
            "reranker_skip_reason": trace.get("reranker_skip_reason"),
            "dense_hit": None,
            "bm25_hit": None,
            "rrf_hit": gold_pos is not None,
            "rerank_input_hit": any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("rerank_input_ids", [])),
            "rerank_output_hit": any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("reranked_ids", [])),
            "final_filter_hit": any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("filtered_ids", [])),
            "final_hit": any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("final_ids", [])),
            "parent_group_id": item["gold_chunk_id"].split("_rs_")[0] if "_rs_" in item["gold_chunk_id"] else None,
            "sibling_count_in_rrf_top40": 0,
            "sibling_count_in_rerank_input": 0,
            "gold_family_rank": gold_pos,
            "gold_exact_rank": gold_pos,
        }

        # derived ranks/dense/bm25 present
        dense_hit = any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("dense_ids", []))
        bm25_hit = any(chunk_ids_match(item["gold_chunk_id"], cid) for cid in trace.get("bm25_ids", []))
        rrf_hit = gold_pos is not None
        row["dense_hit"] = dense_hit
        row["bm25_hit"] = bm25_hit
        row["rrf_hit"] = rrf_hit
        row["dense_rank"] = next((i + 1 for i, cid in enumerate(trace.get("dense_ids", [])) if chunk_ids_match(item["gold_chunk_id"], cid)), None)
        row["bm25_rank"] = next((i + 1 for i, cid in enumerate(trace.get("bm25_ids", [])) if chunk_ids_match(item["gold_chunk_id"], cid)), None)

        if gold_row is not None:
            row["dense_weight"] = None
            row["bm25_weight"] = None
            row["section"] = None
            row["section_boost"] = None
            row["final_rrf_score"] = gold_row.get("score")
            row["tie_group"] = None
            row["insertion_order"] = gold_pos

        rows.append(row)

    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    store.close()
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
