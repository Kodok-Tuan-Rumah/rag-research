from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src._compat  # noqa: F401
from src.evaluation.gold_matching import chunk_ids_match
from src.retrieval.chunk_filters import is_low_quality_chunk
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.qdrant_store import QdrantVectorStore

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
OUT = ROOT / "results" / "metrics" / "rrf_to_candidate_slot_drift.json"
OUT_MD = ROOT / "results" / "metrics" / "rrf_to_candidate_slot_drift.md"


def _is_lowq_text(text: str) -> bool:
    return is_low_quality_chunk({"text": text, "metadata": {}})


def _ranks(rows: list[dict], gold_id: str) -> int | None:
    for i, r in enumerate(rows, 1):
        if chunk_ids_match(gold_id, r.get("chunk_id", "")):
            return i
    return None


def main() -> None:
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    rows = []
    for item in json.loads(DATASET.read_text()):
        out = retriever.search_with_trace(item["question"], top_k=10, rerank=True)
        tr = out["trace"]
        rrf_rows = tr.get("rrf_rows", [])
        rerank_input_ids = tr.get("rerank_input_ids", [])
        reranked_ids = tr.get("reranked_ids", [])
        final_ids = tr.get("final_ids", [])

        gold_id = item["gold_chunk_id"]
        rrf_rank_pre_filter = _ranks(rrf_rows, gold_id)
        # qualified rank: position in filtered candidates (low-quality dropped) before truncation to 20
        qualified_items = []
        for i, row in enumerate(rrf_rows, 1):
            text = ""
            raw_item = None
            qualified_items.append(row)
        # more realistic: reproduce candidate list as rff filtered low-quality and sorted retrieval_score
        candidates = []
        for row in rrf_rows:
            if not _is_lowq_text(row.get("text", "") or ""):
                candidates.append(row)
        qualified_rank = _ranks(candidates, gold_id)
        removed_before_gold = 0
        if rrf_rank_pre_filter is not None:
            gold_or_before = rrf_rows[: rrf_rank_pre_filter - 1]
            removed_before_gold = sum(
                1 for r in gold_or_before if _is_lowq_text(r.get("text", "") or "")
            )
        rerank_input_index = next(
            (i + 1 for i, cid in enumerate(rerank_input_ids) if chunk_ids_match(gold_id, cid)), None
        )
        rerank_input_rank = rerank_input_index
        final_rerank_rank = next(
            (i + 1 for i, cid in enumerate(reranked_ids) if chunk_ids_match(gold_id, cid)), None
        )

        classification = "UNKNOWN"
        if rrf_rank_pre_filter is None:
            classification = "BLANK"
        elif rrf_rank_pre_filter > 40:
            classification = "RRF_BEYOND_RERANK_DEPTH"
        elif qualified_rank is None and rerank_input_index is None:
            classification = "FILTERED_OUT"
        elif rerank_input_index is not None:
            classification = "CANDIDATE_SURVIVED"

        rows.append(
            {
                "query_id": item["query_id"],
                "gold_chunk_id": gold_id,
                "gold_rrf_rank": rrf_rank_pre_filter,
                "rrf_rank_pre_filter": rrf_rank_pre_filter,
                "qualified_rank": qualified_rank,
                "removed_before_gold": removed_before_gold,
                "rerank_input_index": rerank_input_index,
                "rerank_input_rank": rerank_input_rank,
                "rrf_score": (
                    next(
                        (
                            r.get("score")
                            for r in rrf_rows
                            if chunk_ids_match(gold_id, r.get("chunk_id", ""))
                        ),
                        None,
                    )
                ),
                "reranker_score": (
                    next(
                        (
                            r.get("score")
                            for r in tr.get("reranked_rows", [])
                            if chunk_ids_match(gold_id, r.get("chunk_id", ""))
                        ),
                        None,
                    )
                ),
                "final_rerank_rank": final_rerank_rank,
                "gold_in_rrf_top40": (
                    rrf_rank_pre_filter is not None and rrf_rank_pre_filter <= 40
                ),
                "gold_survives_qualification": qualified_rank is not None,
                "candidate_list_size_before_filter": len(rrf_rows),
                "candidate_list_size_after_filter": len(candidates),
                "classification": classification,
            }
        )

    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    lines = [
        "# RRF -> Candidate Slot Drift",
        "",
        "| query_id | gold_rrf_rank | qualified_rank | removed_before_gold | rerank_input_index | final_rerank_rank | classification |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['query_id']} | {r['gold_rrf_rank']} | {r['qualified_rank']} | {r['removed_before_gold']} | {r['rerank_input_index']} | {r['final_rerank_rank']} | {r['classification']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    store.close()
    print(f"[saved] {OUT} and {OUT_MD}")


if __name__ == "__main__":
    main()
