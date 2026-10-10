"""Retrieval-only stage benchmark — no LLM, no RAGAS.

Menjalankan satu jalur canonical (`MasterHybridRetriever.search_with_trace`)
dengan candidate_k default dan mengukur tiap stage secara terpisah:

A. dense_only
B. bm25_only
C. rrf_fused            (dense + BM25 dengan RRF canonical)
D. rrf_reranked         (cross-encoder rerank)
E. final_filtered       (strict → minimum filter fallback)

Output: results/metrics/retrieval_stage_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.retrieval_evaluator import aggregate_rows, evaluate_query
from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever

DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v2.json"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUTPUT_PATH = REPO_ROOT / "results" / "metrics" / "retrieval_stage_eval.json"

STAGE_TRACE_KEYS = {
    "dense_only": "dense_ids",
    "bm25_only": "bm25_ids",
    "rrf_fused": "rrf_ids",
    "rrf_reranked": "reranked_ids",
    "final_filtered": "final_ids",
}
STAGE_LATENCY_KEYS = {
    "dense_only": "dense",
    "bm25_only": "bm25",
    "rrf_fused": "fusion",
    "rrf_reranked": "reranker",
    "final_filtered": "filtering",
}


def run(dataset_path: Path, chunks_path: Path, out_path: Path, top_k: int = 10) -> dict:
    import src._compat  # noqa: F401
    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    dataset = dataset_records(json.loads(dataset_path.read_text(encoding="utf-8")))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_doc = {c["chunk_id"]: c.get("doc_id") for c in chunks}

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    per_stage_rows = {stage: [] for stage in STAGE_TRACE_KEYS}
    per_query = []

    for item in dataset:
        t0 = time.perf_counter()
        out = retriever.search_with_trace(item["question"], top_k=top_k, rerank=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        trace = out["trace"]
        for stage, key in STAGE_TRACE_KEYS.items():
            ids = trace.get(key, [])
            docs = [chunk_doc.get(cid, "") for cid in ids]
            row = evaluate_query(
                query_id=item["query_id"],
                question=item["question"],
                gold_chunk_ids=normalize_gold_chunk_ids(item),
                gold_doc_id=item.get("gold_doc_id"),
                retrieved_chunk_ids=ids,
                retrieved_doc_ids=docs,
                stage=stage,
                latency_ms=trace.get("latency_ms", {}).get(STAGE_LATENCY_KEYS[stage], elapsed),
            )
            per_stage_rows[stage].append(row)
        per_query.append({
            "query_id": item["query_id"],
            "question": item["question"],
            "gold_chunk_ids": normalize_gold_chunk_ids(item),
            "gold_doc_id": item.get("gold_doc_id"),
            "stages": {
                stage: {
                    "retrieved_chunk_ids": row.retrieved_chunk_ids,
                    "retrieved_doc_ids": row.retrieved_doc_ids,
                    "gold_rank": row.gold_rank,
                    "reciprocal_rank": row.reciprocal_rank,
                    "hit_at_1": row.hit_at_1,
                    "hit_at_5": row.hit_at_5,
                    "hit_at_10": row.hit_at_10,
                    "latency_ms": row.latency_ms,
                }
                for stage, rows in per_stage_rows.items()
                for row in rows[-1:]
                if row.query_id == item["query_id"]
            },
            "trace_counts": trace.get("counts"),
            "ranking_mode": trace.get("ranking_mode"),
            "filter_stage": trace.get("filter_stage"),
            "trace_latency_ms": trace.get("latency_ms"),
        })

    summary = {}
    for stage, rows in per_stage_rows.items():
        agg = aggregate_rows(rows)
        agg["stage"] = stage
        summary[stage] = agg
        print(
            f"{stage:15s} recall@5={agg.get('recall@5')} recall@10={agg.get('recall@10')} "
            f"mrr@10={agg.get('mrr@10')} avg_cand={agg.get('avg_candidates')} "
            f"avg_gold_rank_hit={agg.get('avg_gold_rank_when_hit')} failures={agg.get('failure_counts')}"
        )

    payload = {
        "dataset": str(dataset_path),
        "corpus": str(chunks_path),
        "top_k": top_k,
        "stages": summary,
        "queries": per_query,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {out_path}")
    store.close()
    return payload


RESULTS_ADJ = REPO_ROOT / "results" / "metrics" / "retrieval_stage_eval_adversarial.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--chunks", default=str(CHUNKS_PATH))
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-adversarial", action="store_true")
    args = parser.parse_args()

    payload = run(Path(args.dataset), Path(args.chunks), Path(args.out), top_k=args.top_k)

    if args.include_adversarial:
        ADV_PATH = REPO_ROOT / "data" / "adversarial" / "adversarial_queries.json"
        try:
            adv_rows = json.loads(ADV_PATH.read_text())
        except Exception:
            adv_rows = []

        store_adv = QdrantVectorStore()
        retriever_adv = MasterHybridRetriever(vector_store=store_adv)
        abstention_expected = [r for r in adv_rows if r.get("should_abstain") is True]
        no_abstention_expected = [r for r in adv_rows if r.get("should_abstain") is False]

        abstention_correct = 0
        no_abstention_insufficient = 0
        adv_rows_out = []
        for row in adv_rows:
            q = row.get("query", "")
            try:
                out = retriever_adv.search_with_trace(q, top_k=args.top_k, rerank=True)
                tr = out.get("trace", {})
                actual = "ABSTAIN" if tr.get("should_abstain") is True or tr.get("counts", {}).get("final", 0) == 0 else "CANDIDATE_AVAILABLE"
                if row.get("should_abstain") is True:
                    if actual == "ABSTAIN":
                        abstention_correct += 1
                else:
                    if actual == "ABSTAIN":
                        no_abstention_insufficient += 1
                adv_rows_out.append({
                    "query_id": row.get("query_id"),
                    "query": q,
                    "type": row.get("type"),
                    "expected_abstain": row.get("should_abstain"),
                    "actual": actual,
                    "should_abstain_flag": tr.get("should_abstain"),
                    "final_count": tr.get("counts", {}).get("final"),
                })
            except Exception as exc:
                adv_rows_out.append({
                    "query_id": row.get("query_id"),
                    "query": q,
                    "type": row.get("type"),
                    "expected_abstain": row.get("should_abstain"),
                    "actual": "ERROR",
                    "error": str(type(exc)),
                })

        store_adv.close()

        abstention_expected_count = len(abstention_expected)
        no_abstention_expected_count = len(no_abstention_expected)
        adv = {
            "total": len(adv_rows),
            "abstention_expected": abstention_expected_count,
            "abstention_correct": abstention_correct,
            "abstention_correct_rate": round(abstention_correct / abstention_expected_count, 4) if abstention_expected_count else 0.0,
            "no_abstention_expected": no_abstention_expected_count,
            "no_abstention_was_insufficient": no_abstention_insufficient,
            "rows": adv_rows_out,
        }

        payload["adversarial_eval"] = adv
        Path(RESULTS_ADJ).parent.mkdir(parents=True, exist_ok=True)
        Path(RESULTS_ADJ).write_text(json.dumps({**payload, "adversarial_eval": adv}, indent=2, ensure_ascii=False))
        print(f"[Saved] adversarial summary -> {RESULTS_ADJ}")


if __name__ == "__main__":
    main()
