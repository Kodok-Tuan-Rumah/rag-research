from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src._compat  # noqa: F401
from src.evaluation.gold_matching import chunk_ids_match
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.qdrant_store import QdrantVectorStore

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
OUT_JSON = ROOT / "results" / "metrics" / "fase_f_reranker_ablation.json"


def rr_at_10(rank):
    if rank is None or rank < 1 or rank > 10:
        return 0.0
    return 1.0 / rank


def run_config(store, retriever, dataset, use_reranker):
    rows = []
    latencies = []
    for item in dataset:
        t0 = time.perf_counter()
        result = retriever.search(item["question"], top_k=10, rerank=use_reranker)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)

        gold_rank_val = None
        for i, chunk in enumerate(result, 1):
            if chunk_ids_match(item["gold_chunk_id"], chunk.get("chunk_id", "")):
                gold_rank_val = i
                break

        rows.append(
            {
                "query_id": item["query_id"],
                "gold_chunk_id": item["gold_chunk_id"],
                "rank": gold_rank_val,
                "rr_at_10": rr_at_10(gold_rank_val),
                "in_top10": gold_rank_val is not None and gold_rank_val <= 10,
                "latency_ms": round(elapsed, 2),
            }
        )

    n = len(rows)
    recall_10 = sum(1 for r in rows if r["in_top10"]) / n
    mrr_10 = sum(r["rr_at_10"] for r in rows) / n

    return {
        "config": "A_no_reranker" if not use_reranker else "B_with_reranker",
        "n_queries": n,
        "recall_at_10": round(recall_10, 4),
        "mrr_at_10": round(mrr_10, 4),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2)
        if latencies
        else None,
        "rows": rows,
    }


def main():
    dataset = json.loads(DATASET.read_text())
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    print("Running Configuration A (reranker OFF)...")
    config_a = run_config(store, retriever, dataset, use_reranker=False)

    print("Running Configuration B (reranker ON)...")
    config_b = run_config(store, retriever, dataset, use_reranker=True)

    store.close()

    per_query = []
    for a, b in zip(config_a["rows"], config_b["rows"]):
        a_rank = a["rank"]
        b_rank = b["rank"]
        a_rr = a["rr_at_10"]
        b_rr = b["rr_at_10"]
        delta = round(b_rr - a_rr, 4)

        if a_rank is not None and a_rank > 10 and b_rank is not None and b_rank <= 10:
            cls = "IMPROVED"
        elif a_rank is not None and a_rank <= 10 and b_rank is not None and b_rank > 10:
            cls = "HARMED"
        elif (
            a_rank is not None
            and a_rank <= 10
            and b_rank is not None
            and b_rank <= 10
            and b_rank > a_rank
        ):
            cls = "DEMOTED_BUT_TOP10"
        elif (
            a_rank is not None
            and a_rank <= 10
            and b_rank is not None
            and b_rank <= 10
            and b_rank < a_rank
        ):
            cls = "IMPROVED_WITHIN_TOP10"
        elif a_rank == b_rank:
            cls = "UNCHANGED"
        elif a_rank is not None and a_rank > 10 and b_rank is not None and b_rank > 10:
            cls = "A_OUTSIDE_TOP10_AND_B_OUTSIDE"
        elif b_rank is None:
            cls = "BEYOND_RERANK_DEPTH"
        else:
            cls = "UNCHANGED"

        per_query.append(
            {
                "query_id": a["query_id"],
                "A_rank": a_rank,
                "B_rank": b_rank,
                "A_RR@10": a_rr,
                "B_RR@10": b_rr,
                "delta_RR@10": delta,
                "classification": cls,
            }
        )

    count_improved = sum(1 for r in per_query if r["classification"] == "IMPROVED")
    count_harmed = sum(1 for r in per_query if r["classification"] == "HARMED")
    count_demoted = sum(1 for r in per_query if r["classification"] == "DEMOTED_BUT_TOP10")
    count_unchanged = sum(
        1
        for r in per_query
        if r["classification"]
        in (
            "UNCHANGED",
            "IMPROVED_WITHIN_TOP10",
            "A_OUTSIDE_TOP10_AND_B_OUTSIDE",
            "BEYOND_RERANK_DEPTH",
        )
    )

    net_improvement = count_improved - count_harmed
    reranker_value_ratio = round(net_improvement / len(dataset), 4)

    result = {
        "config_A": {k: v for k, v in config_a.items() if k != "rows"},
        "config_B": {k: v for k, v in config_b.items() if k != "rows"},
        "per_query": per_query,
        "value_assessment": {
            "count_improved": count_improved,
            "count_harmed": count_harmed,
            "count_demoted_but_top10": count_demoted,
            "count_unchanged": count_unchanged,
            "net_improvement": net_improvement,
            "reranker_value_ratio": reranker_value_ratio,
            "recall_at_10_delta": round(config_b["recall_at_10"] - config_a["recall_at_10"], 4),
            "mrr_at_10_delta": round(config_b["mrr_at_10"] - config_a["mrr_at_10"], 4),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT_JSON}")
    print(
        f"A: Recall@10={config_a['recall_at_10']}, MRR@10={config_a['mrr_at_10']}, p50={config_a['p50_latency_ms']}ms, p95={config_a['p95_latency_ms']}ms"
    )
    print(
        f"B: Recall@10={config_b['recall_at_10']}, MRR@10={config_b['mrr_at_10']}, p50={config_b['p50_latency_ms']}ms, p95={config_b['p95_latency_ms']}ms"
    )
    print(
        f"Value: improved={count_improved}, harmed={count_harmed}, demoted={count_demoted}, unchanged={count_unchanged}"
    )
    print(f"Net improvement={net_improvement}, ratio={reranker_value_ratio}")


if __name__ == "__main__":
    main()
