from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.qdrant_store import QdrantVectorStore

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
CHUNKS = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT = ROOT / "results" / "metrics" / "reranker_latency_profile.json"


def estimated_tokens(tokenizer, text: str) -> int:
    try:
        toks = tokenizer.encode(text, add_special_tokens=False)
        return len(toks)
    except Exception:
        return max(1, len(text.split()))


def main() -> None:
    chunks = json.loads(CHUNKS.read_text())
    by_id = {c["chunk_id"]: c for c in chunks}
    from transformers import AutoTokenizer

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tok = AutoTokenizer.from_pretrained(model_name)

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    rows = []
    for item in json.loads(DATASET.read_text()):
        out = retriever.search_with_trace(item["question"], top_k=10, rerank=True)
        tr = out["trace"]
        input_ids = tr.get("rerank_input_ids", [])
        texts = []
        for cid in input_ids:
            hit = by_id.get(cid)
            if hit:
                texts.append(hit.get("text", ""))
        token_counts = [estimated_tokens(tok, t) for t in texts]
        rerank_ms = tr.get("latency_ms", {}).get("reranker")
        row = {
            "query_id": item["query_id"],
            "gold_chunk_id": item["gold_chunk_id"],
            "rerank_input_count": len(input_ids),
            "rerank_executed": tr.get("reranker_executed"),
            "rerank_model_name": model_name,
            "candidate_tokens_estimated": sum(token_counts),
            "max_candidate_text_tokens": max(token_counts) if token_counts else 0,
            "total_candidate_text_tokens": sum(token_counts),
            "rerank_ms": rerank_ms,
            "total_ms": tr.get("latency_ms", {}).get("total"),
        }
        rows.append(row)
    store.close()

    import statistics

    counts = [r["rerank_input_count"] for r in rows]
    tokens = [r["total_candidate_text_tokens"] for r in rows]
    rerank_times = [r["rerank_ms"] for r in rows if r["rerank_ms"] is not None]
    total_times = [r["total_ms"] for r in rows if r["total_ms"] is not None]

    def pct(arr, q):
        if not arr:
            return None
        s = sorted(arr)
        idx = int(q / 100 * len(s))
        if idx >= len(s):
            idx = len(s) - 1
        return s[idx]

    stats = {
        "n_queries": len(rows),
        "avg_candidate_count": round(sum(counts) / len(counts), 4),
        "candidate_count_variance": round(statistics.variance(counts), 4)
        if len(counts) > 1
        else 0.0,
        "avg_total_candidate_text_tokens": round(sum(tokens) / len(tokens), 2),
        "median_total_candidate_text_tokens": statistics.median(tokens),
        "max_candidate_text_length": max(tokens) if tokens else 0,
        "p50_reranker_ms": pct(rerank_times, 50),
        "p95_reranker_ms": pct(rerank_times, 95),
        "min_reranker_ms": min(rerank_times) if rerank_times else None,
        "max_reranker_ms": max(rerank_times) if rerank_times else None,
        "p50_total_ms": pct(total_times, 50),
        "p95_total_ms": pct(total_times, 95),
    }

    try:
        import numpy as np

        corr_count = None
        corr_tokens = None
        if len(rows) > 1 and len(set(counts)) > 1 and len(set(rerank_times)) > 1:
            corr_count = float(np.corrcoef(counts, rerank_times)[0, 1])
        if len(rows) > 1 and len(set(tokens)) > 1 and len(set(rerank_times)) > 1:
            corr_tokens = float(np.corrcoef(tokens, rerank_times)[0, 1])
        stats["corr_reranker_ms_vs_candidate_count"] = corr_count
        stats["corr_reranker_ms_vs_candidate_tokens"] = corr_tokens
    except Exception:
        stats["corr_reranker_ms_vs_candidate_count"] = None
        stats["corr_reranker_ms_vs_candidate_tokens"] = None

    artifact = {
        "rows": rows,
        "summary": stats,
    }
    OUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
