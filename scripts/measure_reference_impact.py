from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src._compat  # noqa: F401  must precede qdrant_client on 3.14
from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids

CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v2.json"
EVAL_DIR = REPO_ROOT / "data" / "evaluation"
DATASETS = (
    "retrieval_eval_dataset.json",
    "retrieval_eval_dataset_v2.json",
    "retrieval_eval_dataset_v3.json",
    "retrieval_train_eval.json",
    "retrieval_holdout_eval.json",
)
OUT_PATH = REPO_ROOT / "results" / "metrics" / "k3_reference_impact.json"

STAGE_KEYS = (
    ("dense", "dense_ids"),
    ("bm25", "bm25_ids"),
    ("rrf", "rrf_ids"),
    ("reranked", "reranked_ids"),
    ("final", "final_ids"),
)


def is_reference(chunk: dict) -> bool:
    section = str((chunk.get("metadata") or {}).get("section", "")).lower()
    return "reference" in section or "bibliograph" in section


def corpus_profile(chunks: list[dict]) -> dict:
    ref = [c for c in chunks if is_reference(c)]
    oth = [c for c in chunks if not is_reference(c)]
    ref_len = [len(c["text"]) for c in ref]
    oth_len = [len(c["text"]) for c in oth]
    return {
        "total_chunks": len(chunks),
        "reference_chunks": len(ref),
        "reference_share_pct": round(len(ref) / len(chunks) * 100, 1),
        "reference_char_share_pct": round(sum(ref_len) / max(sum(ref_len + oth_len), 1) * 100, 1),
        "reference_text_chars": {
            "mean": round(statistics.mean(ref_len), 1),
            "median": statistics.median(ref_len),
            "max": max(ref_len),
        },
        "other_text_chars": {
            "mean": round(statistics.mean(oth_len), 1),
            "median": statistics.median(oth_len),
            "max": max(oth_len),
        },
        "length_separable": max(ref_len) < min(oth_len),
        "median_ratio": round(statistics.median(ref_len) / max(statistics.median(oth_len), 1), 3),
    }


def gold_labels(chunks: list[dict]) -> dict:
    ref_ids = {c["chunk_id"] for c in chunks if is_reference(c)}
    out = {}
    for name in DATASETS:
        path = EVAL_DIR / name
        if not path.exists():
            continue
        records = dataset_records(json.loads(path.read_text(encoding="utf-8")))
        offenders = []
        for index, record in enumerate(records):
            try:
                ids = normalize_gold_chunk_ids(record)
            except ValueError as exc:
                offenders.append({"row": index, "error": str(exc)})
                continue
            for gold in ids:
                if gold in ref_ids:
                    offenders.append(
                        {
                            "query_id": record.get("query_id", f"index {index}"),
                            "gold_chunk_id": gold,
                        }
                    )
        out[name] = {
            "queries": len(records),
            "gold_pointing_at_references": len(offenders),
            "details": offenders,
        }
    return out


def stage_leakage(dataset_path: Path, top_k: int = 10) -> dict:
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.retrieval.qdrant_store import QdrantVectorStore

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    ref_ids = {c["chunk_id"] for c in chunks if is_reference(c)}
    records = dataset_records(json.loads(dataset_path.read_text(encoding="utf-8")))

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    totals = {name: [0, 0] for name, _ in STAGE_KEYS}
    rerank_total = rerank_refs = 0
    blocked_reference = 0
    per_query = []

    try:
        for item in records:
            out = retriever.search_with_trace(item["question"], top_k=top_k, rerank=True)
            trace = out["trace"]
            row = {"query_id": item.get("query_id"), "stages": {}}
            for name, key in STAGE_KEYS:
                ids = trace.get(key, [])
                refs = sum(1 for i in ids if i in ref_ids)
                totals[name][0] += refs
                totals[name][1] += len(ids)
                row["stages"][name] = {"reference": refs, "total": len(ids)}
            rerank_total += trace["rerank_input_count"]
            rerank_refs += sum(1 for i in trace["rerank_input_ids"] if i in ref_ids)
            blocked_reference += trace["filter_breakdown"]["reference"]
            row["rerank_input"] = {
                "reference": sum(1 for i in trace["rerank_input_ids"] if i in ref_ids),
                "total": trace["rerank_input_count"],
            }
            row["blocked_by_strict_filter"] = trace["filter_breakdown"]["reference"]
            row["filter_stage"] = trace["filter_stage"]
            per_query.append(row)
    finally:
        store.close()

    return {
        "dataset": str(dataset_path),
        "queries": len(records),
        "stages": {
            name: {
                "reference": a,
                "total": b,
                "share_pct": round(a / b * 100, 1) if b else 0.0,
            }
            for name, (a, b) in totals.items()
        },
        "rerank_input": {
            "reference": rerank_refs,
            "total": rerank_total,
            "share_pct": round(rerank_refs / rerank_total * 100, 1) if rerank_total else 0.0,
        },
        "blocked_by_strict_filter_total": blocked_reference,
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(EVAL_DIR / "retrieval_eval_dataset_v2.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--skip-pipeline", action="store_true", help="corpus and gold-label checks only, no Qdrant"
    )
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    profile = corpus_profile(chunks)
    print("[corpus]")
    print(
        f"  {profile['reference_chunks']}/{profile['total_chunks']} chunks are REFERENCES "
        f"({profile['reference_share_pct']}%), {profile['reference_char_share_pct']}% of characters"
    )
    print(
        f"  median text length: references {profile['reference_text_chars']['median']} vs "
        f"other {profile['other_text_chars']['median']} chars "
        f"(length_separable={profile['length_separable']})"
    )

    labels = gold_labels(chunks)
    print("[gold labels]")
    total_bad = 0
    for name, info in labels.items():
        total_bad += info["gold_pointing_at_references"]
        print(
            f"  {name:36s} {info['queries']:2d} queries, "
            f"{info['gold_pointing_at_references']} gold -> REFERENCES"
        )

    payload = {
        "corpus": profile,
        "gold_labels": labels,
        "gold_pointing_at_references_total": total_bad,
    }

    if not args.skip_pipeline:
        print("[pipeline]")
        leak = stage_leakage(Path(args.dataset), top_k=args.top_k)
        for name, info in leak["stages"].items():
            print(
                f"  {name:10s} {info['reference']:4d}/{info['total']:4d} references survive "
                f"({info['share_pct']}%)"
            )
        r = leak["rerank_input"]
        print(f"  reranker input: {r['reference']}/{r['total']} references ({r['share_pct']}%)")
        print(f"  blocked by strict filter: {leak['blocked_by_strict_filter_total']} occurrences")
        payload["pipeline"] = leak

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
