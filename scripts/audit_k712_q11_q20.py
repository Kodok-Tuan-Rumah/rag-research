"""Forensically compare RRF and reranker@5 for canonical queries q11 and q20."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data/evaluation/retrieval_eval_dataset_v2.json"
CHUNKS_PATH = ROOT / "data/processed/chunks_semantic_v2.json"
DOCUMENTS_PATH = ROOT / "data/processed/documents.json"
K79_PATH = ROOT / "results/metrics/rrf_vs_reranker5_decision_20261005T141527Z.json"
TARGET_QUERY_IDS = ("q11", "q20")
MAX_LENGTH = 326
RERANKER_CAP = 5
TOP_K = 10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def rank_of(ids: list[str], gold_ids: list[str]) -> int | None:
    gold = set(gold_ids)
    return next((index for index, chunk_id in enumerate(ids, 1) if chunk_id in gold), None)


def score_map(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        str(row["chunk_id"]): float(row["score"]) if row.get("score") is not None else None
        for row in rows
        if row.get("chunk_id") is not None
    }


def text_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["chunk_id"]): str(row.get("text") or "")
        for row in rows
        if row.get("chunk_id") is not None
    }


def build_report() -> dict[str, Any]:
    if os.getenv("QDRANT_URL"):
        raise RuntimeError("QDRANT_URL is set; refusing to run forensic retrieval against remote Qdrant")

    envelope = read_json(DATASET_PATH)
    records = envelope["records"]
    dataset_by_id = {row["query_id"]: row for row in records}
    if len(dataset_by_id) != len(records):
        raise ValueError("Dataset contains duplicate query_id values")
    missing_queries = [query_id for query_id in TARGET_QUERY_IDS if query_id not in dataset_by_id]
    if missing_queries:
        raise ValueError(f"Target queries missing from canonical dataset: {missing_queries}")

    reference = read_json(K79_PATH)
    hashes = {
        "dataset_file_sha256": sha256(DATASET_PATH),
        "documents_sha256": sha256(DOCUMENTS_PATH),
        "chunks_sha256": sha256(CHUNKS_PATH),
    }
    reference_identity = reference.get("identity", {})
    for key, actual in hashes.items():
        if reference_identity.get(key) != actual:
            raise ValueError(f"Current {key} differs from the K7.9 reference artifact")
    if reference["common_configuration"].get("max_length") != MAX_LENGTH:
        raise ValueError("K7.9 reference max_length is not 326")

    from src.api.config import settings
    from src.retrieval.model_registry import ModelRegistry

    if settings.embed_model != reference["common_configuration"]["embedding_model"]:
        raise ValueError("Configured embedder does not match K7.9")
    if settings.reranker_model != reference["common_configuration"]["reranker_model"]:
        raise ValueError("Configured reranker does not match K7.9")
    if settings.candidate_k != reference["common_configuration"]["candidate_k_setting"]:
        raise ValueError("Configured candidate_k does not match K7.9")

    registry = ModelRegistry.get()
    registry.embedder.to("cpu")
    registry.reranker.model.to("cpu")
    registry.reranker.max_seq_length = MAX_LENGTH

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store, bm25_chunks_path=str(CHUNKS_PATH))
    if retriever.reranker is None:
        store.close()
        raise RuntimeError("The shared model registry did not load a reranker")
    retriever.reranker.max_seq_length = MAX_LENGTH

    original_rerank = retriever._rerank
    captured: dict[str, Any] = {}

    def rerank_first_five(query: str, candidates: list, text_of=None):
        actual_inputs = list(candidates[:RERANKER_CAP])
        captured["input_ids"] = [
            str(chunk.get("chunk_id") if isinstance(chunk, dict) else getattr(chunk, "chunk_id", ""))
            for chunk in actual_inputs
        ]
        ranked, status = original_rerank(query, actual_inputs, text_of=text_of)
        captured["status"] = status
        return ranked, status

    retriever._rerank = rerank_first_five
    try:
        query_reports = []
        reference_rows = {
            row["query_id"]: row
            for row in reference["modes"]["reranker5"]["per_query_results"]
        }
        for query_id in TARGET_QUERY_IDS:
            item = dataset_by_id[query_id]
            expected = reference_rows[query_id]
            captured.clear()
            response = retriever.search_with_trace(
                item["question"],
                top_k=TOP_K,
                dense_weight=0.5,
                bm25_weight=0.5,
                rerank=True,
                use_multi_query=False,
                use_hyde=False,
            )
            trace = response["trace"]
            actual_input_ids = captured.get("input_ids", [])
            rrf_rows = trace.get("rrf_rows", [])
            reranked_rows = trace.get("reranked_rows", [])
            rrf_ids = list(trace.get("rrf_ids") or [])
            reranked_ids = list(trace.get("reranked_ids") or [])
            final_ids = list(trace.get("final_ids") or [])
            gold_ids = sorted(str(value) for value in item["gold_chunk_ids"])

            rrf_scores = score_map(rrf_rows)
            rerank_scores = score_map(reranked_rows)
            rrf_texts = text_map(rrf_rows)
            rerank_texts = text_map(reranked_rows)
            rrf_rank_by_id = {chunk_id: index for index, chunk_id in enumerate(rrf_ids, 1)}
            reranked_rank_by_id = {chunk_id: index for index, chunk_id in enumerate(reranked_ids, 1)}
            final_rank_by_id = {chunk_id: index for index, chunk_id in enumerate(final_ids, 1)}

            candidate_ids = list(dict.fromkeys([*actual_input_ids, *gold_ids]))
            candidates = []
            for chunk_id in candidate_ids:
                in_input = chunk_id in actual_input_ids
                candidates.append(
                    {
                        "chunk_id": chunk_id,
                        "is_gold": chunk_id in gold_ids,
                        "rrf_rank": rrf_rank_by_id.get(chunk_id),
                        "rrf_score": rrf_scores.get(chunk_id),
                        "reranker_input_rank": (
                            actual_input_ids.index(chunk_id) + 1 if in_input else None
                        ),
                        "reranker_score": rerank_scores.get(chunk_id) if in_input else None,
                        "rank_within_reranker_output": reranked_rank_by_id.get(chunk_id),
                        "final_rank": final_rank_by_id.get(chunk_id),
                        "present_in_final_top10": chunk_id in final_rank_by_id,
                        "text_excerpt": (rerank_texts.get(chunk_id) or rrf_texts.get(chunk_id) or "")[:200],
                    }
                )
            ranked_inputs = [
                row for row in candidates if row["reranker_input_rank"] is not None
            ]
            ranked_inputs.sort(key=lambda row: row["rank_within_reranker_output"] or 10**9)
            gold_reranker_rank = next(
                (row["rank_within_reranker_output"] for row in ranked_inputs if row["is_gold"]),
                None,
            )
            displaced_by = (
                [
                    row["chunk_id"]
                    for row in ranked_inputs
                    if row["rank_within_reranker_output"] is not None
                    and gold_reranker_rank is not None
                    and row["rank_within_reranker_output"] < gold_reranker_rank
                ]
                if gold_reranker_rank is not None
                else []
            )

            query_reports.append(
                {
                    "query_id": query_id,
                    "question": item["question"],
                    "gold_chunk_ids": gold_ids,
                    "reranker_status": captured.get("status"),
                    "reranker_max_length": retriever.reranker.max_seq_length,
                    "reranker_cap": RERANKER_CAP,
                    "rrf_order_matches_k79": rrf_ids == expected.get("rrf_ids"),
                    "reranker_input_matches_k79": actual_input_ids
                    == expected.get("effective_reranker_input_ids"),
                    "reranked_order_matches_k79": reranked_ids
                    == expected.get("final_ids"),
                    "rrf_gold_rank": rank_of(rrf_ids, gold_ids),
                    "reranker_gold_rank": rank_of(reranked_ids, gold_ids),
                    "final_gold_rank": rank_of(final_ids, gold_ids),
                    "reference_k79_rrf_gold_rank": expected.get("rrf_rank"),
                    "reference_k79_final_gold_rank": expected.get("final_rank"),
                    "gold_final_hit_at_10": rank_of(final_ids, gold_ids) is not None
                    and rank_of(final_ids, gold_ids) <= TOP_K,
                    "competitors_above_gold_after_reranking": displaced_by,
                    "rrf_scores_available_for_all_inputs": all(
                        row["rrf_score"] is not None for row in ranked_inputs
                    ),
                    "reranker_scores_available_for_all_inputs": all(
                        row["reranker_score"] is not None for row in ranked_inputs
                    ),
                    "candidates": candidates,
                    "latency_ms": trace.get("latency_ms"),
                }
            )
    finally:
        retriever._rerank = original_rerank
        store.close()

    return {
        "audit": "K7.12 q11/q20 reranker@5 candidate forensics",
        "run_id": f"K7.12-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "identity": {
            **hashes,
            "dataset_declared_content_sha256": envelope.get("dataset_sha256"),
            "k79_run_id": reference.get("run_id"),
            "git_head": run_git("rev-parse", "HEAD"),
            "working_tree": "dirty" if run_git("status", "--porcelain") else "clean",
        },
        "configuration": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "queries": list(TARGET_QUERY_IDS),
            "embedding_model": settings.embed_model,
            "reranker_model": settings.reranker_model,
            "max_length": MAX_LENGTH,
            "device": "cpu",
            "candidate_pool": reference["common_configuration"]["effective_candidate_pool"],
            "reranker_candidate_count": RERANKER_CAP,
            "top_k": TOP_K,
            "generation": False,
            "source": "live local Qdrant + BM25 and saved K7.9 comparison artifact",
        },
        "queries": query_reports,
        "validation": {
            "query_count": len(query_reports),
            "all_reranker_calls_succeeded": all(
                row["reranker_status"] and row["reranker_status"].get("succeeded")
                for row in query_reports
            ),
            "all_rrf_orders_match_k79": all(row["rrf_order_matches_k79"] for row in query_reports),
            "all_reranker_inputs_match_k79": all(
                row["reranker_input_matches_k79"] for row in query_reports
            ),
            "all_reranked_orders_match_k79": all(
                row["reranked_order_matches_k79"] for row in query_reports
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K7.12 — q11/q20 Reranker Forensics",
        "",
        f"- Run: `{report['run_id']}`",
        f"- RRF order reproduced from K7.9: **{report['validation']['all_rrf_orders_match_k79']}**",
        f"- Model: `{report['configuration']['reranker_model']}`; `max_length={report['configuration']['max_length']}`; CPU; reranker input cap 5.",
        "",
    ]
    for query in report["queries"]:
        lines.extend(
            [
                f"## {query['query_id']}",
                "",
                f"> {query['question']}",
                "",
                f"- Gold: `{', '.join(query['gold_chunk_ids'])}`",
                f"- Gold rank: RRF **{query['rrf_gold_rank']}** → reranker output **{query['reranker_gold_rank']}** → final **{query['final_gold_rank']}**.",
                f"- RRF order/input/reranked order match K7.9: **{query['rrf_order_matches_k79']} / {query['reranker_input_matches_k79']} / {query['reranked_order_matches_k79']}**.",
                f"- Competitors above gold after reranking: {', '.join(f'`{value}`' for value in query['competitors_above_gold_after_reranking']) or 'none'}.",
                "",
                "| RRF rank | Candidate | Gold? | RRF score | Reranker input rank | Reranker score | Reranked rank | Final rank |",
                "|---:|---|:---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in sorted(
            query["candidates"],
            key=lambda row: (
                row["rank_within_reranker_output"] is None,
                row["rank_within_reranker_output"] or row["rrf_rank"] or 10**9,
            ),
        ):
            lines.append(
                f"| {candidate['rrf_rank']} | `{candidate['chunk_id']}` | "
                f"{'yes' if candidate['is_gold'] else 'no'} | {candidate['rrf_score']} | "
                f"{candidate['reranker_input_rank']} | {candidate['reranker_score']} | "
                f"{candidate['rank_within_reranker_output']} | {candidate['final_rank']} |"
            )
        lines.extend(["", "Candidate excerpts:", ""])
        for candidate in query["candidates"]:
            excerpt = candidate["text_excerpt"].replace("\n", " ").replace("|", "\\|")
            lines.append(f"- `{candidate['chunk_id']}`: {excerpt}")
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            f"- All reranker calls succeeded: **{report['validation']['all_reranker_calls_succeeded']}**",
            f"- RRF order matches K7.9: **{report['validation']['all_rrf_orders_match_k79']}**",
            f"- Actual reranker inputs match K7.9: **{report['validation']['all_reranker_inputs_match_k79']}**",
            f"- Reranked orders match K7.9: **{report['validation']['all_reranked_orders_match_k79']}**",
            "",
            "Scores are from this live local forensic run; if an order differs from K7.9, the report preserves both evidence sets and does not attribute the difference without further investigation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out_json or ROOT / f"results/metrics/k712_q11_q20_forensics_{stamp}.json"
    out_md = args.out_md or ROOT / f"results/metrics/k712_q11_q20_forensics_{stamp}.md"
    if out_json.exists() or out_md.exists():
        parser.error("Output file already exists; choose a new timestamped path")

    report = build_report()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(json.dumps(report["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
