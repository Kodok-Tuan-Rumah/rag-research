"""Audit canonical query coverage and reranker headroom from saved K7 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/evaluation/retrieval_eval_dataset_v2.json"
K79_PATH = ROOT / "results/metrics/rrf_vs_reranker5_decision_20261005T141527Z.json"
K78_PATH = ROOT / "results/metrics/reranker_candidate_count_ablation_20261005T140325Z.json"
CORPUS_PATH = ROOT / "data/processed/chunks_semantic_v2.json"
DOCUMENTS_PATH = ROOT / "data/processed/documents.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gold_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("gold_chunk_ids")
    if ids is None and row.get("gold_chunk_id"):
        ids = [row["gold_chunk_id"]]
    return sorted(str(value) for value in (ids or []))


def _rank(ids: list[str], gold_ids: list[str]) -> int | None:
    gold = set(gold_ids)
    return next((index for index, chunk_id in enumerate(ids, 1) if chunk_id in gold), None)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_report(
    dataset_path: Path = DATASET_PATH,
    k79_path: Path = K79_PATH,
    k78_path: Path = K78_PATH,
) -> dict[str, Any]:
    dataset_envelope = _read_json(dataset_path)
    dataset_rows = dataset_envelope["records"]
    k79 = _read_json(k79_path)
    k78 = _read_json(k78_path)

    dataset_by_id = {row["query_id"]: row for row in dataset_rows}
    dataset_ids = [row["query_id"] for row in dataset_rows]
    _require(len(dataset_by_id) == len(dataset_rows), "Dataset has duplicate query_id values")

    dataset_file_hash = _sha256(dataset_path)
    declared_content_hash = dataset_envelope.get("dataset_sha256")
    _require(
        k79.get("identity", {}).get("dataset_file_sha256") == dataset_file_hash,
        "K7.9 dataset_file_sha256 does not match the current dataset bytes",
    )
    _require(
        k79.get("identity", {}).get("dataset_content_sha256")
        == dataset_envelope.get("dataset_sha256"),
        "K7.9 dataset content identity does not match the dataset envelope",
    )

    corpus_hashes: dict[str, str] = {}
    for label, path, expected in (
        ("documents", DOCUMENTS_PATH, k79.get("identity", {}).get("documents_sha256")),
        ("chunks", CORPUS_PATH, k79.get("identity", {}).get("chunks_sha256")),
    ):
        actual = _sha256(path)
        _require(actual == expected, f"K7.9 {label} hash does not match {path}")
        corpus_hashes[label] = actual

    modes: dict[str, dict[str, Any]] = {}
    query_rows: dict[str, dict[str, Any]] = {}
    for mode in ("rrf_only", "reranker5"):
        rows = k79["modes"][mode]["per_query_results"]
        _require([row["query_id"] for row in rows] == dataset_ids, f"K7.9 {mode} query IDs/order mismatch")
        mode_rows: dict[str, Any] = {}
        for row in rows:
            query_id = row["query_id"]
            source = dataset_by_id[query_id]
            gold_ids = _gold_ids(source)
            _require(gold_ids, f"{query_id}: dataset has no gold chunk IDs")
            _require(
                _gold_ids(row) == gold_ids,
                f"{query_id}: {mode} gold chunk IDs do not match the dataset",
            )
            _require(
                row.get("question") == source.get("question"),
                f"{query_id}: {mode} question does not match the dataset",
            )

            rrf_ids = list(row.get("rrf_ids") or [])
            final_ids = list(row.get("final_ids") or [])
            actual_rrf_rank = _rank(rrf_ids, gold_ids)
            actual_final_rank = _rank(final_ids, gold_ids)
            _require(
                actual_rrf_rank == row.get("rrf_rank"),
                f"{query_id}: {mode} stored RRF rank does not match rrf_ids",
            )
            _require(
                actual_final_rank == row.get("final_rank"),
                f"{query_id}: {mode} stored final rank does not match final_ids",
            )
            _require(
                bool(actual_final_rank is not None and actual_final_rank <= 10)
                == bool(row.get("hit@10")),
                f"{query_id}: {mode} hit@10 does not match final_ids",
            )

            item = {
                "query_id": query_id,
                "gold_chunk_ids": gold_ids,
                "rrf_gold_rank": actual_rrf_rank,
                "rrf_gold_in_top5": bool(actual_rrf_rank is not None and actual_rrf_rank <= 5),
                "reranker_input_count": row.get("effective_reranker_input_count"),
                "gold_in_reranker_input": any(
                    chunk_id in set(row.get("effective_reranker_input_ids") or [])
                    for chunk_id in gold_ids
                ),
                "final_gold_rank": actual_final_rank,
                "hit_at_10": bool(row.get("hit@10")),
            }
            mode_rows[query_id] = item
            if mode == "reranker5":
                query_rows[query_id] = {
                    **item,
                    "question": source.get("question"),
                    "difficulty_label": source.get("difficulty"),
                    "answer_type_label": source.get("answer_type"),
                    "rrf_only_final_gold_rank": query_by_id(
                        k79["modes"]["rrf_only"]["per_query_results"], query_id
                    ).get("final_rank"),
                    "rrf_only_hit_at_10": bool(
                        query_by_id(k79["modes"]["rrf_only"]["per_query_results"], query_id).get("hit@10")
                    ),
                }
        modes[mode] = {
            "metrics": k79["modes"][mode].get("aggregate_metrics"),
            "queries": list(mode_rows.values()),
        }

    rrf_ranks = [row["rrf_gold_rank"] for row in query_rows.values()]
    rank1_count = sum(rank == 1 for rank in rrf_ranks)
    rank_gt1_count = sum(rank is not None and rank > 1 for rank in rrf_ranks)
    exposed = [row for row in query_rows.values() if row["gold_in_reranker_input"]]
    regressions = [
        row for row in exposed
        if row["final_gold_rank"] is not None
        and row["rrf_gold_rank"] is not None
        and row["final_gold_rank"] > row["rrf_gold_rank"]
    ]
    improvements = [
        row for row in exposed
        if row["final_gold_rank"] is not None
        and row["rrf_gold_rank"] is not None
        and row["final_gold_rank"] < row["rrf_gold_rank"]
    ]
    q06 = query_rows.get("q06")

    k78_rows = k78["comparison_by_query"]
    _require(
        [row["query_id"] for row in k78_rows] == dataset_ids,
        "K7.8 query IDs/order do not match the dataset",
    )
    k78_by_id = {row["query_id"]: row for row in k78_rows}
    for query_id, row in query_rows.items():
        k78_row = k78_by_id[query_id]
        _require(
            k78_row.get("rrf_rank") == row["rrf_gold_rank"],
            f"{query_id}: K7.8 and K7.9 RRF gold ranks differ",
        )
        row["candidate_count_final_ranks"] = {
            str(count): k78_row.get(f"final_rank_{count}")
            for count in (5, 10, 15, 20)
        }

    difficulty_counts = dict(Counter(str(row.get("difficulty", "UNKNOWN")) for row in dataset_rows))
    answer_type_counts = dict(Counter(str(row.get("answer_type", "UNKNOWN")) for row in dataset_rows))
    n = len(dataset_rows)
    return {
        "audit": "K7.10 canonical retrieval evaluation coverage",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "dataset": str(dataset_path.relative_to(ROOT)),
            "k79_artifact": str(k79_path.relative_to(ROOT)),
            "k78_artifact": str(k78_path.relative_to(ROOT)),
            "dataset_file_sha256": dataset_file_hash,
            "dataset_declared_content_sha256": declared_content_hash,
            "declared_content_hash_matches_file_sha256": declared_content_hash == dataset_file_hash,
            "corpus_sha256": corpus_hashes,
            "k79_run_id": k79.get("run_id"),
            "k79_git_head": k79.get("identity", {}).get("git_head"),
            "k79_working_tree": k79.get("identity", {}).get("working_tree"),
            "k79_configuration": k79.get("common_configuration"),
        },
        "validation": {
            "status": "passed",
            "query_count": n,
            "query_ids_match_dataset": True,
            "gold_ids_match_dataset": True,
            "questions_match_dataset": True,
            "stored_ranks_match_candidate_order": True,
            "hit_at_10_matches_final_ids": True,
            "k78_k79_rrf_ranks_match": True,
        },
        "coverage_summary": {
            "rrf_gold_rank_1_count": rank1_count,
            "rrf_gold_rank_1_fraction": round(rank1_count / n, 4),
            "rrf_gold_rank_greater_than_1_count": rank_gt1_count,
            "rrf_gold_rank_buckets": {
                "rank_1": sum(rank == 1 for rank in rrf_ranks),
                "rank_2_to_5": sum(rank is not None and 2 <= rank <= 5 for rank in rrf_ranks),
                "rank_6_to_10": sum(rank is not None and 6 <= rank <= 10 for rank in rrf_ranks),
                "rank_11_to_20": sum(rank is not None and 11 <= rank <= 20 for rank in rrf_ranks),
                "rank_over_20": sum(rank is not None and rank > 20 for rank in rrf_ranks),
                "not_found_in_rrf_candidates": sum(rank is None for rank in rrf_ranks),
            },
            "gold_in_reranker5_input_count": len(exposed),
            "gold_rank_gt1_and_in_reranker5_input_count": sum(
                row["rrf_gold_rank"] is not None
                and row["rrf_gold_rank"] > 1
                and row["gold_in_reranker_input"]
                for row in query_rows.values()
            ),
            "reranker5_improved_gold_rank_count": len(improvements),
            "reranker5_regressed_gold_rank_count": len(regressions),
            "reranker5_unchanged_rank_count": len(exposed) - len(improvements) - len(regressions),
            "difficulty_label_counts": difficulty_counts,
            "answer_type_label_counts": answer_type_counts,
            "q06": {
                "rrf_gold_rank": q06["rrf_gold_rank"] if q06 else None,
                "reranker5_input_count": q06["reranker_input_count"] if q06 else None,
                "gold_in_reranker5_input": q06["gold_in_reranker_input"] if q06 else None,
                "hit_at_10_rrf_only": q06["rrf_only_hit_at_10"] if q06 else None,
                "hit_at_10_reranker5": q06["hit_at_10"] if q06 else None,
            },
            "metric_delta_reranker5_minus_rrf_only": k79.get("comparison", {}).get(
                "quality_delta_reranker5_minus_rrf_only"
            ),
        },
        "modes": modes,
        "per_query": list(query_rows.values()),
        "interpretation": [
            f"{rank1_count}/{n} gold chunks are already ranked first by RRF; these cases provide no headroom for reranking to improve the gold rank.",
            f"The only gold ranked below first place is q06 at RRF rank {q06['rrf_gold_rank'] if q06 else 'unknown'}; it is outside the reranker@5 input, so this set has zero rank>1 gold cases that reranker@5 can improve.",
            f"Among gold chunks actually exposed to reranker@5, {len(improvements)} improve and {len(regressions)} regress in final rank; see per-query rows.",
            "All difficulty and answer_type labels are UNKNOWN, so the current canonical set cannot be stratified by those dimensions.",
            "This 13-query audit describes this fixed set only; it does not establish general reranker performance on a broader query distribution.",
        ],
    }


def query_by_id(rows: list[dict[str, Any]], query_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["query_id"] == query_id)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["coverage_summary"]
    lines = [
        "# K7.10 — Canonical Retrieval Evaluation Coverage",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Validation",
        "",
        f"- Status: **{report['validation']['status']}**",
        f"- Dataset: `{report['sources']['dataset']}` ({report['validation']['query_count']} queries)",
        f"- Dataset file SHA-256: `{report['sources']['dataset_file_sha256']}`",
        f"- Declared content SHA-256: `{report['sources']['dataset_declared_content_sha256']}`",
        f"- Declared content hash equals file hash: **{report['sources']['declared_content_hash_matches_file_sha256']}**; the artifacts record both identities separately.",
        f"- K7.9 source tree recorded as: `{report['sources']['k79_working_tree']}` at `{report['sources']['k79_git_head']}`.",
        "- Query IDs, gold IDs, question text, stored ranks, hit@10, and K7.8/K7.9 RRF ranks validated.",
        "",
        "## Coverage and headroom",
        "",
        f"- Gold at RRF rank 1: **{summary['rrf_gold_rank_1_count']}/{report['validation']['query_count']}** ({summary['rrf_gold_rank_1_fraction']:.1%}).",
        f"- Gold rank >1: **{summary['rrf_gold_rank_greater_than_1_count']}**; rank >1 and inside reranker@5 input: **{summary['gold_rank_gt1_and_in_reranker5_input_count']}**.",
        f"- Reranker@5 exposed golds: **{summary['gold_in_reranker5_input_count']}**; rank improvements: **{summary['reranker5_improved_gold_rank_count']}**; regressions: **{summary['reranker5_regressed_gold_rank_count']}**.",
        f"- q06: RRF rank **{summary['q06']['rrf_gold_rank']}**, gold in reranker@5 input **{summary['q06']['gold_in_reranker5_input']}**, hit@10 RRF-only/reranker@5 **{summary['q06']['hit_at_10_rrf_only']}/{summary['q06']['hit_at_10_reranker5']}**.",
        f"- Difficulty labels: `{summary['difficulty_label_counts']}`; answer-type labels: `{summary['answer_type_label_counts']}`.",
        "",
        "## Per-query results",
        "",
        "| Query | Gold RRF rank | In reranker@5 | RRF-only final rank / Hit@10 | Reranker@5 final rank / Hit@10 | K7.8 final rank (5/10/15/20) |",
        "|---|---:|:---:|---:|---:|---|",
    ]
    for row in report["per_query"]:
        ranks = row["candidate_count_final_ranks"]
        lines.append(
            f"| {row['query_id']} | {row['rrf_gold_rank']} | "
            f"{'yes' if row['gold_in_reranker_input'] else 'no'} | "
            f"{row['rrf_only_final_gold_rank']} / {'yes' if row['rrf_only_hit_at_10'] else 'no'} | "
            f"{row['final_gold_rank']} / {'yes' if row['hit_at_10'] else 'no'} | "
            f"{ranks['5']}/{ranks['10']}/{ranks['15']}/{ranks['20']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.extend(
        [
            "",
            "The rows above are derived from saved K7.8/K7.9 results. This audit does not edit dataset labels, retrieval code, or production settings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--k79", type=Path, default=K79_PATH)
    parser.add_argument("--k78", type=Path, default=K78_PATH)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out_json or ROOT / f"results/metrics/k710_coverage_audit_{stamp}.json"
    out_md = args.out_md or ROOT / f"results/metrics/k710_coverage_audit_{stamp}.md"
    report = build_report(args.dataset, args.k79, args.k78)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(f"[validated] {report['validation']['query_count']} queries; "
          f"{report['coverage_summary']['rrf_gold_rank_1_count']} gold rank-1; "
          f"{report['coverage_summary']['gold_rank_gt1_and_in_reranker5_input_count']} rank>1 gold in reranker@5 input")


if __name__ == "__main__":
    main()
