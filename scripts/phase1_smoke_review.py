#!/usr/bin/env python3
"""Review Phase 1 acceptance answers against the evidence the LLM actually read.

`phase1_smoke.py --review-report` renders `debug_full_chunks[].text`, which
`run_single_query.py` truncates to 300 characters, and it renders all of them as
though each one were evidence. Both are wrong for a grounding judgement:

    p1_answer_02   gold chunk 415 chars  -> shown 300  ("We map FEVER clas...")
    p1_answer_10   gold chunk 2075 chars -> shown 300  (14% of the evidence)

The truncation point lands mid-answer. In that case the reviewer sees
"supports, refutes, not enough info" as absent, when chunk
tier1_rag_lewis2020_rs_0023 contains it verbatim:

    We map FEVER class labels (supports, refutes, or not enough info) to single
    output tokens and directly train with claim-class pairs.

11 of the 15 answerable cases have a gold chunk longer than 300 characters, so
the Phase 1 exit criterion that depends on this review -- "human review
confirms at least 18/20 correct final decisions" -- was being decided on
partial evidence.

The second problem is larger. Ten chunks were retrieved, but the generator's
diversity filter keeps at most two per document, so only five reached the
prompt:

    retrieved 10 -> filter_chunks_by_score 10 -> filter_chunks_by_diversity 5
                  -> build_context_with_ids 5 (2,470 of 3,000 chars)

tier1_rag_lewis2020_rs_0024 was dropped, and it carries "claim-class pairs" --
one of the two claims in the answer. A reviewer who sees it listed cannot tell
the model never read it, so "grounded=yes" can be recorded for a claim the
generator could not have grounded.

This script does neither of those things. It makes no LLM call, replays the
generator's own deterministic chunk-selection chain to recover which chunks
reached the prompt, and shows full corpus text for those chunks. Chunks that
were retrieved but dropped are listed separately and cannot ground a claim.

    python scripts/phase1_smoke_review.py                    # review what is pending
    python scripts/phase1_smoke_review.py --list             # pending ids, no prompts
    python scripts/phase1_smoke_review.py --show p1_answer_03
    python scripts/phase1_smoke_review.py --reviewer mayzar
    python scripts/phase1_smoke_review.py --all              # re-review everything
    python scripts/phase1_smoke_review.py --no-write         # dry run

After it finishes, evaluate the real gate with the script that owns it:

    python scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase1_smoke as smoke
from src.generation.generator import (
    DEFAULT_MAX_CHARS,
    PARTIAL_TRUST_CHUNK_LIMIT,
    build_context_with_ids,
    filter_chunks_by_diversity,
    filter_chunks_by_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "results/metrics/phase1_smoke.json"
DEFAULT_DATASET = REPO_ROOT / "data/evaluation/phase1_smoke_set_v1.json"
CORPUS_PATH = REPO_ROOT / "data/processed/chunks_semantic_v2.json"

RUBRIC = (
    "Rubrik: answer_correct=yes jika jawaban menjawab pertanyaan dan memuat semua fakta "
    "penting dari gold answer (parafrasa boleh); grounded=yes hanya jika setiap klaim "
    "material didukung chunk yang benar-benar ada di prompt LLM di bawah. Chunk bertanda "
    "[DROPPED] tidak pernah sampai ke LLM dan tidak bisa dijadikan dasar grounded. "
    "Chunk gold yang terambil saja tidak otomatis membuat jawaban grounded. Jika "
    "jawaban tidak lengkap, pilih answer_correct=no dan kategori incomplete_answer."
)


def load_corpus(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the frozen corpus, refusing to review against a different one.

    The report does not carry the corpus fingerprint; the dataset does. If the
    corpus on disk no longer matches the hash the acceptance set was frozen
    against, every chunk_id lookup below would silently resolve to the wrong
    text or to nothing at all.
    """
    if not CORPUS_PATH.exists():
        raise ValueError(f"Canonical corpus {CORPUS_PATH} not found")
    actual = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    expected = dataset.get("canonical_chunks_sha256")
    if expected and actual != expected:
        raise ValueError(
            "Canonical corpus no longer matches the hash this Phase 1 set was "
            f"frozen against.\n  expected {expected}\n  actual   {actual}\n"
            "Evidence text cannot be trusted for review. Restore the corpus or "
            "rebuild the acceptance set."
        )
    corpus: dict[str, dict[str, Any]] = {}
    for chunk in json.loads(CORPUS_PATH.read_text(encoding="utf-8")):
        if isinstance(chunk, dict) and chunk.get("chunk_id"):
            corpus[chunk["chunk_id"]] = chunk
    return corpus


def load_report(report_path: Path, dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("report_version") != "phase1-smoke-report-v1":
        raise ValueError("Not a Phase 1 smoke report v1")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Review requires a completed 20-case acceptance report")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if report.get("dataset_version") != dataset.get("dataset_version"):
        raise ValueError(
            f"Report was produced from dataset {report.get('dataset_version')!r} but "
            f"{dataset_path} is {dataset.get('dataset_version')!r}"
        )
    return report, dataset


def _replay_chunk(
    reported: dict[str, Any], corpus_entry: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild one chunk as the generator would have received it.

    `text` and `metadata` come from the corpus, not the report: the report's
    copy is truncated to 300 characters. `rerank_score` is taken from the
    report's recorded score because the corpus does not store ranking output.

    `filter_chunks_by_score` reads only rerank_score, and with the cross-encoder
    active its threshold is -3.0 against scores that rarely go below it, so the
    score stage is a near no-op; the binding constraint is the per-document
    diversity cap, which depends only on order and doc_id -- both exact.
    """
    score = reported.get("score")
    return {
        "chunk_id": reported.get("chunk_id"),
        "doc_id": reported.get("doc_id") or corpus_entry.get("doc_id"),
        "text": corpus_entry.get("text", ""),
        "metadata": corpus_entry.get("metadata") or {},
        "rerank_score": float(score) if isinstance(score, (int, float)) else None,
    }


def replay_prompt_selection(
    result: dict[str, Any], corpus: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Recover which chunks the generator actually placed in the LLM prompt.

    Mirrors GroqGenerator.generate(): PARTIAL_TRUST truncation, then
    filter_chunks_by_score, filter_chunks_by_diversity(max_per_doc=2), then
    build_context_with_ids under the character budget.
    """
    reported = result.get("debug_full_chunks") or result.get("reranked_chunks") or []
    usable = [row for row in reported if isinstance(row, dict) and row.get("chunk_id")]
    if not usable:
        return {"prompt_ids": [], "dropped_ids": [], "unresolved": [], "chars": 0, "note": "no chunks recorded"}

    unresolved = [
        row["chunk_id"] for row in usable
        if row["chunk_id"] not in corpus
    ]
    if unresolved:
        return {
            "prompt_ids": [],
            "dropped_ids": [],
            "unresolved": unresolved,
            "chars": 0,
            "note": (
                "chunk_id used by this run is absent from the frozen corpus: "
                + ", ".join(unresolved)
                + ". The pipeline and the corpus disagree; do not review this case."
            ),
        }

    candidates = [_replay_chunk(row, corpus[row["chunk_id"]]) for row in usable]
    decision = result.get("confidence_decision")
    if decision == "PARTIAL_TRUST":
        candidates = candidates[:PARTIAL_TRUST_CHUNK_LIMIT]

    by_score = filter_chunks_by_score(candidates)
    by_diversity = filter_chunks_by_diversity(by_score, max_per_doc=2)
    _context, prompt_ids = build_context_with_ids(by_diversity, max_chars=DEFAULT_MAX_CHARS)

    prompt_set = set(prompt_ids)
    return {
        "prompt_ids": prompt_ids,
        "dropped_ids": [row["chunk_id"] for row in usable if row["chunk_id"] not in prompt_set],
        "unresolved": [],
        "chars": sum(len(c["text"]) for c in by_diversity),
        "note": None,
    }


def render_case(
    row: dict[str, Any],
    position: int,
    corpus: dict[str, dict[str, Any]],
    output_fn: Callable[[str], None],
) -> dict[str, Any]:
    result = row.get("result") or {}
    selection = replay_prompt_selection(result, corpus)

    output_fn("")
    output_fn("=" * 78)
    output_fn(f"Review {position}/15 | {row.get('query_id')}")
    output_fn(f"Question: {row.get('query')}")
    output_fn(f"Expected answer: {row.get('gold_answer')}")
    output_fn(f"Gold chunk: {row.get('gold_chunk_id')}")
    output_fn(
        f"System status: {result.get('status')} | "
        f"confidence: {result.get('confidence_score')} | "
        f"decision: {result.get('confidence_decision')}"
    )
    if result.get("abstain_reason"):
        output_fn(f"Abstain reason: {result.get('abstain_reason')}")
    output_fn(f"System answer: {result.get('answer')}")

    if selection["note"]:
        output_fn("")
        output_fn(f"!! {selection['note']}")
        return selection

    output_fn("")
    output_fn(
        f"Evidence actually read by the LLM ({len(selection['prompt_ids'])} of "
        f"{len(selection['prompt_ids']) + len(selection['dropped_ids'])} retrieved, "
        f"~{selection['chars']} chars of {DEFAULT_MAX_CHARS}):"
    )
    for chunk_id in selection["prompt_ids"]:
        entry = corpus.get(chunk_id, {})
        marker = " [GOLD]" if chunk_id == row.get("gold_chunk_id") else ""
        section = (entry.get("metadata") or {}).get("section", "General")
        output_fn(
            f"- {chunk_id} | {entry.get('doc_id')} | {section}{marker} "
            f"| {len(entry.get('text', ''))} chars"
        )
        output_fn(f"  {entry.get('text', '')}")

    if selection["dropped_ids"]:
        output_fn("")
        output_fn(
            f"Retrieved but DROPPED before the prompt ({len(selection['dropped_ids'])}) "
            "-- cannot ground a claim:"
        )
        for chunk_id in selection["dropped_ids"]:
            entry = corpus.get(chunk_id, {})
            section = (entry.get("metadata") or {}).get("section", "General")
            marker = " [GOLD]" if chunk_id == row.get("gold_chunk_id") else ""
            output_fn(
                f"- [DROPPED] {chunk_id} | {entry.get('doc_id')} | {section}{marker}"
            )
            output_fn(f"  {entry.get('text', '')}")
    return selection


def pending_rows(
    report: dict[str, Any], include_reviewed: bool = False
) -> list[dict[str, Any]]:
    pending = []
    for row in report.get("records", []):
        if row.get("should_abstain") is not False:
            continue
        review = row.get("human_review") or {}
        labelled = (
            isinstance(review.get("answer_correct"), bool)
            and isinstance(review.get("grounded"), bool)
            and isinstance(review.get("reviewer"), str)
            and bool(review["reviewer"].strip())
        )
        needs_categories = labelled and (
            review["answer_correct"] is False or review["grounded"] is False
        ) and not smoke._valid_failure_categories(review.get("failure_categories"))
        if include_reviewed or not labelled or needs_categories:
            pending.append(row)
    return pending


def review(
    report_path: Path,
    dataset_path: Path,
    reviewer: Optional[str] = None,
    include_reviewed: bool = False,
    write: bool = True,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    report, dataset = load_report(report_path, dataset_path)
    corpus = load_corpus(dataset)

    rows = pending_rows(report, include_reviewed=include_reviewed)
    if not rows:
        output_fn("No pending answerable cases. Run --check-report to evaluate the gate.")
        return report

    output_fn(
        f"{len(rows)} case(s) to review | corpus {len(corpus)} chunks | "
        f"fingerprint verified against {dataset.get('dataset_version')}"
    )
    output_fn(RUBRIC)

    if not reviewer:
        reviewer = input_fn("Nama reviewer manusia: ").strip()
        while not reviewer:
            output_fn("Nama reviewer wajib diisi.")
            reviewer = input_fn("Nama reviewer manusia: ").strip()

    answerable = [r for r in report["records"] if r.get("should_abstain") is False]
    for row in rows:
        selection = render_case(row, answerable.index(row) + 1, corpus, output_fn)

        review_entry = row.setdefault("human_review", {})
        if selection["unresolved"]:
            output_fn(
                "Skipping: run and corpus disagree. Resolve the chunk_id mismatch "
                "before recording a judgement."
            )
            continue

        review_entry["answer_correct"] = smoke._ask_yes_no(
            "Is the answer correct and complete?", input_fn
        )
        review_entry["grounded"] = smoke._ask_yes_no(
            "Is every material claim supported by the prompt evidence above?", input_fn
        )
        review_entry["reviewer"] = reviewer
        review_entry["evidence_reviewed"] = {
            "chunks_retrieved": len(selection["prompt_ids"]) + len(selection["dropped_ids"]),
            "chunks_in_prompt": selection["prompt_ids"],
            "chunks_dropped": selection["dropped_ids"],
            "source": "scripts/phase1_smoke_review.py (full corpus text)",
        }

        failed = not review_entry["answer_correct"] or not review_entry["grounded"]
        if failed and not review_entry.get("failure_categories"):
            review_entry["failure_categories"] = smoke._ask_failure_categories(input_fn, output_fn)
        elif not failed:
            review_entry["failure_categories"] = []

        notes = input_fn("Review notes (Enter to keep existing/skip): ").strip()
        if notes:
            review_entry["notes"] = notes

        # Write after every case so an interrupted session keeps completed work.
        if write:
            smoke._save_report(report_path, report)
        output_fn("Review saved.")

    if write:
        smoke._save_report(report_path, report)
        output_fn(f"Annotations saved to {report_path}.")
    else:
        output_fn("--no-write: annotations were discarded, the report is unchanged.")
    output_fn("")
    output_fn(json.dumps(smoke.summarize(report), indent=2, ensure_ascii=False))
    output_fn(
        "Run `python scripts/phase1_smoke.py --check-report "
        f"{report_path}` to evaluate the Phase 1 gate."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review Phase 1 answers against the evidence the LLM actually read.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reviewer", help="Reviewer name; prompted for when omitted")
    parser.add_argument("--all", action="store_true", help="Re-review already labelled cases")
    parser.add_argument("--list", action="store_true", help="Print pending case ids and exit")
    parser.add_argument("--show", metavar="QUERY_ID", help="Print one case with full evidence and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not save annotations")
    args = parser.parse_args()

    try:
        report, dataset = load_report(args.report, args.dataset)

        if args.list:
            rows = pending_rows(report)
            print(f"{len(rows)} pending answerable case(s):")
            for row in rows:
                print(f"  {row.get('query_id')}  ({str(row.get('query'))[:60]}...)")
            return 0

        if args.show:
            corpus = load_corpus(dataset)
            target = next(
                (r for r in report["records"] if r.get("query_id") == args.show), None
            )
            if target is None:
                print(f"[phase1_smoke_review] no case with query_id {args.show!r}", file=sys.stderr)
                return 1
            answerable = [r for r in report["records"] if r.get("should_abstain") is False]
            position = answerable.index(target) + 1 if target in answerable else 1
            render_case(target, position, corpus, print)
            return 0

        review(
            args.report,
            args.dataset,
            reviewer=args.reviewer,
            include_reviewed=args.all,
            write=not args.no_write,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[phase1_smoke_review] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
