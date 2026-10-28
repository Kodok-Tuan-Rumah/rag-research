from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_single_query as runner
from scripts.export_phase1_blind_review import export as export_blind_review

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/evaluation/phase1_smoke_set_v3.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/metrics/phase1_smoke_v3.json"
VALID_STATUSES = {"ANSWERED", "INSUFFICIENT_CONTEXT"}
REQUIRED_LOG_FIELDS = {
    "query_id",
    "query",
    "query_type",
    "retrieval_scores",
    "reranker_scores",
    "chunks_used",
    "prompt_chunk_ids",
    "confidence_score",
    "failure_type",
    "answer_status",
    "latency_ms",
    "retry_triggered",
}
REQUIRED_LATENCY_FIELDS = {"retrieval", "reranker", "generation", "total"}
MIN_CORRECT_CASES = 18
REVIEW_FAILURE_CATEGORIES = {
    "answer_incorrect",
    "unsupported_claim",
    "retrieval_miss",
    "overly_specific",
    "incomplete_answer",
    "false_abstention",
    "provider_error",
    "incorrect_gold",
    "other",
}


def load_cases(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases):
        raise ValueError("Phase 1 dataset must contain a records array")
    if payload.get("dataset_version") not in {
        "phase1-smoke-v1",
        "phase1-smoke-v2",
        "phase1-smoke-v3",
    }:
        raise ValueError("Unsupported Phase 1 smoke dataset version")
    if len(cases) != 20 or payload.get("expected_cases") != 20:
        raise ValueError("Phase 1 acceptance requires exactly 20 frozen cases")

    ids = [row.get("query_id") for row in cases]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 20:
        raise ValueError("Every Phase 1 case must have a unique query_id")
    answerable = [row for row in cases if row.get("should_abstain") is False]
    abstain = [row for row in cases if row.get("should_abstain") is True]
    if len(answerable) != 15 or len(abstain) != 5:
        raise ValueError("Phase 1 acceptance set must contain 15 answerable and 5 abstain cases")
    for row in cases:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{row['query_id']}: question is empty")
        if row["should_abstain"] is False:
            if not row.get("gold_answer") or not row.get("gold_chunk_id"):
                raise ValueError(f"{row['query_id']}: answerable case needs gold answer and chunk")
        elif row.get("gold_answer") not in (None, "") or row.get("gold_chunk_id") not in (None, ""):
            raise ValueError(
                f"{row['query_id']}: abstain case must not declare a gold answer/chunk"
            )

    corpus_path = REPO_ROOT / payload.get(
        "canonical_chunks_path", "data/processed/chunks_semantic_v2.json"
    )
    actual_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if actual_hash != payload.get("canonical_chunks_sha256"):
        raise ValueError(
            "Canonical chunk corpus changed since this Phase 1 set was frozen; "
            "review/rebuild the set before using it as an acceptance gate"
        )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_ids = {chunk.get("chunk_id") for chunk in corpus if isinstance(chunk, dict)}
    missing_gold = [
        row["query_id"]
        for row in answerable
        if row["gold_chunk_id"] not in corpus_ids
        or any(
            evidence_id not in corpus_ids
            for evidence_id in row.get("gold_evidence_chunk_ids", [row["gold_chunk_id"]])
        )
    ]
    if missing_gold:
        raise ValueError(f"Gold chunks missing from canonical corpus: {', '.join(missing_gold)}")
    return payload


def _valid_confidence(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def _valid_failure_categories(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(category, str) and category in REVIEW_FAILURE_CATEGORIES
            for category in value
        )
    )


def _is_provider_error(result: dict[str, Any]) -> bool:
    reason = str(result.get("abstain_reason") or result.get("error") or "").casefold()
    return any(
        marker in reason
        for marker in (
            "provider rejected",
            "response_format",
            "provider error",
            "api error",
            "provider_error",
            "generation_exception",
            "generation failed",
            "authentication error",
            "rate limit",
            "rate_limit_exceeded",
            "429",
            "timed out",
        )
    )


def _make_event(query_id: str, result: dict[str, Any]) -> dict[str, Any]:
    latency = result.get("latency_ms") or {}
    return {
        "query_id": query_id,
        "query": result.get("query"),
        "query_type": result.get("query_type"),
        "retrieval_scores": result.get("retrieval_scores", []),
        "reranker_scores": result.get("reranker_scores", []),
        "chunks_used": result.get("chunks_used", []),
        "prompt_chunk_ids": result.get("prompt_chunk_ids", []),
        "confidence_score": result.get("confidence_score"),
        "failure_type": result.get("failure_type", "none"),
        "answer_status": result.get("answer_status", "failed"),
        "latency_ms": {key: latency.get(key) for key in REQUIRED_LATENCY_FIELDS},
        "retry_triggered": bool(result.get("retry_triggered", False)),
    }


def run_smoke(dataset: dict[str, Any]) -> dict[str, Any]:
    # Acceptance runs must always exercise the real generation path.
    os.environ.pop("SMOKE_STOP_BEFORE_GEN", None)
    corpus_path = REPO_ROOT / dataset.get(
        "canonical_chunks_path", "data/processed/chunks_semantic_v2.json"
    )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    records = []
    for index, case in enumerate(dataset["records"], 1):
        entry: dict[str, Any] = {
            "index": index,
            "query_id": case["query_id"],
            "query": case["question"],
            "should_abstain": case["should_abstain"],
            "gold_answer": case.get("gold_answer"),
            "gold_chunk_id": case.get("gold_chunk_id"),
            "gold_evidence_chunks": [
                {
                    "chunk_id": chunk_id,
                    "doc_id": corpus_by_id[chunk_id].get("doc_id"),
                    "section": (corpus_by_id[chunk_id].get("metadata") or {}).get("section"),
                    "text": corpus_by_id[chunk_id].get("text", ""),
                }
                for chunk_id in case.get("gold_evidence_chunk_ids", [case.get("gold_chunk_id")])
                if chunk_id in corpus_by_id
            ],
            "exception": None,
            "human_review": {
                "answer_correct": None,
                "grounded": None,
                "reviewer": None,
                "notes": None,
                "failure_categories": [],
            },
        }
        try:
            result = runner.run_single_query(
                query=case["question"],
                mode="full",
                ground_truth=case.get("gold_answer"),
                save_output=False,
            )
            entry["result"] = result
            entry["structured_log"] = _make_event(case["query_id"], result)
        except Exception as exc:
            entry["exception"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
            entry["result"] = {
                "query": case["question"],
                "status": "FAILED",
                "answer_status": "failed",
                "failure_type": "pipeline_error",
                "confidence_score": 0.0,
                "retry_triggered": False,
                "generation_attempts": 0,
                "answer": "",
                "abstain_reason": entry["exception"],
            }
            entry["structured_log"] = _make_event(case["query_id"], entry["result"])

        result = entry["result"]
        status = result.get("status")
        confidence = result.get("confidence_score")
        qdrant = result.get("qdrant_index") or {}
        entry["runtime_checks"] = {
            "valid_status": status in VALID_STATUSES,
            "confidence_in_range": _valid_confidence(confidence),
            "retry_within_limit": result.get("generation_attempts", 0) <= 2,
            "log_schema_complete": REQUIRED_LOG_FIELDS <= entry["structured_log"].keys(),
            "log_latency_complete": REQUIRED_LATENCY_FIELDS
            <= (entry["structured_log"].get("latency_ms", {}).keys()),
            "qdrant_count_matches": qdrant.get("vector_count")
            == dataset.get("expected_qdrant_vectors"),
            "dense_probe_nonempty": (qdrant.get("probe_hit_count") or 0) > 0,
        }
        print(
            f"[{index:02d}/20] {status or 'CRASHED'} | {case['query_id']} | {case['question'][:72]}"
        )
        records.append(entry)

    report = {
        "report_version": "phase1-smoke-report-v1",
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": hashlib.sha256(
            json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "mode": "full",
        "generation_enabled": True,
        "quality_review_required": True,
        "acceptance": {
            "case_count": 20,
            "answerable_count": 15,
            "abstain_count": 5,
            "minimum_correct_cases": MIN_CORRECT_CASES,
            "maximum_confident_wrong": 0,
        },
        "records": records,
    }
    report["summary"] = summarize(report)
    return report


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    records = report.get("records", [])
    runtime_pass = len(records) == 20
    runtime_failures = []
    for row in records:
        checks = row.get("runtime_checks", {})
        if (
            row.get("exception")
            or set(checks)
            != {
                "valid_status",
                "confidence_in_range",
                "retry_within_limit",
                "log_schema_complete",
                "log_latency_complete",
                "qdrant_count_matches",
                "dense_probe_nonempty",
            }
            or not all(checks.values())
        ):
            runtime_pass = False
            runtime_failures.append(row.get("query_id"))

    reviewed = True
    failure_reasons_complete = True
    correct = 0
    confident_wrong = 0
    provider_errors = 0
    decision_results = []
    for row in records:
        result = row.get("result", {})
        review = row.get("human_review") or {}
        provider_error = _is_provider_error(result)
        provider_errors += int(provider_error)
        if row.get("should_abstain") is True:
            decision_correct = (
                result.get("status") == "INSUFFICIENT_CONTEXT"
                and bool(result.get("abstain_reason"))
                and not provider_error
            )
        else:
            answer_correct = review.get("answer_correct")
            grounded = review.get("grounded")
            reviewer = review.get("reviewer")
            if (
                not isinstance(answer_correct, bool)
                or not isinstance(grounded, bool)
                or not isinstance(reviewer, str)
                or not reviewer.strip()
            ):
                reviewed = False
                decision_correct = False
            else:
                failed_review = (
                    not answer_correct or not grounded or result.get("status") != "ANSWERED"
                )
                categories = review.get("failure_categories")
                if failed_review and not _valid_failure_categories(categories):
                    failure_reasons_complete = False
                decision_correct = (
                    result.get("status") == "ANSWERED" and answer_correct and grounded
                )
                if (
                    result.get("status") == "ANSWERED"
                    and not (answer_correct and grounded)
                    and result.get("confidence_decision") == "GENERATE"
                ):
                    confident_wrong += 1
        decision_results.append(bool(decision_correct))
        correct += bool(decision_correct)

    quality_pass = (
        reviewed
        and failure_reasons_complete
        and correct >= MIN_CORRECT_CASES
        and confident_wrong == 0
        and provider_errors == 0
    )
    return {
        "total": len(records),
        "runtime_pass": runtime_pass,
        "runtime_failed_query_ids": runtime_failures,
        "human_review_complete": reviewed,
        "failure_reasons_complete": failure_reasons_complete,
        "correct_final_decisions": correct if reviewed else None,
        "minimum_correct_final_decisions": MIN_CORRECT_CASES,
        "confident_wrong_answers": confident_wrong,
        "provider_errors": provider_errors,
        "quality_pass": quality_pass,
        "phase1_pass": runtime_pass and quality_pass,
    }


def check_reviewed_report(report_path: Path) -> tuple[bool, dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("report_version") != "phase1-smoke-report-v1":
        raise ValueError("Not a Phase 1 smoke report v1")
    report["summary"] = summarize(report)
    return report["summary"]["phase1_pass"], report


def _save_report(report_path: Path, report: dict[str, Any]) -> None:
    report["summary"] = summarize(report)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ask_yes_no(prompt: str, input_fn=input) -> bool:
    while True:
        answer = input_fn(f"{prompt} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Masukkan y atau n.")


def _ask_failure_categories(input_fn=input, output_fn=print) -> list[str]:
    choices = ", ".join(sorted(REVIEW_FAILURE_CATEGORIES))
    while True:
        raw = input_fn(
            f"Failure categories, dipisahkan koma ({choices}; wajib untuk jawaban gagal): "
        ).strip()
        categories = sorted({item.strip() for item in raw.split(",") if item.strip()})
        if categories and all(item in REVIEW_FAILURE_CATEGORIES for item in categories):
            return categories
        output_fn("Pilih minimal satu kategori yang tercantum di atas.")


def review_report(
    report_path: Path,
    input_fn=input,
    output_fn=print,
) -> dict[str, Any]:
    """Collect explicit human annotations for answerable acceptance cases."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("report_version") != "phase1-smoke-report-v1":
        raise ValueError("Not a Phase 1 smoke report v1")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Human review requires a completed 20-case acceptance report")

    answerable = [row for row in records if row.get("should_abstain") is False]
    if len(answerable) != 15:
        raise ValueError("Human review requires exactly 15 answerable cases")

    pending = []
    for row in answerable:
        review = row.get("human_review") or {}
        labels_present = (
            isinstance(review.get("answer_correct"), bool)
            and isinstance(review.get("grounded"), bool)
            and isinstance(review.get("reviewer"), str)
            and bool(review.get("reviewer", "").strip())
        )
        has_failure = labels_present and (
            review["answer_correct"] is False
            or review["grounded"] is False
            or (row.get("result") or {}).get("status") != "ANSWERED"
        )
        has_failure_categories = _valid_failure_categories(review.get("failure_categories"))
        if not labels_present or (has_failure and not has_failure_categories):
            pending.append(row)

    reviewer = ""
    if any(
        not isinstance((row.get("human_review") or {}).get("reviewer"), str)
        or not (row.get("human_review") or {}).get("reviewer", "").strip()
        for row in pending
    ):
        reviewer = input_fn("Nama reviewer manusia: ").strip()
        while not reviewer:
            output_fn("Nama reviewer wajib diisi.")
            reviewer = input_fn("Nama reviewer manusia: ").strip()

    output_fn(
        "Rubrik: answer_correct=yes hanya jika sistem menjawab pertanyaan dan memuat semua fakta "
        "penting dari gold answer (parafrasa boleh); grounded=yes hanya jika setiap klaim "
        "material didukung konteks yang benar-benar dikirim ke generator. Chunk gold yang "
        "terambil tetapi tidak masuk prompt tidak membuktikan groundedness. Jika pertanyaan "
        "answerable tetapi sistem abstain, pilih answer_correct=no dan kategori false_abstention."
    )

    for row in pending:
        position = answerable.index(row) + 1
        result = row.get("result") or {}
        output_fn("\n" + "=" * 78)
        output_fn(f"Review {position}/15 | {row.get('query_id')}")
        output_fn(f"Question: {row.get('query')}")
        output_fn(f"Expected answer (full): {row.get('gold_answer')}")
        output_fn(f"Gold chunk: {row.get('gold_chunk_id')}")
        for gold_chunk in row.get("gold_evidence_chunks") or []:
            output_fn(
                f"Gold evidence (full): {gold_chunk.get('chunk_id')} | "
                f"{gold_chunk.get('doc_id')} | {gold_chunk.get('section')}\n"
                f"{gold_chunk.get('text', '')}"
            )
        retrieval_confidence = result.get("retrieval_confidence") or {}
        output_fn(
            f"System status: {result.get('status')} | answer confidence: "
            f"{result.get('confidence_score')} | retrieval confidence: "
            f"{retrieval_confidence.get('confidence_score')} "
            f"({retrieval_confidence.get('decision')})"
        )
        output_fn(
            f"Generation attempts: {result.get('generation_attempts')} | "
            f"retry: {result.get('retry_triggered')} | abstain reason: "
            f"{result.get('abstain_reason')}"
        )
        output_fn(f"System answer: {result.get('answer')}")

        contexts = result.get("generation_contexts") or []
        if contexts:
            for attempt in contexts:
                output_fn(
                    f"Exact generation context, attempt {attempt.get('attempt')} "
                    f"(chunk IDs: {attempt.get('prompt_chunk_ids')}):\n"
                    f"{attempt.get('context', '')}"
                )
        else:
            output_fn("No generation prompt was sent. Retrieved evidence chunks:")
            for chunk in result.get("debug_full_chunks") or []:
                output_fn(
                    f"- {chunk.get('chunk_id')} | {chunk.get('doc_id')} | "
                    f"{chunk.get('section')}\n{chunk.get('text', '')}"
                )

        review = row.setdefault("human_review", {})
        labels_present = (
            isinstance(review.get("answer_correct"), bool)
            and isinstance(review.get("grounded"), bool)
            and isinstance(review.get("reviewer"), str)
            and bool(review.get("reviewer", "").strip())
        )
        if not labels_present:
            review["answer_correct"] = _ask_yes_no("Is the answer correct and complete?", input_fn)
            review["grounded"] = _ask_yes_no(
                "Is every material claim supported by the retrieved evidence?", input_fn
            )
            review["reviewer"] = reviewer

        failed_review = (
            not review["answer_correct"]
            or not review["grounded"]
            or result.get("status") != "ANSWERED"
        )
        if failed_review and not review.get("failure_categories"):
            review["failure_categories"] = _ask_failure_categories(input_fn, output_fn)
        elif not failed_review:
            review["failure_categories"] = []

        previous_notes = review.get("notes") or ""
        while True:
            entered_notes = input_fn(
                "Review notes (min 8 chars; Enter to keep existing/skip): "
            ).strip()
            if not entered_notes:
                break
            if entered_notes.casefold() in {"y", "yes", "n", "no"} or len(entered_notes) < 8:
                output_fn(
                    "Catatan terlalu pendek atau hanya jawaban y/n; tulis alasan yang jelas atau Enter."
                )
                continue
            review["notes"] = entered_notes
            break
        if not entered_notes and not previous_notes:
            review["notes"] = None

        # Save after every case so an interrupted terminal session preserves
        # completed human work. The checker can be run after the final case.
        _save_report(report_path, report)
        output_fn("Review saved.")

    _save_report(report_path, report)
    output_fn(f"\nReviewed or resumed {len(pending)} pending answerable case(s).")
    output_fn(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    output_fn(
        "Run --review-report again to continue pending reviews, then --check-report "
        "to evaluate the final Phase 1 gate."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-report",
        type=Path,
        help="Evaluate a completed report after filling human_review annotations",
    )
    parser.add_argument(
        "--review-report",
        type=Path,
        help="Interactively review answerable cases and save human_review annotations",
    )
    parser.add_argument(
        "--review-pack",
        action="store_true",
        help="Generate the Phase 1 blind review pack from the latest smoke report",
    )
    parser.add_argument(
        "--require-no-errors",
        action="store_true",
        help="Fail unless the report has zero provider errors",
    )
    args = parser.parse_args()

    if args.check_report and args.review_report:
        parser.error("--check-report and --review-report cannot be used together")

    if args.review_report:
        try:
            review_report(args.review_report)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[phase1_smoke] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.check_report:
        passed, report = check_reviewed_report(args.check_report)
        _save_report(args.check_report, report)
        print(json.dumps(report["summary"], indent=2))
        return 0 if passed else 1

    if args.review_pack:
        from scripts.export_phase1_blind_review import DEFAULT_IDS

        count = export_blind_review(
            DEFAULT_OUTPUT, REPO_ROOT / "results/metrics/phase1_blind_review_v3.json", DEFAULT_IDS
        )
        print(f"Exported {count} gold-blind cases to results/metrics/phase1_blind_review_v3.json")
        return 0

    try:
        dataset = load_cases(args.dataset)
        report = run_smoke(dataset)
    except Exception as exc:
        print(f"[phase1_smoke] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_report = archive_existing_report(args.output)
    if previous_report:
        try:
            archived_path = previous_report.relative_to(REPO_ROOT)
        except ValueError:
            archived_path = previous_report.resolve()
        report["previous_report_archive"] = str(archived_path)
        print(f"Previous report archived: {previous_report}")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report saved: {args.output}")
    print("Quality review is pending; annotate answerable cases before --check-report.")
    if args.require_no_errors and report["summary"].get("provider_errors", 0) != 0:
        return 1
    return 0 if report["summary"]["runtime_pass"] else 1


def archive_existing_report(report_path: Path) -> Path | None:
    """Copy an existing acceptance report before a fresh run replaces it."""
    if not report_path.exists():
        return None
    archive_dir = report_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_path = archive_dir / f"{report_path.stem}_{timestamp}{report_path.suffix}"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / (
            f"{report_path.stem}_{timestamp}_{suffix}{report_path.suffix}"
        )
        suffix += 1
    shutil.copy2(report_path, archive_path)
    return archive_path


if __name__ == "__main__":
    sys.exit(main())
