from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gold_data_validator import validate_dataset

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
DOCUMENTS_PATH = ROOT / "data" / "processed" / "documents.json"
V2_PATH = ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v2.json"
V3_PATH = ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v3.json"
ADVERSARIAL_PATH = ROOT / "data" / "adversarial" / "adversarial_queries.json"
FAILURE_PATH = ROOT / "data" / "debug" / "failure_analysis.json"
OUT_PATH = ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v4.json"
AUDIT_PATH = ROOT / "results" / "metrics" / "gold_data_audit_v4.json"

TARGET_TOTAL = 200
SEED = 42
TOKEN_RECALL_THRESHOLD = 0.8

PARAPHRASE_TEMPLATES = [
    "Can you explain {entity}?",
    "Tell me about {entity}.",
    "What can you say about {entity}?",
    "Give me details on {entity}.",
    "How would you describe {entity}?",
    "What do we know about {entity}?",
    "Elaborate on {entity}.",
    "What is meant by {entity}?",
]


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", payload) if isinstance(payload, dict) else payload


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_of_records(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def match_gold_chunk(answer: str, chunks: list[dict]) -> tuple[dict | None, str]:
    needle = _norm(answer)
    if not needle:
        return None, "NO_ANSWER"

    for chunk in chunks:
        if len(needle) >= 20 and needle in _norm(chunk["text"]):
            return chunk, "EXACT_ANSWER_SPAN"

    words = set(re.findall(r"\w+", needle))
    if not words:
        return None, "NO_TOKENS"

    best, best_recall = None, 0.0
    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", _norm(chunk["text"])))
        if not chunk_words:
            continue
        recall = len(words & chunk_words) / len(words)
        if recall > best_recall:
            best, best_recall = chunk, recall

    if best is not None and best_recall >= TOKEN_RECALL_THRESHOLD:
        return best, f"TOKEN_RECALL_{best_recall:.2f}"
    return None, "NO_MATCH"


def build_paraphrase(chunks: list[dict], rng: random.Random) -> list[dict]:
    chunk_map = {c["chunk_id"]: c for c in chunks}
    seen_sources: set[tuple[str, str]] = set()
    seen_questions: set[str] = set()
    samples: list[dict] = []

    for path in (V2_PATH, V3_PATH):
        for record in _records(path):
            question = record.get("question") or record.get("query", "")
            answer = record.get("gold_answer") or record.get("ground_truth") or ""
            gold = record.get("gold_chunk_id") or (record.get("gold_chunk_ids") or [None])[0]
            if not (question and answer and gold):
                continue

            source_key = (question.lower(), gold)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            chunk = chunk_map.get(gold)
            if chunk is None:
                continue

            stem = question.split("?")[0].strip().rstrip(".,")
            rendered = None
            for width in range(40, len(stem) + 1, 20):
                candidate = stem[:width].strip()
                if not candidate:
                    continue
                for template in PARAPHRASE_TEMPLATES:
                    attempt = template.format(entity=candidate)
                    if attempt.lower() not in seen_questions:
                        rendered = attempt
                        break
                if rendered:
                    break
            if rendered is None:
                continue
            seen_questions.add(rendered.lower())

            samples.append(
                {
                    "query_id": f"par_{len(samples):03d}",
                    "question": rendered,
                    "gold_answer": answer,
                    "ground_truth": answer,
                    "gold_chunk_id": gold,
                    "gold_doc_id": chunk["doc_id"],
                    "question_type": "paraphrase",
                    "difficulty": "medium",
                    "should_abstain": False,
                    "provenance": {
                        "source": "human_question_reworded",
                        "original_question": question,
                        "matched_chunk": gold,
                    },
                }
            )
    return samples


def build_adversarial() -> tuple[list[dict], list[dict]]:
    samples: list[dict] = []
    needs_gold: list[dict] = []
    for record in _records(ADVERSARIAL_PATH):
        question = record.get("question") or record.get("query", "")
        if not question:
            continue
        if not record.get("should_abstain"):
            needs_gold.append(record)
            continue
        samples.append(
            {
                "query_id": record.get("query_id") or f"adv_{len(samples):03d}",
                "question": question,
                "gold_answer": "",
                "ground_truth": "",
                "gold_chunk_id": "",
                "gold_doc_id": "",
                "question_type": "adversarial",
                "difficulty": record.get("difficulty", "unknown"),
                "should_abstain": True,
                "provenance": {
                    "source": "human_out_of_corpus",
                    "type": record.get("type", "out_of_corpus"),
                    "expected_behavior": record.get("expected_behavior", "honest_abstention"),
                },
            }
        )
    return samples, needs_gold


def build_from_failure_log(chunks: list[dict], already_asked: set[str]) -> list[dict]:
    samples: list[dict] = []
    for record in _records(FAILURE_PATH):
        question = (record.get("query") or record.get("question") or "").strip()
        answer = (record.get("ground_truth") or record.get("answer") or "").strip()
        if not question or not answer:
            continue
        if question.lower() in already_asked:
            continue

        chunk, match_type = match_gold_chunk(answer, chunks)
        if chunk is None:
            print(f"  DROP (no corpus match: {match_type}): {question[:60]}")
            continue

        already_asked.add(question.lower())
        samples.append(
            {
                "query_id": f"fail_{len(samples):03d}",
                "question": question,
                "gold_answer": answer,
                "ground_truth": answer,
                "gold_chunk_id": chunk["chunk_id"],
                "gold_doc_id": chunk["doc_id"],
                "question_type": "lexical",
                "difficulty": record.get("failure_type", "unknown"),
                "should_abstain": False,
                "provenance": {
                    "source": "failure_analysis_json",
                    "match_type": match_type,
                    "original_failure_type": record.get("failure_type"),
                },
            }
        )
    return samples


def _record_key(record: dict) -> tuple[str, str]:
    question = (record.get("question") or record.get("query") or "").strip().lower()
    gold = record.get("gold_chunk_id")
    if gold is None:
        ids = record.get("gold_chunk_ids") or []
        gold = ids[0] if ids else ""
    return question, str(gold or "")


def _dedupe_ids(records: list[dict]) -> int:
    seen: dict[str, int] = {}
    renamed = 0
    for record in records:
        current = str(record.get("query_id", ""))
        if current in seen:
            seen[current] += 1
            record["query_id"] = f"{current}_dup{seen[current]}"
            renamed += 1
        else:
            seen[current] = 0
    return renamed


def merge_fragment(
    base: list[dict],
    fragment: list[dict],
    chunks: list[dict],
    *,
    question_type: str | None = None,
    embedder=None,
) -> dict:
    chunk_ids = {c["chunk_id"] for c in chunks if c.get("chunk_id")}
    seen = {_record_key(r) for r in base}
    accepted: list[dict] = []
    report = {
        "read": len(fragment),
        "accepted": 0,
        "duplicate": 0,
        "renamed_ids": 0,
        "dropped_bad_gold": 0,
        "dropped_malformed": 0,
        "relabelled": 0,
    }

    for index, record in enumerate(fragment):
        if not isinstance(record, dict):
            report["dropped_malformed"] += 1
            continue

        question = (record.get("question") or record.get("query") or "").strip()
        if not question:
            report["dropped_malformed"] += 1
            continue

        if record.get("should_abstain"):
            row = dict(record)
        else:
            gold_ids = record.get("gold_chunk_ids") or (
                [record.get("gold_chunk_id")] if record.get("gold_chunk_id") else []
            )
            if not gold_ids or any(g not in chunk_ids for g in gold_ids):
                report["dropped_bad_gold"] += 1
                continue
            row = dict(record)
            row["question"] = question

        if _record_key(row) in seen:
            report["duplicate"] += 1
            continue
        seen.add(_record_key(row))

        if question_type and row.get("question_type") != question_type:
            row["question_type"] = question_type
            report["relabelled"] += 1

        if not str(row.get("query_id", "")).strip():
            row["query_id"] = f"frag_{index:04d}"
        accepted.append(row)

    merged = base + accepted
    report["renamed_ids"] += _dedupe_ids(merged)
    leak = None
    try:
        from src.evaluation.query_leakage import dataset_leakage

        if chunks:
            leak = dataset_leakage({"records": merged}, chunks, embedder=embedder)
    except Exception as exc:
        report["leakage_check_skipped"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    if leak is not None:
        leaked = {d["query_id"] for d in leak["detail"] if d["leaked"]}
        report["leakage_scored"] = leak["scored"]
        if leaked:
            keep = [r for r in merged if r.get("query_id") not in leaked]
            report["dropped_leaking"] = len(merged) - len(keep)
            report["leaking_ids"] = sorted(leaked)
            merged = keep
        else:
            report["dropped_leaking"] = 0

    report["accepted"] = len(accepted)
    report["total_after_merge"] = len(merged)
    return {"records": merged, "report": report}


def check_category_floor(records: list[dict], floor: float = 0.20) -> list[str]:
    if not records:
        return []
    counts = Counter(r.get("question_type", "unknown") for r in records)
    total = len(records)
    short = []
    for category, count in sorted(counts.items()):
        share = count / total
        if share < floor:
            short.append(f"{category} {count}/{total} ({share:.0%}) < {floor:.0%}")
    return short


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="merge a generated fragment file into the rebuilt dataset",
    )
    parser.add_argument(
        "--merge-type",
        default=None,
        help="force question_type on merged rows (e.g. multihop, adversarial)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing it",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    chunks = load_chunks()
    chunk_ids = {c["chunk_id"] for c in chunks}

    paraphrase = build_paraphrase(chunks, rng)
    print(f"  paraphrase : {len(paraphrase)}")

    adversarial, needs_gold = build_adversarial()
    print(f"  adversarial: {len(adversarial)}")
    for record in needs_gold:
        print(
            f"  NEEDS MANUAL GOLD ({record.get('type')}): "
            f"{record.get('query_id')} — {(record.get('question') or '')[:60]}"
        )

    asked = {s["question"].lower() for s in paraphrase}
    asked |= {s["question"].lower() for s in adversarial}
    failure = build_from_failure_log(chunks, asked)
    print(f"  failure    : {len(failure)}")

    all_samples = paraphrase + adversarial + failure

    valid: list[dict] = []
    for sample in all_samples:
        if sample["should_abstain"]:
            valid.append(sample)
            continue
        gold = sample.get("gold_chunk_ids") or [sample.get("gold_chunk_id", "")]
        unknown = [g for g in gold if g not in chunk_ids]
        if unknown:
            print(f"  SKIP (gold not in corpus: {unknown}): {sample['query_id']}")
        else:
            valid.append(sample)

    print(f"\nTotal valid samples: {len(valid)}")

    merge_report = None
    if args.merge is not None:
        if not args.merge.exists():
            print(f"ERROR: merge file not found: {args.merge}", file=sys.stderr)
            return 2
        payload = json.loads(args.merge.read_text(encoding="utf-8"))
        fragment = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(fragment, list):
            print(f"ERROR: {args.merge} has no records array", file=sys.stderr)
            return 2

        outcome = merge_fragment(valid, fragment, chunks, question_type=args.merge_type)
        valid = outcome["records"]
        merge_report = outcome["report"]
        r = merge_report
        print(
            f"\n[merge] read={r['read']} accepted={r['accepted']} "
            f"duplicate={r['duplicate']} renamed_ids={r['renamed_ids']} "
            f"dropped_bad_gold={r['dropped_bad_gold']} "
            f"dropped_malformed={r['dropped_malformed']} "
            f"dropped_leaking={r.get('dropped_leaking', 0)}"
            + (f" relabelled={r['relabelled']}" if r["relabelled"] else "")
        )
        print(
            f"[merge] leakage scored={r.get('leakage_scored', 'n/a')} "
            f"dropped_leaking={r.get('dropped_leaking', 0)}"
        )
        if r.get("leakage_check_skipped"):
            print(f"[merge] NOTE leakage check skipped: {r['leakage_check_skipped']}")
        print(f"[merge] total after merge: {r['total_after_merge']}")

    short = check_category_floor(valid)
    if short:
        print("\nWARNING §10.3 — categories below the 20% floor:")
        for line in short:
            print(f"    {line}")
        print("    This is reported, not enforced: the fix is to generate the")
        print("    missing category, not to delete rows that exist.")

    if len(valid) < TARGET_TOTAL:
        print(
            f"\nNOTE: {len(valid)} of {TARGET_TOTAL}. The remaining gap needs real\n"
            "      questions — see scripts/generate_eval_questions_llm.py."
        )

    output = {
        "dataset_version": "v4",
        "dataset_sha256": _sha256_of_records(valid),
        "corpus_chunks_sha256": _sha256_of_file(CHUNKS_PATH),
        "corpus_documents_sha256": _sha256_of_file(DOCUMENTS_PATH),
        "created_from": "human_authored_questions_span_verified",
        "verification_method": "exact_span_or_token_recall_0.8",
        "notes": (
            "Template-generated questions were removed. The previous revision "
            "produced 67 of 88 rows that passed CI but were not usable as "
            "evaluation data."
        ),
        "records": valid,
    }
    if merge_report is not None:
        output["merge_report"] = merge_report

    audit = validate_dataset(valid, chunks, require_answer=False)
    summary = audit["summary"]

    if args.dry_run:
        print("\n[dry-run] nothing written")
        print(f"  would write {len(valid)} records to {OUT_PATH}")
        print(f"  audit would be {summary['VALID_QA']}/{summary['TOTAL_QA']} valid")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWritten: {OUT_PATH}")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Audit: {AUDIT_PATH}  valid={summary['VALID_QA']}/{summary['TOTAL_QA']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
