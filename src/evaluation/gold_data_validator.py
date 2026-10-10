"""Gold-label / ground-truth dataset validator.

Membuktikan apakah dataset ground truth untuk retrieval evaluation valid:
- setiap item punya question + gold chunk id
- gold chunk benar-benar ada di corpus
- doc_id gold konsisten dengan chunk di corpus
- tidak ada duplicate question / gold chunk yang ambigu

Item yang invalid diklasifikasikan ``evaluation_data_error`` dan HARUS tidak
dihitung sebagai retrieval failure biasa maupun menguntungkan metric.

CLI:
    python -m src.evaluation.gold_data_validator \
        --dataset data/evaluation/retrieval_eval_dataset.json \
        --chunks data/processed/chunks_semantic_v2.json \
        --out results/metrics/gold_data_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids

logger = logging.getLogger("rag.gold_data_validator")

ITEM_ERROR = "evaluation_data_error"


def _get_question(item: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "query"):
        value = item.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _get_answer(item: Dict[str, Any]) -> Optional[str]:
    for key in ("gold_answer", "answer", "ground_truth"):
        value = item.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def validate_dataset(
    items: List[Dict[str, Any]] | Dict[str, Any],
    chunks: List[Dict[str, Any]],
    *,
    require_answer: bool = False,
) -> Dict[str, Any]:
    items = dataset_records(items)
    chunk_index: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            chunk_index[str(cid)] = c

    question_counts = Counter()
    for it in items:
        q = _get_question(it)
        if q:
            question_counts[q.lower()] += 1

    rows: List[Dict[str, Any]] = []
    counts = {
        "TOTAL_QA": len(items),
        "VALID_QA": 0,
        "MISSING_GOLD_CHUNK": 0,
        "MISSING_GOLD_DOC": 0,
        "MISSING_QUESTION": 0,
        "MISSING_ANSWER": 0,
        "DUPLICATE_IDS": 0,
        "INVALID_MAPPING": 0,
    }

    seen_pairs: Counter = Counter()
    for it in items:
        q = _get_question(it)
        gold_item = it
        if "gold_chunk_id" not in it and "gold_chunk_ids" not in it:
            for alias in ("source_chunk", "source_chunk_id", "chunk_id", "gold_chunk"):
                if alias in it:
                    gold_item = {"gold_chunk_id": it[alias]}
                    break
        try:
            gold_ids = normalize_gold_chunk_ids(gold_item)
            gold_error = None
        except ValueError as exc:
            gold_ids = []
            gold_error = str(exc)
        answer = _get_answer(it)
        row: Dict[str, Any] = {
            "question": q,
            "gold_chunk_id": gold_ids[0] if len(gold_ids) == 1 else None,
            "gold_chunk_ids": gold_ids,
            "status": "ok",
            "error_type": None,
            "details": [],
        }

        if not q:
            row["status"] = ITEM_ERROR
            row["error_type"] = "missing_question"
            row["details"].append("item has no question/query field")
            counts["MISSING_QUESTION"] += 1
        if require_answer and not answer:
            row["status"] = ITEM_ERROR
            row["error_type"] = "missing_answer"
            row["details"].append("item has no answer/gold_answer/ground_truth field")
            counts["MISSING_ANSWER"] += 1

        if gold_error:
            row["status"] = ITEM_ERROR
            row["error_type"] = "invalid_gold_chunk_ids"
            row["details"].append(gold_error)
            counts["MISSING_GOLD_CHUNK"] += 1
        elif not gold_ids:
            row["status"] = ITEM_ERROR
            row["error_type"] = "missing_gold_chunk_id"
            row["details"].append("item has no gold chunk reference")
            counts["MISSING_GOLD_CHUNK"] += 1
        else:
            missing_chunks = [gold for gold in gold_ids if gold not in chunk_index]
            if missing_chunks:
                row["status"] = ITEM_ERROR
                row["error_type"] = "missing_gold_chunk"
                row["details"].extend(
                    f"gold chunk '{gold}' not present in corpus" for gold in missing_chunks
                )
                counts["MISSING_GOLD_CHUNK"] += 1
            else:
                item_doc = it.get("doc_id") or it.get("gold_doc_id")
                for gold in gold_ids:
                    chunk = chunk_index[gold]
                    corpus_doc = str(chunk.get("doc_id", "")).strip()
                    if item_doc and corpus_doc and str(item_doc).strip() != corpus_doc:
                        row["status"] = ITEM_ERROR
                        row["error_type"] = "invalid_mapping"
                        row["details"].append(
                            f"item doc_id '{item_doc}' != corpus doc_id '{corpus_doc}' for '{gold}'"
                        )
                        counts["INVALID_MAPPING"] += 1
                    elif not corpus_doc:
                        row["status"] = ITEM_ERROR
                        row["error_type"] = "missing_gold_doc"
                        row["details"].append(f"corpus chunk '{gold}' has no doc_id")
                        counts["MISSING_GOLD_DOC"] += 1

        if q and gold_ids:
            for gold in gold_ids:
                pair = (q.lower(), gold)
                seen_pairs[pair] += 1
                if seen_pairs[pair] > 1:
                    row["status"] = ITEM_ERROR
                    row["error_type"] = "duplicate_id"
                    row["details"].append(f"duplicate (question, gold_chunk_id) pair: '{gold}'")
                    counts["DUPLICATE_IDS"] += 1

        if q and question_counts[q.lower()] > 1 and gold_ids:
            row["details"].append(f"question occurs {question_counts[q.lower()]}x in dataset")

        if row["status"] == "ok":
            counts["VALID_QA"] += 1
        rows.append(row)

    report = {"summary": counts, "rows": rows}
    return report


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Validate gold annotations against corpus chunks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--require-answer", action="store_true")
    args = parser.parse_args(argv)

    items = dataset_records(json.loads(Path(args.dataset).read_text(encoding="utf-8")))
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    report = validate_dataset(items, chunks, require_answer=args.require_answer)

    print(json.dumps(report["summary"], indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Saved] {out_path}")
    return report


def _exit_code(report: Dict[str, Any]) -> int:
    """Non-zero when any row is an evaluation_data_error.

    Without this the CLI always exited 0, so wiring it into CI produced a gate
    that could not fail no matter what the data contained.
    """
    invalid = report["summary"].get("TOTAL_QA", 0) - report["summary"].get("VALID_QA", 0)
    if invalid:
        logger.error("%s of %s rows are invalid", invalid, report["summary"]["TOTAL_QA"])
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_exit_code(main()))
