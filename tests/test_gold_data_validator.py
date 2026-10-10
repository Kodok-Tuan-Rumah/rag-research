import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gold_data_validator import validate_dataset

CHUNKS = [
    {"chunk_id": "doc_a_rs_0001", "doc_id": "doc_a", "text": "hello world."},
    {"chunk_id": "doc_b_rs_0002", "doc_id": "doc_b", "text": "another chunk."},
]


def _item(question="Q?", gold="doc_a_rs_0001", doc_id=None, answer="ans"):
    item = {"question": question, "gold_chunk_id": gold, "gold_answer": answer}
    if doc_id is not None:
        item["doc_id"] = doc_id
    return item


def test_valid_dataset_counts():
    report = validate_dataset([_item(), _item(question="Q2?", gold="doc_b_rs_0002")], CHUNKS)
    s = report["summary"]
    assert s["TOTAL_QA"] == 2 and s["VALID_QA"] == 2
    assert s["MISSING_GOLD_CHUNK"] == 0 and s["INVALID_MAPPING"] == 0


def test_missing_gold_chunk_is_flagged():
    report = validate_dataset([_item(gold="missing_rs_9999")], CHUNKS)
    s = report["summary"]
    assert s["VALID_QA"] == 0 and s["MISSING_GOLD_CHUNK"] == 1
    assert report["rows"][0]["error_type"] == "missing_gold_chunk"


def test_doc_mismatch_is_invalid_mapping():
    report = validate_dataset([_item(gold="doc_a_rs_0001", doc_id="doc_b")], CHUNKS)
    s = report["summary"]
    assert s["INVALID_MAPPING"] == 1 and s["VALID_QA"] == 0


def test_duplicate_pairs_detected():
    items = [_item(), _item()]
    report = validate_dataset(items, CHUNKS)
    assert report["summary"]["DUPLICATE_IDS"] == 1


def test_missing_question_detected():
    item = _item()
    del item["question"]
    report = validate_dataset([item], CHUNKS)
    assert report["summary"]["MISSING_QUESTION"] == 1


def test_missing_answer_only_when_required():
    item = _item(answer="")
    assert validate_dataset([item], CHUNKS)["summary"]["MISSING_ANSWER"] == 0
    assert validate_dataset([item], CHUNKS, require_answer=True)["summary"]["MISSING_ANSWER"] == 1


def test_canonical_multi_gold_dataset_is_validated():
    item = {
        "question": "Q?",
        "ground_truth": "ans",
        "gold_doc_id": "doc_a",
        "gold_chunk_ids": ["doc_a_rs_0001", "doc_a_rs_0001_sub2"],
    }
    chunks = CHUNKS + [
        {"chunk_id": "doc_a_rs_0001_sub2", "doc_id": "doc_a", "text": "subchunk"}
    ]
    report = validate_dataset({"dataset_version": "v2", "records": [item]}, chunks, require_answer=True)
    assert report["summary"]["VALID_QA"] == 1
    assert report["rows"][0]["gold_chunk_ids"] == item["gold_chunk_ids"]


def test_invalid_canonical_gold_schema_is_reported():
    report = validate_dataset(
        [{"question": "Q?", "gold_chunk_ids": []}],
        CHUNKS,
    )
    assert report["summary"]["VALID_QA"] == 0
    assert report["rows"][0]["error_type"] == "invalid_gold_chunk_ids"


def test_legacy_validator_chunk_alias_remains_supported():
    item = {"question": "Q?", "source_chunk": "doc_a_rs_0001"}
    report = validate_dataset([item], CHUNKS)
    assert report["summary"]["VALID_QA"] == 1
    assert report["rows"][0]["gold_chunk_id"] == "doc_a_rs_0001"
