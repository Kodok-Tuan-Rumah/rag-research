import pytest

from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"gold_chunk_id": " A "}, ["A"]),
        ({"gold_chunk_ids": ["A", "B"]}, ["A", "B"]),
    ],
)
def test_normalize_legacy_and_canonical_gold_ids(row, expected):
    assert normalize_gold_chunk_ids(row) == expected


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({}, "missing gold_chunk_id"),
        ({"gold_chunk_id": None}, "non-empty string"),
        ({"gold_chunk_ids": None}, "array of strings"),
        ({"gold_chunk_ids": []}, "must not be empty"),
        ({"gold_chunk_ids": "A"}, "array of strings"),
        ({"gold_chunk_ids": ["A", None]}, "element must be"),
        ({"gold_chunk_ids": ["A", "A"]}, "unique"),
        ({"gold_chunk_id": ["A"]}, "non-empty string"),
        ({"gold_chunk_id": "A", "gold_chunk_ids": ["A"]}, "exactly one"),
    ],
)
def test_invalid_gold_schema_has_explicit_error(row, message):
    with pytest.raises(ValueError, match=message):
        normalize_gold_chunk_ids(row)


def test_dataset_records_accepts_legacy_list_and_versioned_envelope():
    records = [{"query_id": "q01"}]
    assert dataset_records(records) == records
    assert dataset_records({"dataset_version": "v2", "records": records}) == records


def test_dataset_records_rejects_invalid_envelope():
    with pytest.raises(ValueError, match="records array"):
        dataset_records({"dataset_version": "v2", "records": {}})
