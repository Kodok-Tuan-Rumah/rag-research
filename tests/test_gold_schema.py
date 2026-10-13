import pytest

from src.evaluation.gold_schema import (
    dataset_records,
    is_abstain_row,
    normalize_gold_chunk_ids,
)


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


# --- abstain rows: the one row type that legitimately has no gold chunk -----
#
# PROJECT_GOALS.md §7 requires a should_abstain label and §8 requires abstain
# accuracy over labelled data, so the gold contract has to express "this
# question is not answerable from the corpus". normalize_gold_chunk_ids()
# returned an empty string error for exactly those rows, which made an abstain
# row impossible to ship.


def test_abstain_row_normalizes_to_no_gold_instead_of_raising():
    assert normalize_gold_chunk_ids(
        {"query_id": "adv_000", "should_abstain": True, "gold_chunk_id": ""}
    ) == []


def test_abstain_row_wins_over_a_present_but_empty_gold_list():
    assert normalize_gold_chunk_ids(
        {"should_abstain": True, "gold_chunk_ids": []}
    ) == []


@pytest.mark.parametrize(
    "row",
    [
        {"should_abstain": False, "gold_chunk_id": "c1"},
        {"should_abstain": "UNKNOWN", "gold_chunk_id": "c1"},
        {"should_abstain": 1, "gold_chunk_id": "c1"},
        {"should_abstain": "true", "gold_chunk_id": "c1"},
        {"gold_chunk_id": "c1"},
    ],
)
def test_only_a_literal_true_marks_an_abstain_row(row):
    """The regression this guards is silent, so it is pinned explicitly.

    retrieval_eval_dataset_v2.json stores should_abstain as the string
    "UNKNOWN". A truthiness test would read those 13 rows as abstain cases and
    exempt them from validation — dropping a third of the labeled corpus out of
    every metric while reporting ok. Integer 1 and the string "true" are
    rejected for the same reason: only True is a decision, everything else is
    data this contract does not understand.
    """
    assert is_abstain_row(row) is False
    assert normalize_gold_chunk_ids(row) == ["c1"]


def test_non_abstain_row_with_empty_gold_is_still_rejected():
    """The exemption must not become a blanket escape hatch."""
    with pytest.raises(ValueError, match="non-empty"):
        normalize_gold_chunk_ids({"should_abstain": False, "gold_chunk_id": ""})
