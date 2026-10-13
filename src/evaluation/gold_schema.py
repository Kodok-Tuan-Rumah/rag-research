"""Normalize legacy and canonical gold chunk fields to one internal schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_UNSET = object()


def is_abstain_row(item: Mapping[str, Any]) -> bool:
    """Return True only for a row explicitly labelled as an abstain case.

    The check is identity against ``True``, never truthiness. Datasets in this
    repo disagree about the type of this field: retrieval_eval_dataset_v2.json
    stores the string ``"UNKNOWN"``, while v1/v3/v4 store real booleans. A
    truthiness test would treat every "UNKNOWN" row as an abstain case and
    silently drop v2's 13 rows out of validation — an abstain row is the one
    row type that has no gold chunk, so exempting it is exactly the mistake
    that would go unnoticed.

    An abstain row is not a defect. PROJECT_GOALS.md §7 requires a
    `should_abstain` label and §8 requires abstain accuracy over labelled data,
    so the contract has to be able to express "this question must not be
    answered from the corpus". Such a row has no supporting chunk and no gold
    answer by definition.
    """
    return item.get("should_abstain") is True


def dataset_records(payload: object) -> list[dict[str, Any]]:
    """Accept a legacy list or a versioned dataset envelope."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = payload.get("records")
    else:
        records = None
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("dataset must be a record array or an object with a records array")
    return records


def normalize_gold_chunk_ids(item: Mapping[str, Any]) -> list[str]:
    """Return gold chunk IDs, rejecting ambiguous or malformed dataset rows.

    An abstain row yields an empty list. Every other row must name its gold
    chunk exactly once; an empty result therefore always means "no gold is
    declared", which callers must treat as unscorable rather than as a hit.
    """
    if is_abstain_row(item):
        return []

    has_legacy = "gold_chunk_id" in item
    has_canonical = "gold_chunk_ids" in item

    if has_legacy == has_canonical:
        if has_legacy:
            raise ValueError("provide exactly one of gold_chunk_id or gold_chunk_ids")
        raise ValueError("missing gold_chunk_id and gold_chunk_ids")

    if has_legacy:
        value = item["gold_chunk_id"]
        if not isinstance(value, str):
            raise ValueError("gold_chunk_id must be a non-empty string")
        gold_ids = [value]
    else:
        value = item["gold_chunk_ids"]
        if not isinstance(value, list):
            raise ValueError("gold_chunk_ids must be a non-empty array of strings")
        if not value:
            raise ValueError("gold_chunk_ids must not be empty")
        if any(not isinstance(chunk_id, str) for chunk_id in value):
            raise ValueError("every gold_chunk_ids element must be a non-empty string")
        gold_ids = value

    normalized = [chunk_id.strip() for chunk_id in gold_ids]
    if any(not chunk_id for chunk_id in normalized):
        field = "gold_chunk_id" if has_legacy else "gold_chunk_ids"
        raise ValueError(f"{field} must contain non-empty chunk IDs")
    if len(normalized) != len(set(normalized)):
        raise ValueError("gold chunk IDs must be unique")
    return normalized
