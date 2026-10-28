"""Normalize legacy and canonical gold chunk fields to one internal schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_UNSET = object()


def is_abstain_row(item: Mapping[str, Any]) -> bool:
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
