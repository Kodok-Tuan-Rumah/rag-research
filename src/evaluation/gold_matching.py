from __future__ import annotations

from collections.abc import Iterable

SUBCHUNK_MARKER = "_sub"


def normalize_chunk_id(chunk_id: object) -> str:
    if chunk_id is None:
        return ""
    return str(chunk_id).strip()


def is_subchunk_of(candidate_id: str, parent_id: str) -> bool:
    return candidate_id.startswith(parent_id + SUBCHUNK_MARKER)


def chunk_ids_match(gold_chunk_id: object, retrieved_chunk_id: object) -> bool:
    gold = normalize_chunk_id(gold_chunk_id)
    candidate = normalize_chunk_id(retrieved_chunk_id)
    if not gold or not candidate:
        return False
    if gold == candidate:
        return True
    return is_subchunk_of(candidate, gold) or is_subchunk_of(gold, candidate)


def is_gold_hit(gold_chunk_id: object, retrieved_ids: Iterable[object]) -> bool:
    return is_any_gold_hit([gold_chunk_id], retrieved_ids)


def gold_rank(gold_chunk_id: object, retrieved_ids: list[object]) -> int | None:
    return best_gold_rank([gold_chunk_id], retrieved_ids)


def is_any_gold_hit(gold_chunk_ids: Iterable[object], retrieved_ids: Iterable[object]) -> bool:
    gold_ids = list(gold_chunk_ids)
    retrieved = list(retrieved_ids)
    return any(
        chunk_ids_match(gold_id, retrieved_id) for gold_id in gold_ids for retrieved_id in retrieved
    )


def best_gold_rank(gold_chunk_ids: Iterable[object], retrieved_ids: list[object]) -> int | None:
    """Return the 1-based rank of the earliest retrieved match to any gold chunk."""
    gold_ids = list(gold_chunk_ids)
    for i, rid in enumerate(retrieved_ids):
        if any(chunk_ids_match(gold_id, rid) for gold_id in gold_ids):
            return i + 1
    return None


def doc_ids_match(gold_doc_id: object, retrieved_doc_id: object) -> bool:
    return normalize_chunk_id(gold_doc_id).lower() == normalize_chunk_id(
        retrieved_doc_id
    ).lower() and bool(normalize_chunk_id(gold_doc_id))
