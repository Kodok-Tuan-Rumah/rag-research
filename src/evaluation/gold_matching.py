"""Canonical gold-chunk ↔ retrieved-chunk identity matching.

Satu-satunya sumber kebenaran untuk aturan kecocokan identitas chunk antara
gold label di dataset dan chunk_id yang dikembalikan retriever.
Jangan menduplikasi aturan matching ini di file lain.

Aturan eksplisit:
- exact match (string identik setelah strip)
- parent/subchunk relation pada pola ``<base>_sub<...>``:
    base chunk cocok dengan turunannya ``<base>_sub...`` dan sebaliknya.

Doc-level matching TIDAK dihitung sebagai hit chunk; gunakan fungsi doc-level
secara terpisah bila diperlukan.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

SUBCHUNK_MARKER = "_sub"


def normalize_chunk_id(chunk_id: object) -> str:
    if chunk_id is None:
        return ""
    return str(chunk_id).strip()


def is_subchunk_of(candidate_id: str, parent_id: str) -> bool:
    """True jika candidate adalah subchunk dari parent (``parent_sub...``)."""
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


def gold_rank(gold_chunk_id: object, retrieved_ids: List[object]) -> Optional[int]:
    """Rank 1-based posisi pertama gold chunk muncul; None jika tidak ketemu."""
    return best_gold_rank([gold_chunk_id], retrieved_ids)


def is_any_gold_hit(gold_chunk_ids: Iterable[object], retrieved_ids: Iterable[object]) -> bool:
    """True when any gold chunk matches any retrieved chunk."""
    gold_ids = list(gold_chunk_ids)
    retrieved = list(retrieved_ids)
    return any(
        chunk_ids_match(gold_id, retrieved_id)
        for gold_id in gold_ids
        for retrieved_id in retrieved
    )


def best_gold_rank(gold_chunk_ids: Iterable[object], retrieved_ids: List[object]) -> Optional[int]:
    """Return the 1-based rank of the earliest retrieved match to any gold chunk."""
    gold_ids = list(gold_chunk_ids)
    for i, rid in enumerate(retrieved_ids):
        if any(chunk_ids_match(gold_id, rid) for gold_id in gold_ids):
            return i + 1
    return None


def doc_ids_match(gold_doc_id: object, retrieved_doc_id: object) -> bool:
    return normalize_chunk_id(gold_doc_id).lower() == normalize_chunk_id(retrieved_doc_id).lower() and bool(normalize_chunk_id(gold_doc_id))
