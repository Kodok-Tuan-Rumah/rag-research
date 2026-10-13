"""Query-leakage gate — PROJECT_GOALS.md §10.2.

    For each (query, chunk):
        compute similarity(query, chunk_text)
    If similarity > 0.9 -> sample MUST be excluded
    Prevents trivial retrieval where query is approximately the chunk text.

The failure this prevents is not a scoring bug, it is a measurement bug. A
question that restates its gold chunk needs no ranking skill at all: returning
that one chunk scores Recall@10 = 1.0. An eval set containing such samples
reports perfect retrieval for a system that cannot retrieve, and every
downstream number — the Phase 2 metrics, the Phase 3 calibration pairs, the
production checklist — inherits that.

Similarity is computed with the same embedding model the retriever uses
(`settings.embed_model`), not a lexical overlap heuristic. A token-overlap
check would miss the semantic near-duplicates that are the actual hazard, and
would pass paraphrases that a dense retriever resolves instantly.

The embedder is injected rather than constructed here. Loading it costs
seconds and needs the model on disk; the arithmetic must be testable without
either.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.api.config import settings
from src.evaluation.gold_schema import dataset_records, is_abstain_row, normalize_gold_chunk_ids

# PROJECT_GOALS.md §10.2 states the threshold as "similarity > 0.9".
LEAKAGE_THRESHOLD = 0.9

Embedder = Callable[[Sequence[str]], Any]


def default_embedder() -> Embedder:
    """The retriever's own encoder, so the check matches retrieval."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)

    def encode(texts: Sequence[str]):
        return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    return encode


def _normalise(vectors: Any) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return arr.reshape(0, 0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # A zero row cannot be compared to anything; scale it to zero similarity
    # rather than dividing by zero and producing NaN that compares False.
    safe = np.where(norms == 0.0, 1.0, norms)
    return arr / safe


def cosine_similarities(
    queries: Sequence[str],
    passages: Sequence[str],
    embedder: Embedder | None = None,
) -> list[float]:
    """Row-wise cosine similarity between parallel query and passage lists."""
    if len(queries) != len(passages):
        raise ValueError(
            f"queries and passages must be parallel: {len(queries)} != {len(passages)}"
        )
    if not queries:
        return []
    embed = embedder or default_embedder()
    q = _normalise(embed(list(queries)))
    p = _normalise(embed(list(passages)))
    if q.shape[1] != p.shape[1]:
        raise ValueError(f"embedding width mismatch: {q.shape[1]} != {p.shape[1]}")
    return [float(v) for v in (q * p).sum(axis=1)]


def find_leaks(
    records: Iterable[Mapping[str, Any]],
    chunk_lookup: Mapping[str, Mapping[str, Any]],
    *,
    embedder: Embedder | None = None,
    threshold: float = LEAKAGE_THRESHOLD,
    get_question: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Score every scorable row and report the ones above `threshold`.

    Rows without gold are skipped rather than scored as 1.0: an abstain row
    declares no supporting chunk, and an unresolvable gold id is a labelling
    error that `validate_dataset` already reports. Neither is evidence of
    leakage, and counting them would bury the rows that matter.
    """
    items = list(records)
    scored: list[dict[str, Any]] = []
    skipped = 0

    for index, row in enumerate(items):
        question = (
            get_question(row) if get_question else _question_of(row)
        )
        if not question:
            skipped += 1
            continue
        if is_abstain_row(row):
            skipped += 1
            continue
        try:
            gold_ids = normalize_gold_chunk_ids(row)
        except ValueError:
            # Malformed gold is check_datasets' failure, not leakage.
            skipped += 1
            continue
        chunk = next(
            (chunk_lookup[c] for c in gold_ids if c in chunk_lookup), None
        )
        if chunk is None:
            skipped += 1
            continue
        scored.append({
            "_index": index,
            "query_id": row.get("query_id", f"index {index}"),
            "question": question,
            "gold_chunk_id": chunk.get("chunk_id"),
            "_passage": chunk.get("text", ""),
        })

    if not scored:
        return {
            "threshold": threshold,
            "embed_model": settings.embed_model,
            "total_rows": len(items),
            "scored": 0,
            "skipped": skipped,
            "leaked": 0,
            "max_similarity": None,
            "detail": [],
        }

    embed = embedder or default_embedder()
    sims = cosine_similarities(
        [s["question"] for s in scored],
        [s["_passage"] for s in scored],
        embedder=embed,
    )

    detail = []
    for entry, sim in zip(scored, sims):
        record = {k: v for k, v in entry.items() if not k.startswith("_")}
        record["similarity"] = round(sim, 4)
        record["leaked"] = sim > threshold
        detail.append(record)

    leaks = [d for d in detail if d["leaked"]]
    return {
        "threshold": threshold,
        "embed_model": settings.embed_model,
        "total_rows": len(items),
        "scored": len(detail),
        "skipped": skipped,
        "leaked": len(leaks),
        "max_similarity": round(max(sims), 4),
        "detail": sorted(detail, key=lambda d: -d["similarity"]),
    }


def _question_of(row: Mapping[str, Any]) -> str | None:
    for key in ("question", "query"):
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def dataset_leakage(
    dataset: Any,
    chunks: list[Mapping[str, Any]],
    *,
    embedder: Embedder | None = None,
    threshold: float = LEAKAGE_THRESHOLD,
) -> dict[str, Any]:
    """Convenience wrapper over `find_leaks` for a raw dataset payload."""
    records = dataset_records(dataset)
    lookup = {c["chunk_id"]: c for c in chunks if c.get("chunk_id")}
    report = find_leaks(
        records, lookup, embedder=embedder, threshold=threshold
    )
    return report