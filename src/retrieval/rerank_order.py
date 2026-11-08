from __future__ import annotations

from typing import Any


def rank_reranked_chunks(chunks: list[Any]) -> list[Any]:
    """Order candidates by cross-encoder relevance, not section-adjusted score."""

    def score(chunk: Any) -> float:
        if isinstance(chunk, dict):
            value = chunk.get("rerank_score")
            if value is None:
                value = chunk.get("score")
        else:
            value = getattr(chunk, "rerank_score", None)
            if value is None:
                value = getattr(chunk, "score", None)
        return float(value if value is not None else 0.0)

    return sorted(chunks, key=score, reverse=True)
