"""Span-level attribution of answer claims to retrieved chunks."""
from __future__ import annotations

import re
from typing import Any


def split_into_claims(answer: str) -> list[str]:
    if not answer:
        return []
    # Split on sentence boundaries.
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def compute_claim_chunk_similarity(claim: str, chunk_text: str) -> float:
    if not claim or not chunk_text:
        return 0.0
    claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
    chunk_words = set(re.findall(r"[a-z0-9]+", chunk_text.lower()))
    if not claim_words or not chunk_words:
        return 0.0
    intersection = claim_words & chunk_words
    union = claim_words | chunk_words
    if not union:
        return 0.0
    return len(intersection) / len(union)


def attribute_claims_to_chunks(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims = split_into_claims(answer)
    attributions: list[dict[str, Any]] = []
    if not claims or not chunks:
        return attributions

    for idx, claim in enumerate(claims):
        best_score = -1.0
        best_chunk = None
        for chunk in chunks:
            chunk_text = chunk.get("text", "") or chunk.get("content", "")
            score = compute_claim_chunk_similarity(claim, chunk_text)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is None:
            continue
        evidence_text = best_chunk.get("text", "") or best_chunk.get("content", "")
        if len(evidence_text) > 200:
            evidence_text = evidence_text[:200] + "..."
        attributions.append({
            "claim_span": claim,
            "chunk_id": best_chunk.get("chunk_id"),
            "doc_id": best_chunk.get("doc_id"),
            "score": round(float(best_score), 4) if best_score >= 0 else 0.0,
            "text": evidence_text,
            "unverified": bool(best_score <= 0.1),
        })

    return attributions
