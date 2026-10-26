"""Span-level attribution: map answer claims to evidence chunks."""
from __future__ import annotations

import re
from typing import Any


def split_into_claims(answer: str) -> list[str]:
    if not answer or answer.strip() == "":
        return []
    if "INSUFFICIENT_CONTEXT" in answer or "not sufficient" in answer.lower():
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', answer.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 3:
            continue
        s = re.sub(r'^[-*•]\s+', '', s)
        claims.append(s)
    return claims


def compute_claim_chunk_similarity(claim: str, chunk_text: str) -> float:
    claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
    chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
    
    if not claim_words or not chunk_words:
        return 0.0
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those', 'it', 'its'}
    claim_words = claim_words - stop_words
    chunk_words = chunk_words - stop_words
    
    if not claim_words:
        return 0.0
    
    intersection = claim_words.intersection(chunk_words)
    union = claim_words.union(chunk_words)
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def attribute_claims_to_chunks(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims = split_into_claims(answer)
    attributions = []
    
    if not claims or not chunks:
        return attributions
    
    for idx, claim in enumerate(claims):
        best_score = -1.0
        best_chunk = None
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '') or chunk.get('content', '')
            score = compute_claim_chunk_similarity(claim, chunk_text)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        if best_chunk:
            evidence_text = best_chunk.get('text', '')
            if len(evidence_text) > 200:
                evidence_text = evidence_text[:200] + "..."
            attributions.append({
                "claim_id": f"claim_{idx:03d}",
                "claim_text": claim,
                "chunk_id": best_chunk.get('chunk_id'),
                "doc_id": best_chunk.get('doc_id'),
                "score": round(best_score, 4),
                "evidence_text": evidence_text,
                "unverified": best_score <= 0.1
            })
    
    return attributions
