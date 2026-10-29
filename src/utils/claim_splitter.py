"""Deterministic, non-LLM claim splitter for manuscript text.

Splits free-form text into atomic claim spans using punctuation boundaries.
Deterministic: identical input always yields identical output.
"""
from __future__ import annotations

import re
from typing import List


# Sentence/claim terminators. A claim ends at . ! ? followed by whitespace or EOS.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str) -> List[str]:
    """Split text into deterministic claim spans.

    Returns a list of non-empty trimmed strings. Empty/None input returns [].
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    # Normalize internal whitespace to keep output deterministic.
    text = re.sub(r"\s+", " ", text)
    parts = _SENTENCE_RE.split(text)
    claims: List[str] = []
    for part in parts:
        part = part.strip()
        if part:
            claims.append(part)
    # If no terminator was present, the whole text is one claim.
    if not claims and text:
        claims.append(text)
    return claims
