"""Shared chunk-quality filters.

Satu sumber kebenaran untuk heuristik noise chunk yang sebelumnya
duplikasi di hybrid_retriever.py, qdrant_store.py, dan semantic_chunker.py.

Dua tingkat filter:
- STRICT : dipakai di jalur utama retrieval (informative + bukan ref/title/author)
- MINIMUM: dipakai di fallback — hanya buang noise keras (fragment, title/author,
  reference), panjang kalimat boleh dilonggarkan.

Contract semantik (jangan gunakan satu boolean "valid" untuk semua tujuan):
1. ``is_retrieval_candidate`` / ``is_low_quality_chunk`` — gate murah pra-rerank.
2. ``is_informative_chunk`` — kelengkapan & isi kalimat.
3. ``is_reference_chunk`` + ``reference_policy_allows`` — kebijakan references
   bergantung pada query type (factual/citation vs reasoning).
4. ``is_generation_eligible`` — eligibility masuk context LLM.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("rag.chunk_filters")

FRAGMENT_STARTERS = (
    "and ", "or ", "but ", "so ",
    "which ", "that ", "who ", "whose ", "where ",
)

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return chunk.get("text", "") or ""
    return getattr(chunk, "text", "") or ""


def get_metadata(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return chunk.get("metadata", {}) or {}
    return getattr(chunk, "metadata", {}) or {}


# ---------------------------------------------------------------------------
# Komponen heuristik
# ---------------------------------------------------------------------------

def is_fragment_text(text: str) -> bool:
    """Chunk diawali conjunction/relatif lowercase → potongan kalimat."""
    return text.strip().lower().startswith(FRAGMENT_STARTERS)


# ---------------------------------------------------------------------------
# Kandidat retrieval — dipakai sebagai gerbang awal sebelum rerank
# ---------------------------------------------------------------------------

def is_retrieval_candidate(chunk: Any) -> bool:
    """Gate murah sebelum rerank: buang noise obvious saja."""
    return not is_low_quality_chunk(chunk)


def is_fragment_chunk(chunk: Any) -> bool:
    return is_fragment_text(get_text(chunk))


def is_reference_chunk(chunk: Any) -> bool:
    meta = get_metadata(chunk)
    text = get_text(chunk)

    section = str(meta.get("section", "")).lower()
    if any(ref in section for ref in ["reference", "bibliography"]):
        return True

    # Bracket citation yang banyak — body text biasanya ≤2
    if len(re.findall(r"\[\d+\]", text)) > 3:
        return True

    # URL atau DOI
    if re.search(r"https?://|doi:\s*10\.", text, re.IGNORECASE):
        return True

    # Dense citation: banyak tahun dalam kurung "(2020)"
    if len(re.findall(r"\(\d{4}\)", text)) >= 3:
        return True

    # Comma overload (heuristic lama, tetap dipertahankan)
    if text.count(",") > 10 and len(text.split()) < 30:
        return True

    return False


def is_title_or_author_chunk(chunk: Any) -> bool:
    """Deteksi chunk yang isinya judul paper atau baris author/afiliasi."""
    text = get_text(chunk).strip()
    if not text:
        return False

    words = text.split()

    # Judul paper: mayoritas Title Case, pendek (< 20 kata)
    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(len(words), 1)
    if cap_ratio > 0.75 and len(words) < 20:
        logger.debug(f"[TITLE_FILTER] title-like chunk filtered: '{text[:80]}'")
        return True

    # Author line: inisial "A. Name", "et al.", atau "Name, Name and Name"
    has_author_pattern = bool(
        re.search(r'\b[A-Z]\.\s+[A-Z][a-z]+\b', text) or
        re.search(r'\bet al\.?\b', text) or
        re.search(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+and\s+', text)
    )
    if has_author_pattern and len(words) < 30:
        logger.debug(f"[AUTHOR_FILTER] author-line chunk filtered: '{text[:80]}'")
        return True

    # Afiliasi institusi tanpa kalimat substantif
    has_affiliation = bool(
        re.search(r'\b(?:University|Institute|Department|Laboratory|Lab)\b', text)
    )
    if has_affiliation and len(words) < 20 and not text.endswith(('.', '?')):
        logger.debug(f"[AFFIL_FILTER] affiliation chunk filtered: '{text[:80]}'")
        return True

    return False


def is_low_quality_chunk(chunk: Any) -> bool:
    """Filter murah sebelum rerank (tahap 1 retrieval)."""
    text = get_text(chunk).strip()
    if len(text.split()) < 10:
        return True
    if text.endswith(":"):
        return True
    if text.count(",") > 10 and len(text.split()) < 15:
        return True
    return False

DANGLING_TAIL = {"and", "but", "because", "since", "while", "although", "if"}
VERB_SIGNALS = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "will",
}

def _ends_with_full_punctuation(text: str) -> bool:
    return text.endswith((".", "?", ";", ")"))

def _looks_substantively_complete(text: str) -> bool:
    """TRUE bila kalimat substantive meski tanda baca akhir {".?"} hilang."""
    if _ends_with_full_punctuation(text):
        return True
    words = text.split()
    if len(words) < 12:
        return False
    if text.endswith((",", ":", "::", ";")):
        return False
    last = words[-1].strip(".,;:!?)]}\"'`").lower()
    if last in DANGLING_TAIL:
        return False
    tokens = {w.strip(".,;:!?)]}\"'").lower() for w in words}
    return bool(tokens & VERB_SIGNALS)


def is_informative_chunk(chunk: Any) -> bool:
    """Filter tahap 2: informativeness.

    Contract updated (Fase B): keterputusan kalimat karena chunk boundary
    TIDAK otomatis menolak chunk; syarat lengkapnya cukup
    `_looks_substantively_complete`.
    """
    text_stripped = get_text(chunk).strip()
    words = text_stripped.split()
    word_count = len(words)

    starts_complete = not text_stripped.lower().startswith(("and ", "or ", "but "))
    ends_complete = _looks_substantively_complete(text_stripped)

    starts_fragment = text_stripped.lower().startswith(FRAGMENT_STARTERS)

    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(word_count, 1)
    is_title_like = cap_ratio > 0.7 and word_count < 15

    is_author_line = (
        bool(re.search(
            r'\b[A-Z]\.\s+[A-Z][a-z]+\b|\bet al\.?\b|[A-Z][a-z]+,\s+[A-Z][a-z]+\s+and\s+',
            text_stripped,
        ))
        and word_count < 25
    )

    return (
        word_count >= 12
        and starts_complete
        and ends_complete
        and not starts_fragment
        and not is_title_like
        and not is_author_line
    )


# ---------------------------------------------------------------------------
# Gabungan filter
# ---------------------------------------------------------------------------

def passes_strict_filter(chunk: Any, *, allow_references: bool = False) -> bool:
    """Jalur utama: semua syarat harus terpenuhi."""
    return (
        is_informative_chunk(chunk)
        and (allow_references or not is_reference_chunk(chunk))
        and not is_title_or_author_chunk(chunk)
    )


def passes_minimum_filter(chunk: Any, *, allow_references: bool = False) -> bool:
    """Fallback: hanya buang noise keras. Panjang/kelengkapan dilonggarkan."""
    text = get_text(chunk).strip()
    if not text:
        return False
    if not allow_references and is_reference_chunk(chunk):
        return False
    if is_title_or_author_chunk(chunk):
        return False
    if is_fragment_text(text):
        return False
    return True


def classify_chunk(chunk: Any, *, allow_references: bool = False) -> str:
    """Alasan pertama sebuah chunk ditolak (urutan prioritas), atau 'ok'."""
    if not allow_references and is_reference_chunk(chunk):
        return "reference"
    if is_title_or_author_chunk(chunk):
        return "title_author"
    if is_fragment_text(get_text(chunk)):
        return "fragment"
    if is_low_quality_chunk(chunk):
        return "low_quality"
    if not is_informative_chunk(chunk):
        return "not_informative"
    return "ok"


def filter_breakdown(chunks: List[Any], *, allow_references: bool = False) -> Dict[str, int]:
    """Hitung distribusi alasan penolakan — untuk logging terstruktur."""
    counts: Dict[str, int] = {
        "ok": 0, "reference": 0, "title_author": 0,
        "fragment": 0, "low_quality": 0, "not_informative": 0,
    }
    for c in chunks:
        counts[classify_chunk(c, allow_references=allow_references)] += 1
    return counts


def passes_minimum_filter_text(text: str) -> bool:
    return passes_minimum_filter({"text": text, "metadata": {}})


def reference_policy_allows(section: str, allow_references: bool) -> bool:
    """Reference hanya boleh lolos bila query memang butuh citation."""
    section = str(section or "").lower()
    if section in ("references", "bibliography"):
        return bool(allow_references)
    return True


def get_chunk_section(chunk: Any) -> str:
    return str(get_metadata(chunk).get("section", "") or "")


def is_generation_eligible(chunk: Any, *, allow_references: bool = False) -> bool:
    """Chunk layak masuk context generation.

    - informatif (panjang, bukan title/author, kalimat lengkap)
    - bukan fragment
    - bukan references/bibliography bila query tidak meminta citation
    """
    if not is_informative_chunk(chunk):
        return False
    if not reference_policy_allows(get_chunk_section(chunk), allow_references):
        return False
    return True


def is_noise_chunk_text(text: str) -> bool:
    """
    Versi teks-murni untuk ingestion (qdrant_store / semantic_chunker).
    Aturan sama dengan versi sebelumnya yang diduplikasi:
    noise = judul / author / afiliasi / reference / fragment / terlalu pendek.
    """
    text = text.strip()
    if not text:
        return True

    words = text.split()
    word_count = len(words)

    if word_count < 12:
        return True

    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(word_count, 1)
    if cap_ratio > 0.75 and word_count < 20:
        return True

    if word_count < 30 and bool(
        re.search(r'\b[A-Z]\.\s+[A-Z][a-z]+\b', text) or
        re.search(r'\bet al\.?\b', text) or
        re.search(r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+and\s+', text)
    ):
        return True

    if (
        word_count < 20
        and not text.endswith(('.', '?'))
        and re.search(r'\b(?:University|Institute|Department|Laboratory|Lab)\b', text)
    ):
        return True

    if len(re.findall(r'\[\d+\]', text)) > 3:
        return True
    if re.search(r'https?://|doi:\s*10\.', text, re.IGNORECASE):
        return True
    if len(re.findall(r'\(\d{4}\)', text)) >= 3:
        return True
    if text.count(',') > 10 and word_count < 30:
        return True

    if text.lower().startswith(FRAGMENT_STARTERS):
        return True

    if not text.endswith(('.', '?', ';', ')')) and word_count < 20:
        return True

    return False
