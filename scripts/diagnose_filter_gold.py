"""Fase A: diagnosis gold yang hilang di final filter (read-only).

Menghasilkan: results/debug/filter_gold_diagnosis.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src._compat  # noqa: F401
from src.evaluation.gold_matching import chunk_ids_match, gold_rank
from src.retrieval.chunk_filters import (
    FRAGMENT_STARTERS,
    classify_chunk,
    is_fragment_text,
    is_informative_chunk,
    is_low_quality_chunk,
    is_reference_chunk,
    is_title_or_author_chunk,
)

DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT_PATH = REPO_ROOT / "results" / "debug" / "filter_gold_diagnosis.json"


def trace_rule(chunk: dict) -> dict:
    text = (chunk.get("text") or "").strip()
    meta = chunk.get("metadata") or {}
    words = text.split()
    word_count = len(words)
    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(word_count, 1)
    starts_fragment = text.lower().startswith(FRAGMENT_STARTERS)
    ends_complete = text.endswith((".", "?", ";", ")"))
    has_author_pattern = bool(
        re.search(r"\b[A-Z]\.\s+[A-Z][a-z]+\b", text)
        or re.search(r"\bet al\.?\b", text)
        or re.search(r"[A-Z][a-z]+,\s+[A-Z][a-z]+\s+and\s+", text)
    )
    has_citation_cluster = len(re.findall(r"\[\d+\]", text)) > 3
    has_year_density = len(re.findall(r"\(\d{4}\)", text)) >= 3
    has_url = bool(re.search(r"https?://|doi:\s*10\.", text, re.IGNORECASE))
    section = str(meta.get("section", "")).lower()

    verdict = classify_chunk(chunk)
    reasons = []
    if verdict == "reference":
        if any(r in section for r in ("reference", "bibliography")):
            reasons.append(f"section '{section}' menandai reference")
        if has_citation_cluster:
            reasons.append("banyak citation [n]")
        if has_year_density:
            reasons.append("densitas tahun (YYYY) tinggi")
        if has_url:
            reasons.append("memuat URL/DOI")
    elif verdict == "title_author":
        if cap_ratio > 0.75 and word_count < 20:
            reasons.append("cap_ratio>0.75 & <20 kata")
        if has_author_pattern and word_count < 30:
            reasons.append("pola author line")
        if (
            re.search(r"\b(?:University|Institute|Department|Laboratory|Lab)\b", text)
            and word_count < 20
            and not text.endswith((".", "?"))
        ):
            reasons.append("baris afiliasi tanpa period akhir")
    elif verdict == "fragment":
        reasons.append("mulai dengan FRAGMENT_STARTERS (conjunction/relative)")
    elif verdict == "low_quality":
        if word_count < 10:
            reasons.append("word_count < 10")
        if text.endswith(":"):
            reasons.append("berakhir ':'")
        if text.count(",") > 10 and word_count < 15:
            reasons.append("koma overload & < 15 kata")
    elif verdict == "not_informative":
        if word_count < 12:
            reasons.append("word_count < 12")
        if not ends_complete:
            reasons.append("tidak berakhir .?;)")
        if starts_fragment:
            reasons.append("diawali conjunction/fragment starter")
        if cap_ratio > 0.7 and word_count < 15:
            reasons.append("cap_ratio > 0.7 dan < 15 kata")
        if has_author_pattern and word_count < 25:
            reasons.append("pola author line di heur informative")

    if not reasons and verdict != "ok":
        reasons.append(f"classify_chunk='{verdict}'")

    return {
        "first_reason": reasons[0] if reasons else "ok",
        "reasons": reasons,
        "diagnostics": {
            "section": section,
            "word_count": word_count,
            "char_count": len(text),
            "cap_ratio": round(cap_ratio, 3),
            "starts_fragment": starts_fragment,
            "ends_complete": ends_complete,
            "has_author_pattern": has_author_pattern,
            "has_citation_cluster": has_citation_cluster,
            "has_year_density": has_year_density,
            "has_url": has_url,
            "is_reference": is_reference_chunk(chunk),
            "is_title_author": is_title_or_author_chunk(chunk),
            "is_fragment": is_fragment_text(text),
            "is_low_quality": is_low_quality_chunk(chunk),
            "is_informative": is_informative_chunk(chunk),
            "classify_chunk": verdict,
        },
    }


def rank_of(gold_id: str, ids: list) -> int | None:
    for i, cid in enumerate(ids, start=1):
        if chunk_ids_match(gold_id, cid):
            return i
    return None


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunk_index = {c["chunk_id"]: c for c in chunks}

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)

    rows = []
    agg = {
        "total_queries": 0,
        "gold_missing_from_rrf": 0,
        "gold_missing_from_reranked": 0,
        "gold_missing_from_final": 0,
        "filter_failures": 0,
        "false_positive_filter": 0,
        "true_positive_filter": 0,
        "not_filter_failure": 0,
        "rejected_by_fragment": 0,
        "rejected_by_title_author": 0,
        "rejected_by_reference": 0,
        "rejected_by_low_quality": 0,
        "rejected_by_not_informative": 0,
    }

    for item in dataset:
        qid = item["query_id"]
        gold = item["gold_chunk_id"]
        out = retriever.search_with_trace(item["question"], top_k=10, rerank=True)
        trace = out["trace"]
        rrf_ids = trace.get("rrf_ids", [])
        reranked_ids = trace.get("reranked_ids", [])
        final_ids = trace.get("final_ids", [])

        in_rrf = rank_of(gold, rrf_ids) is not None
        in_reranked = rank_of(gold, reranked_ids) is not None
        in_final = rank_of(gold, final_ids) is not None

        gold_chunk = chunk_index.get(gold)
        trace_result = trace_rule(gold_chunk) if gold_chunk else None

        if not in_rrf:
            classification = "NOT_FILTER_FAILURE"
            rejected_by = None
            first_reason = "gold_missing_from_rrf"
            agg["gold_missing_from_rrf"] += 1
            agg["not_filter_failure"] += 1
        elif not in_reranked:
            classification = "NOT_FILTER_FAILURE"
            rejected_by = None
            first_reason = "gold_missing_from_reranked"
            agg["gold_missing_from_reranked"] += 1
            agg["not_filter_failure"] += 1
        elif not in_final:
            agg["gold_missing_from_final"] += 1
            agg["filter_failures"] += 1
            verdict = trace_result["diagnostics"]["classify_chunk"] if trace_result else None
            rejected_by = verdict
            first_reason = trace_result["first_reason"] if trace_result else None
            if verdict == "reference":
                agg["rejected_by_reference"] += 1
                classification = "TRUE_POSITIVE_FILTER"
                agg["true_positive_filter"] += 1
            elif verdict == "title_author":
                agg["rejected_by_title_author"] += 1
                classification = "FALSE_POSITIVE_FILTER"
                agg["false_positive_filter"] += 1
            elif verdict == "fragment":
                agg["rejected_by_fragment"] += 1
                classification = "FALSE_POSITIVE_FILTER"
                agg["false_positive_filter"] += 1
            elif verdict == "low_quality":
                agg["rejected_by_low_quality"] += 1
                classification = "FALSE_POSITIVE_FILTER"
                agg["false_positive_filter"] += 1
            elif verdict == "not_informative":
                agg["rejected_by_not_informative"] += 1
                # heuristik: kebanyakan false positive pada chunk pendek/punctuation
                classification = "FALSE_POSITIVE_FILTER"
                agg["false_positive_filter"] += 1
            else:
                classification = "UNKNOWN"
        else:
            classification = "VALID_AT_FINAL"
            rejected_by = None
            first_reason = None

        row = {
            "query_id": qid,
            "query": item["question"],
            "gold_chunk_id": gold,
            "gold_in_rrf": in_rrf,
            "gold_in_reranked": in_reranked,
            "gold_in_final": in_final,
            "rrf_rank": rank_of(gold, rrf_ids),
            "reranked_rank": rank_of(gold, reranked_ids),
            "final_rank": rank_of(gold, final_ids),
            "filter_verdict": (trace_result["diagnostics"]["classify_chunk"] if trace_result else None),
            "filter_reasons": (trace_result["reasons"] if trace_result else []),
            "first_reason": first_reason,
            "classification": classification,
            "evidence": {
                "section": (trace_result["diagnostics"]["section"] if trace_result else None),
                "word_count": (trace_result["diagnostics"]["word_count"] if trace_result else None),
                "char_count": (trace_result["diagnostics"]["char_count"] if trace_result else None),
                "cap_ratio": (trace_result["diagnostics"]["cap_ratio"] if trace_result else None),
                "ends_complete": (trace_result["diagnostics"]["ends_complete"] if trace_result else None),
                "starts_fragment": (trace_result["diagnostics"]["starts_fragment"] if trace_result else None),
                "text": (gold_chunk["text"][:200] if gold_chunk else None),
            },
        }
        rows.append(row)
        agg["total_queries"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"summary": agg, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    store.close()
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
