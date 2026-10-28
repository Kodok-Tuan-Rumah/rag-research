"""BM25-anchored fusion: keep BM25's top hits in BM25 order.

The anchor is opt-in (`bm25_anchor_k=0` disables it) so the production default
stays unchanged until a larger eval justifies it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_retriever import MasterHybridRetriever

BODY = (
    "Hybrid search combines dense vector retrieval with sparse keyword "
    "retrieval using reciprocal rank fusion. This sentence is complete."
)


def _hit(chunk_id, score=1.0, suffix=""):
    return {
        "chunk_id": chunk_id,
        "text": f"{BODY} Variant {suffix or chunk_id}.",
        "doc_id": "doc",
        "score": score,
        "metadata": {"section": "Introduction"},
    }


def _retriever(monkeypatch, dense_hits, bm25_hits):
    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.rrf_k = 60
    retriever.reranker = None
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            allow_references=False,
            max_ref_ratio=0.0,
            top_k_hint=10,
        )
    )
    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 40)
    monkeypatch.setattr(
        retriever,
        "_expand_and_hyde_parallel",
        lambda query, use_mq=False, use_hyde=False: ([query], None),
    )
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: list(dense_hits))
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: list(bm25_hits))
    return retriever


def test_bm25_anchor_puts_bm25_top_hit_first(monkeypatch):
    dense = [_hit("d1", 0.9), _hit("d2", 0.8)]
    bm25 = [_hit("b1", 5.0), _hit("b2", 4.0)]
    retriever = _retriever(monkeypatch, dense, bm25)

    result = retriever.search_with_trace(
        "how does hybrid search work", top_k=10, bm25_anchor_k=1
    )

    assert result["results"][0]["chunk_id"] == "b1"
    assert result["trace"]["bm25_anchor_k"] == 1
    assert result["trace"]["bm25_anchor_ids"] == ["b1"]


def test_bm25_anchor_keeps_multiple_hits_in_bm25_order(monkeypatch):
    dense = [_hit("d1", 0.9), _hit("d2", 0.8)]
    bm25 = [_hit("b1", 5.0), _hit("b2", 4.0)]
    retriever = _retriever(monkeypatch, dense, bm25)

    result = retriever.search_with_trace(
        "how does hybrid search work", top_k=10, bm25_anchor_k=2
    )

    ids = [chunk["chunk_id"] for chunk in result["results"]]
    assert ids[:2] == ["b1", "b2"]
    assert len(ids) == len(set(ids))


def test_anchor_disabled_preserves_rrf_order(monkeypatch):
    dense = [_hit("d1", 0.9), _hit("d2", 0.8)]
    bm25 = [_hit("b1", 5.0), _hit("b2", 4.0)]
    retriever = _retriever(monkeypatch, dense, bm25)

    result = retriever.search_with_trace(
        "how does hybrid search work", top_k=10, bm25_anchor_k=0
    )

    assert result["results"][0]["chunk_id"] == "d1"
    assert result["trace"]["bm25_anchor_ids"] == []


def test_anchor_skips_hits_absent_from_the_fused_pool(monkeypatch):
    dense = [_hit("d1", 0.9)]
    missing = _hit("missing", 5.0)
    missing["text"] = "short"
    bm25 = [missing, _hit("b2", 4.0)]
    retriever = _retriever(monkeypatch, dense, bm25)

    result = retriever.search_with_trace(
        "how does hybrid search work", top_k=10, bm25_anchor_k=2
    )

    assert result["trace"]["bm25_anchor_ids"] == ["b2"]
    assert result["results"][0]["chunk_id"] == "b2"
