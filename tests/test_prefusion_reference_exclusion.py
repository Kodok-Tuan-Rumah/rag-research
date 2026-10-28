"""References must not occupy fusion slots when the query forbids them.

Raw dense/BM25 hits stay untouched in the trace so per-stage metrics remain
honest; only the lists handed to RRF are pruned.
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
BODY_2 = (
    "Reciprocal rank fusion merges ranked lists from dense and sparse "
    "retrievers into a single ranking. This sentence is also complete."
)
REFERENCE = (
    'References [1] A. Author, "Retrieval augmented generation," arXiv '
    "preprint arXiv:2312.05708, 2023. [2] B. Author, Neural information "
    "processing systems, 2022. [3] C. Author, International conference on "
    "learning representations, 2021. [4] D. Author, Association for "
    "computational linguistics, 2020."
)


def _hit(chunk_id, text, section):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "doc_id": "doc",
        "score": 1.0,
        "metadata": {"section": section},
    }


def _retriever(monkeypatch, dense_hits, bm25_hits):
    from src.retrieval import hybrid_retriever

    monkeypatch.setattr(
        hybrid_retriever.settings, "prefusion_reference_exclusion", True
    )
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


def test_references_are_removed_before_fusion(monkeypatch):
    reference = _hit("ref", REFERENCE, "REFERENCES")
    body = _hit("body", BODY, "Introduction")
    body_2 = _hit("body-2", BODY_2, "Methods")
    retriever = _retriever(
        monkeypatch,
        dense_hits=[reference, body, body_2],
        bm25_hits=[reference, body_2, body],
    )

    result = retriever.search_with_trace("how does hybrid search work", top_k=10)

    assert "ref" not in result["trace"]["rrf_ids"]
    assert set(result["trace"]["rrf_ids"]) == {"body", "body-2"}
    assert {chunk["chunk_id"] for chunk in result["results"]} == {"body", "body-2"}


def test_prefusion_removal_is_counted_in_trace(monkeypatch):
    reference = _hit("ref", REFERENCE, "REFERENCES")
    body = _hit("body", BODY, "Introduction")
    retriever = _retriever(
        monkeypatch,
        dense_hits=[reference, body],
        bm25_hits=[reference],
    )

    trace = retriever.search_with_trace("how does hybrid search work", top_k=10)["trace"]

    assert trace["prefusion_references_removed"] == 2


def test_raw_source_traces_still_show_references(monkeypatch):
    reference = _hit("ref", REFERENCE, "REFERENCES")
    body = _hit("body", BODY, "Introduction")
    retriever = _retriever(
        monkeypatch,
        dense_hits=[reference, body],
        bm25_hits=[reference, body],
    )

    trace = retriever.search_with_trace("how does hybrid search work", top_k=10)["trace"]

    assert trace["dense_ids"] == ["ref", "body"]
    assert trace["bm25_ids"] == ["ref", "body"]


def test_references_survive_fusion_when_policy_allows_them(monkeypatch):
    reference = _hit("ref", REFERENCE, "REFERENCES")
    body = _hit("body", BODY, "Introduction")
    retriever = _retriever(
        monkeypatch,
        dense_hits=[reference, body],
        bm25_hits=[reference, body],
    )

    result = retriever.search_with_trace(
        "which paper introduced reciprocal rank fusion",
        top_k=10,
        allow_references=True,
        max_ref_ratio=1.0,
    )

    assert "ref" in result["trace"]["rrf_ids"]
    assert result["trace"]["prefusion_references_removed"] == 0


def test_prefusion_exclusion_can_be_disabled_for_ablation(monkeypatch):
    from src.retrieval import hybrid_retriever

    reference = _hit("ref", REFERENCE, "REFERENCES")
    body = _hit("body", BODY, "Introduction")
    retriever = _retriever(
        monkeypatch,
        dense_hits=[reference, body],
        bm25_hits=[reference, body],
    )
    monkeypatch.setattr(
        hybrid_retriever.settings, "prefusion_reference_exclusion", False
    )

    trace = retriever.search_with_trace("how does hybrid search work", top_k=10)["trace"]

    assert "ref" in trace["rrf_ids"]
    assert trace["prefusion_references_removed"] == 0
