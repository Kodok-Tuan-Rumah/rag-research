def _classify_query(query: str) -> int:
    words = len(query.strip().split())
    if words <= 5:
        return 3
    if words >= 15:
        return 7
    return 5


def _retriever_with_fused_candidates(monkeypatch, fused, reranker):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            allow_references=False,
            max_ref_ratio=0.0,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = reranker
    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 40)
    monkeypatch.setattr(retriever, "_expand_and_hyde_parallel", lambda query: ([query], None))
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: fused)
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: fused)
    return retriever


def test_query_classifier():
    assert _classify_query('what is RAG') == 3
    assert _classify_query('this example query contains significantly more than fifteen words to ensure long query behavior extra tokens') == 7
    assert _classify_query('explain the difference between dense and BM25 retrieval techniques in RAG systems') == 5


def test_search_returns_empty_results_when_no_candidates_remain(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 30)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: [])

    assert retriever.search("query with no candidates") == []


def test_search_abstains_instead_of_falling_back_to_fragment(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general", allow_multi_query=False, allow_hyde=False,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None
    fragment = {
        "chunk_id": "fragment", "doc_id": "doc", "retrieval_score": 0.2,
        "text": "and generation must use this incomplete fragment",
        "metadata": {"section": "body"},
    }

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 30)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [fragment])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: [fragment])

    assert retriever.search("query") == []


def test_rerank_candidates_preserve_rrf_order(monkeypatch):
    """Regression: candidate selection must preserve RRF order after filtering."""
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general", allow_multi_query=False, allow_hyde=False,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None

    class FakeDense:
        def __init__(self, cid):
            self.chunk_id = cid
            self.doc_id = "doc"
            self.text = "This is a sufficiently long and informative body sentence with enough words."
            self.score = 0.9
            self.metadata = {"section": "body"}

    bm25_item = {
        "chunk_id": "bm25_chunk",
        "text": "This is another sufficiently long and informative body sentence with enough words.",
        "score": 0.5,
        "doc_id": "doc",
        "metadata": {"section": "body"},
        "retrieval_score": 0.01,
    }

    rrf_order = [FakeDense("dense_a"), bm25_item, FakeDense("dense_c")]

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 3)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: rrf_order)

    result = retriever.search("test query", top_k=3)
    ids = [r["chunk_id"] for r in result]
    assert ids == ["dense_a", "bm25_chunk", "dense_c"]
    assert {r["retrieval_method"] for r in result} == {"hybrid_rrf"}


def test_default_search_is_rrf_only_and_filters_full_candidate_list(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            allow_references=False,
            max_ref_ratio=0.0,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = object()
    references = [
        {
            "chunk_id": f"ref-{index}",
            "doc_id": "doc",
            "text": "This reference source provides academic findings about retrieval systems and helpfulness score measures.",
            "metadata": {"section": "References"},
            "retrieval_score": 0.5,
        }
        for index in range(20)
    ]
    evidence = {
        "chunk_id": "answer",
        "doc_id": "doc",
        "text": "The answer-bearing evidence describes a reliable approach to retrieval and ranking systems for question answering.",
        "metadata": {"section": "body"},
        "retrieval_score": 0.4,
    }
    fused = references + [evidence]
    dense_calls = []
    bm25_calls = []
    reranker_calls = []

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 40)
    monkeypatch.setattr(retriever, "_expand_and_hyde_parallel", lambda query: ([query], None))
    monkeypatch.setattr(
        retriever,
        "_dense_search",
        lambda query, k: dense_calls.append((query, k)) or fused,
    )
    monkeypatch.setattr(
        retriever,
        "_bm25_search",
        lambda query, k: bm25_calls.append((query, k)) or [],
    )
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: fused)
    monkeypatch.setattr(
        retriever,
        "_rerank",
        lambda query, candidates: reranker_calls.append(
            (query, [candidate["chunk_id"] for candidate in candidates])
        ) or (list(candidates), {
            "attempted": True,
            "succeeded": True,
            "error": None,
            "skipped_reason": None,
        }),
    )

    default = retriever.search_with_trace("question", top_k=10)
    explicit = retriever.search_with_trace("question", top_k=10, rerank=True)

    default_trace = default["trace"]
    explicit_trace = explicit["trace"]
    assert reranker_calls == [("question", [f"ref-{index}" for index in range(20)])]
    assert dense_calls == [("question", 40), ("question", 40)]
    assert bm25_calls == [("question", 40), ("question", 40)]
    assert default_trace["rrf_ids"] == explicit_trace["rrf_ids"]
    assert default_trace["dense_ids"] == explicit_trace["dense_ids"]
    assert default_trace["bm25_ids"] == explicit_trace["bm25_ids"]
    assert default_trace["reranker_executed"] is False
    assert default_trace["reranker_attempted"] is False
    assert default_trace["reranker_succeeded"] is False
    assert default_trace["rerank_input_count"] == 0
    assert default_trace["rerank_input_ids"] == []
    assert default_trace["ranking_mode"] == "rrf_only"
    assert [chunk["chunk_id"] for chunk in default["results"]] == ["answer"]
    assert default["results"][0]["retrieval_method"] == "hybrid_rrf"
    assert default["results"][0]["ranking_mode"] == "rrf_only"
    assert explicit_trace["reranker_executed"] is True
    assert explicit_trace["reranker_attempted"] is True
    assert explicit_trace["reranker_succeeded"] is True
    assert explicit_trace["rerank_input_count"] == 20
    assert explicit_trace["ranking_mode"] == "rrf_reranker"
    assert explicit["results"] == []


def test_reranker_failure_is_traced_and_preserves_rrf_order(monkeypatch, caplog):
    import logging

    class FailingReranker:
        max_seq_length = 512

        def predict(self, pairs, **kwargs):
            raise RuntimeError("inference unavailable")

    fused = [
        {
            "chunk_id": "rrf-first",
            "doc_id": "doc-1",
            "text": "This first evidence chunk explains retrieval ranking behavior in the system.",
            "metadata": {"section": "body"},
            "retrieval_score": 0.8,
        },
        {
            "chunk_id": "rrf-second",
            "doc_id": "doc-2",
            "text": "This second evidence chunk describes another valid system behavior clearly.",
            "metadata": {"section": "body"},
            "retrieval_score": 0.7,
        },
    ]
    retriever = _retriever_with_fused_candidates(monkeypatch, fused, FailingReranker())

    with caplog.at_level(logging.WARNING, logger="rag.retriever"):
        result = retriever.search_with_trace("test query", top_k=2, rerank=True)

    trace = result["trace"]
    assert trace["reranker_attempted"] is True
    assert trace["reranker_succeeded"] is False
    assert trace["reranker_executed"] is False
    assert "RuntimeError" in trace["reranker_error"]
    assert trace["reranker_skipped_reason"] is None
    assert trace["ranking_mode"] == "rrf_only"
    assert [chunk["ranking_mode"] for chunk in result["results"]] == ["rrf_only", "rrf_only"]
    assert trace["rrf_ids"] == ["rrf-first", "rrf-second"]
    assert [chunk["chunk_id"] for chunk in result["results"]] == trace["rrf_ids"]
    assert "RuntimeError" in caplog.text


def test_reranker_success_sets_actual_ranking_mode(monkeypatch):
    class SuccessfulReranker:
        max_seq_length = 512

        def predict(self, pairs, **kwargs):
            return [0.1, 0.9]

    fused = [
        {
            "chunk_id": "rrf-first",
            "doc_id": "doc-1",
            "text": "This first evidence chunk explains retrieval ranking behavior in the system.",
            "metadata": {"section": "body"},
            "retrieval_score": 0.8,
        },
        {
            "chunk_id": "rrf-second",
            "doc_id": "doc-2",
            "text": "This second evidence chunk describes another valid system behavior clearly.",
            "metadata": {"section": "body"},
            "retrieval_score": 0.7,
        },
    ]
    retriever = _retriever_with_fused_candidates(monkeypatch, fused, SuccessfulReranker())

    result = retriever.search_with_trace("test query", top_k=2, rerank=True)

    assert result["trace"]["reranker_attempted"] is True
    assert result["trace"]["reranker_succeeded"] is True
    assert result["trace"]["reranker_executed"] is True
    assert result["trace"]["reranker_error"] is None
    assert result["trace"]["ranking_mode"] == "rrf_reranker"
    assert {chunk["ranking_mode"] for chunk in result["results"]} == {"rrf_reranker"}


def test_reranker_skip_is_not_reported_as_failure(monkeypatch):
    fused = [
        {
            "chunk_id": "rrf-first",
            "doc_id": "doc-1",
            "text": "This first evidence chunk explains retrieval ranking behavior in the system.",
            "metadata": {"section": "body"},
            "retrieval_score": 0.8,
        },
    ]
    retriever = _retriever_with_fused_candidates(monkeypatch, fused, None)

    result = retriever.search_with_trace("test query", top_k=1, rerank=True)

    trace = result["trace"]
    assert trace["reranker_attempted"] is False
    assert trace["reranker_succeeded"] is False
    assert trace["reranker_executed"] is False
    assert trace["reranker_error"] is None
    assert trace["reranker_skipped_reason"] == "reranker unavailable"
    assert trace["ranking_mode"] == "rrf_only"
    assert result["results"][0]["ranking_mode"] == "rrf_only"


def test_rerank_sort_failure_does_not_mutate_candidates(monkeypatch):
    import builtins
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = object()
    candidates = [
        {"chunk_id": "first", "text": "first"},
        {"chunk_id": "second", "text": "second"},
    ]
    original_candidates = [dict(candidate) for candidate in candidates]
    monkeypatch.setattr(retriever, "score_pairs", lambda pairs: [0.1, 0.9])

    def fail_sort(*args, **kwargs):
        raise RuntimeError("sort failed")

    monkeypatch.setattr(builtins, "sorted", fail_sort)
    ranked, status = retriever._rerank("query", candidates)

    assert ranked is candidates
    assert status["attempted"] is True
    assert status["succeeded"] is False
    assert "RuntimeError" in status["error"]
    assert candidates == original_candidates


def test_public_rerank_text_of_uses_shared_rerank_path(monkeypatch):
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    class SuccessfulReranker:
        max_seq_length = 512

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = SuccessfulReranker()
    scored_pairs = []

    def score_pairs(pairs):
        scored_pairs.extend(pairs)
        return [0.1, 0.9]

    monkeypatch.setattr(retriever, "score_pairs", score_pairs)
    candidates = [
        {"chunk_id": "first", "window": "first window"},
        {"chunk_id": "second", "window": "second window"},
    ]

    ranked = retriever.rerank("query", candidates, text_of=lambda hit: hit["window"])

    assert scored_pairs == [
        ["query", "first window"],
        ["query", "second window"],
    ]
    assert [chunk["chunk_id"] for chunk in ranked] == ["second", "first"]
    assert [chunk["rerank_score"] for chunk in ranked] == [0.9, 0.1]


def test_public_rerank_text_of_failure_uses_logged_fallback(caplog):
    import logging
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    class FailingReranker:
        max_seq_length = 512

        def predict(self, pairs, **kwargs):
            raise RuntimeError("inference unavailable")

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = FailingReranker()
    candidates = [
        {"chunk_id": "first", "window": "first window"},
        {"chunk_id": "second", "window": "second window"},
    ]

    with caplog.at_level(logging.WARNING, logger="rag.retriever"):
        ranked = retriever.rerank("query", candidates, text_of=lambda hit: hit["window"])

    assert ranked is candidates
    assert [chunk["chunk_id"] for chunk in ranked] == ["first", "second"]
    assert all("rerank_score" not in chunk for chunk in ranked)
    assert "RuntimeError" in caplog.text


def test_rrf_preserves_original_dense_and_bm25_scores():
    from src.retrieval.hybrid_retriever import rrf_fuse
    from src.retrieval.qdrant_store import RetrievalResult

    dense = RetrievalResult(
        chunk_id="shared", doc_id="doc", text="dense result", score=0.82, metadata={}
    )
    bm25 = {
        "chunk_id": "shared", "doc_id": "doc", "text": "bm25 result",
        "score": 7.5, "metadata": {},
    }

    fused = rrf_fuse(
        [[dense], [bm25]], k=60, weights=[0.7, 0.3],
        source_labels=["dense", "bm25"],
    )

    assert fused == [dense]
    assert dense.score == 0.82
    assert dense.rrf_score == dense.retrieval_score
    assert dense.source_scores == {"dense": 0.82, "bm25": 7.5}


def test_candidate_pool_respects_configured_cap(monkeypatch):
    from src.retrieval import hybrid_retriever
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    monkeypatch.setattr(hybrid_retriever.settings, "candidate_k", 32)

    assert retriever._choose_candidate_k(5) == 30
    assert retriever._choose_candidate_k(10) == 32


def test_multi_query_and_hyde_are_used_by_dense_and_sparse_search(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            allow_references=False,
            max_ref_ratio=0.0,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None
    dense_queries = []
    bm25_queries = []
    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 30)
    monkeypatch.setattr(
        retriever, "_expand_and_hyde_parallel",
        lambda query: ([query, "alternative"], "hypothetical"),
    )
    monkeypatch.setattr(
        retriever, "_dense_search",
        lambda query, k: dense_queries.append(query) or [],
    )
    monkeypatch.setattr(
        retriever, "_bm25_search",
        lambda query, k: bm25_queries.append(query) or [],
    )
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: [])

    trace = retriever.search_with_trace(
        "original", top_k=5, use_multi_query=True, use_hyde=True
    )["trace"]

    assert dense_queries == ["original", "alternative", "hypothetical"]
    assert bm25_queries == ["original", "alternative"]
    assert trace["dense_query_count"] == 3
    assert trace["bm25_query_count"] == 2
    assert trace["hyde_used"] is True


if __name__ == '__main__':
    test_query_classifier()
    print('OK')
