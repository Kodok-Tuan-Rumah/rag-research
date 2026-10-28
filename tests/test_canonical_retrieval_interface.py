"""Unit tests memastikan adapter-adapter (API/CLI/Agent) memakai canonical
retrieval interface, dan MasterHybridRetriever menyediakan public stage API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_retriever import MasterHybridRetriever


def test_canonical_public_methods_exist():
    for name in (
        "dense_search",
        "bm25_search",
        "rrf_fuse_search",
        "score_pairs",
        "rerank",
        "search_with_trace",
        "search",
    ):
        assert callable(getattr(MasterHybridRetriever, name, None)), f"missing {name}"


def test_consumers_do_not_use_private_retrieval_internals():
    forbidden = ("._bm25", "._chunks", "vector_store.search(")
    consumer_files = [
        ROOT / "src" / "api" / "main.py",
        ROOT / "src" / "controller" / "agent.py",
        ROOT / "src" / "retrieval" / "tools.py",
        ROOT / "scripts" / "run_single_query.py",
    ]
    offenders = []
    for path in consumer_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append((str(path.relative_to(ROOT)), token))
    assert not offenders, f"private retrieval internals still used by adapters: {offenders}"


def test_consumer_uses_shared_rrf():
    text = (ROOT / "scripts" / "run_single_query.py").read_text(encoding="utf-8")
    assert "from src.retrieval.hybrid_retriever import" in text
    assert "rrf_fuse" in text
    assert "def reciprocal_rank_fusion" not in text


def test_stage_evaluator_explicitly_keeps_reranker_stage():
    text = (ROOT / "scripts" / "run_retrieval_eval.py").read_text(encoding="utf-8")
    assert "search_with_trace(" in text
    assert "rerank=rerank" in text
    assert '"--no-rerank"' in text
    assert "dense_weight=dense_weight" in text
    assert "bm25_weight=bm25_weight" in text
    assert "bm25_anchor_k=bm25_anchor_k" in text
    assert '"--bm25-anchor-k"' in text
