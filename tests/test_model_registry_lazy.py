from __future__ import annotations

import numpy as np


def test_model_registry_loads_cross_encoder_only_on_first_access(monkeypatch):
    from src.retrieval import model_registry

    calls = []

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False):
            return [np.array([1.0, 0.0])]

    class FakeReranker:
        max_seq_length = 128

        def predict(self, pairs, batch_size=1):
            return np.array([0.5])

    monkeypatch.setattr(
        model_registry, "SentenceTransformer",
        lambda name: calls.append(("embedder", name)) or FakeEmbedder(),
    )
    monkeypatch.setattr(
        model_registry, "CrossEncoder",
        lambda name: calls.append(("reranker", name)) or FakeReranker(),
    )

    registry = model_registry.ModelRegistry()

    assert [kind for kind, _ in calls] == ["embedder"]
    first = registry.reranker
    second = registry.reranker

    assert first is second
    assert [kind for kind, _ in calls] == ["embedder", "reranker"]


def test_master_retriever_constructor_does_not_request_reranker(monkeypatch, tmp_path):
    from src.retrieval.model_registry import ModelRegistry
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    accesses = []

    class FakeRegistry:
        reranker_loaded = False

        @property
        def reranker(self):
            accesses.append("reranker")
            return None

    monkeypatch.setattr(ModelRegistry, "get", classmethod(lambda cls: FakeRegistry()))

    retriever = MasterHybridRetriever(
        vector_store=object(),
        bm25_chunks_path=str(tmp_path / "missing_chunks.json"),
    )

    assert accesses == []
    assert retriever.reranker_loaded is False
    assert accesses == []
    assert retriever.reranker is None
    assert accesses == ["reranker"]


def test_typed_policy_plan_matches_dictionary_policy():
    from src.controller.policy_engine import PolicyEngine

    engine = PolicyEngine()
    query = "How does retrieval augmented generation compare with fine tuning?"
    resolved = engine.resolve(query)
    plan = engine.initial_plan(query)

    assert plan.query_type == resolved["query_type"]
    assert plan.top_k_hint == resolved["retrieval"]["top_k"]
    assert plan.allow_multi_query == resolved["retrieval"]["use_multi_query"]
    assert plan.allow_hyde == resolved["retrieval"]["use_hyde"]
    assert plan.allow_references == resolved["reference"]["allow_references"]
    assert plan.max_ref_ratio == resolved["reference"]["max_ref_ratio"]
    assert plan.generation_hint == resolved["generation"]


def test_agent_state_tracks_shared_retrieval_retry_limit():
    from src.controller.agent_state import MAX_RETRIEVAL_RETRIES, AgentState

    state = AgentState(query="question")
    assert state.retry_attempted is False
    assert MAX_RETRIEVAL_RETRIES == 1

    state.retrieval_retries += 1

    assert state.retry_attempted is True
    assert state.retrieval_retries == MAX_RETRIEVAL_RETRIES
