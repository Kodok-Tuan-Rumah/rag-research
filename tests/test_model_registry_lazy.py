from __future__ import annotations

import numpy as np


def test_model_registry_loads_cross_encoder_only_on_first_access(monkeypatch):
    from src.retrieval import model_registry
    from src.retrieval.reranker_rust import RustCrossEncoderReranker

    calls = []

    class FakeEmbedder:
        def encode(self, texts, show_progress_bar=False):
            return [np.array([1.0, 0.0])]

    class FakeReranker:
        max_seq_length = 128

        def predict(self, pairs, batch_size=1):
            return np.array([0.5])

    monkeypatch.setattr(
        model_registry, "_load_local_embedder",
        lambda name: calls.append(("embedder", name)) or FakeEmbedder(),
    )
    monkeypatch.setattr(
        model_registry, "HybridReranker",
        lambda model_name=None, prefer_rust=True: calls.append(("reranker", model_name)) or FakeReranker(),
    )

    registry = model_registry.ModelRegistry()

    # Reranker should NOT be loaded at construction time
    assert [kind for kind, _ in calls] == ["embedder"]

    first = registry.reranker
    second = registry.reranker

    assert first is second
    # Either Python CrossEncoder or Rust reranker (or both during fallback) is loaded on first access
    reranker_calls = [kind for kind, _ in calls if kind.startswith("reranker")]
    assert len(reranker_calls) >= 1, f"Expected at least one reranker call, got {calls}"


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
