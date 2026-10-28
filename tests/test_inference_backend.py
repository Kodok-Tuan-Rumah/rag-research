import numpy as np
import pytest

from src.retrieval import model_registry


class FakeClient:
    def __init__(self, *, ready=True, fail=False):
        self.ready_value = ready
        self.fail = fail
        self.embed_calls = 0
        self.rerank_calls = 0

    def ready(self):
        if self.fail:
            raise RuntimeError("service down")
        return self.ready_value

    def embed(self, texts):
        self.embed_calls += 1
        return np.ones((len(texts), 384), dtype=np.float32)

    def rerank(self, query, documents):
        self.rerank_calls += 1
        return [0.5 for _ in documents]

    def close(self):
        pass


def test_service_backend_uses_remote_embedder(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(model_registry.settings, "inference_backend", "service")
    monkeypatch.setattr(model_registry, "InferenceClient", lambda **kwargs: fake)
    monkeypatch.setattr(model_registry, "RemoteEmbedder", lambda client, **kwargs: fake)
    monkeypatch.setattr(model_registry, "RemoteReranker", lambda client, **kwargs: fake)

    registry = model_registry.ModelRegistry()
    assert registry.embedder is fake
    assert registry.reranker is fake
    assert fake.embed_calls >= 1  # probe


def test_registry_fails_fast_when_required_and_service_down(monkeypatch):
    class DownClient(FakeClient):
        def ready(self):
            raise RuntimeError("service down")

    monkeypatch.setattr(model_registry.settings, "inference_backend", "service")
    monkeypatch.setattr(model_registry.settings, "inference_required", True)
    monkeypatch.setattr(model_registry, "InferenceClient", lambda **kwargs: DownClient())

    with pytest.raises(RuntimeError, match="service down"):
        model_registry.ModelRegistry()


def test_local_backend_does_not_require_inference_service(monkeypatch):
    class FakeLocalEmbedder:
        def encode(self, texts, show_progress_bar=False):
            return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(model_registry.settings, "inference_backend", "local")
    monkeypatch.setattr(model_registry, "_load_local_embedder", lambda name: FakeLocalEmbedder())
    registry = model_registry.ModelRegistry()
    assert isinstance(registry.embedder, FakeLocalEmbedder)
