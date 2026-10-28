import json

import httpx
import numpy as np
import pytest

from src.retrieval.inference_client import (
    InferenceClient,
    InferenceError,
    RemoteEmbedder,
    RemoteReranker,
)


def client_with_handler(handler, **kwargs):
    return InferenceClient(
        "http://inference",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_embed_returns_float32_matrix():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "dim": 2,
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "truncated": 0,
                "max_length": 256,
                "latency_ms": 1.0,
            },
        )

    client = client_with_handler(handler)
    array = client.embed(["a", "b"])
    assert array.shape == (2, 2)
    assert array.dtype == np.float32
    assert array[0].tolist() == pytest.approx([0.1, 0.2])


def test_remote_embedder_single_string_returns_1d():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "m", "dim": 2, "embeddings": [[0.1, 0.2]],
                  "truncated": 0, "max_length": 256, "latency_ms": 1.0},
        )

    embedder = RemoteEmbedder(client_with_handler(handler), dim=2)
    array = embedder.encode("single")
    assert array.shape == (2,)


def test_embed_splits_large_batches():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(len(payload["texts"]))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "dim": 2,
                "embeddings": [[0.0, 1.0] for _ in payload["texts"]],
                "truncated": 0,
                "max_length": 256,
                "latency_ms": 1.0,
            },
        )

    embedder = RemoteEmbedder(client_with_handler(handler), dim=2, batch_size=64)
    array = embedder.encode([f"text-{index}" for index in range(130)])
    assert array.shape == (130, 2)
    assert calls == [64, 64, 2]


def test_rerank_groups_pairs_by_query_and_preserves_order():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "scores": [
                    {"id": document["id"], "score": float(index)}
                    for index, document in enumerate(payload["documents"])
                ],
                "truncated": 0,
                "max_length": 256,
                "latency_ms": 1.0,
            },
        )

    reranker = RemoteReranker(client_with_handler(handler), max_seq_length=256)
    scores = reranker.predict([("q1", "a"), ("q2", "b"), ("q1", "c")])
    assert scores == [0.0, 0.0, 1.0]


def test_rerank_empty_pairs_returns_empty():
    reranker = RemoteReranker(client_with_handler(lambda request: httpx.Response(500)))
    assert reranker.predict([]) == []


def test_client_retries_503_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": {"code": "queue_full", "message": "full"}})
        return httpx.Response(
            200,
            json={"model": "m", "dim": 2, "embeddings": [[0.0, 1.0]],
                  "truncated": 0, "max_length": 256, "latency_ms": 1.0},
        )

    client = client_with_handler(handler, max_retries=1)
    assert client.embed(["a"]).shape == (1, 2)
    assert calls["count"] == 2


def test_client_raises_after_retries_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "queue_full", "message": "full"}})

    client = client_with_handler(handler, max_retries=1)
    with pytest.raises(InferenceError, match="queue_full"):
        client.embed(["a"])


def test_client_timeout_raises_inference_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = client_with_handler(handler, max_retries=0)
    with pytest.raises(InferenceError):
        client.embed(["a"])


def test_client_rejects_wrong_dimension():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "m", "dim": 3, "embeddings": [[0.0, 1.0, 2.0]],
                  "truncated": 0, "max_length": 256, "latency_ms": 1.0},
        )

    embedder = RemoteEmbedder(client_with_handler(handler), dim=2)
    with pytest.raises(InferenceError):
        embedder.encode(["a"])


def test_circuit_breaker_opens_after_repeated_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "queue_full", "message": "full"}})

    client = client_with_handler(handler, max_retries=0)
    for _ in range(5):
        with pytest.raises(InferenceError):
            client.embed(["a"])
    with pytest.raises(InferenceError, match="circuit open"):
        client.embed(["a"])
