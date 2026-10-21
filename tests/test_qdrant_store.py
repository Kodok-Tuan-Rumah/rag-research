"""Unit tests untuk QdrantVectorStore dengan client + embedder ter-inject.

Store menerima override ``client`` dan ``embedder`` sehingga test ini tidak
pernah meng-import ``sentence_transformers`` (aman offline, tanpa cache model)
dan tidak pernah menyentuh storage Qdrant nyata.

Revisi sebelumnya menambal ``sys.modules`` secara borongan, dan tambalan itu
hanya menutupi *import* modul: ``ModelRegistry`` asli tetap di-import malas saat
method dipanggil, sehingga SentenceTransformer ikut dimuat (~14 detik) di dalam
test. Injeksi menghapus seluruh mekanisme tambalan tersebut.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.qdrant_store import (
    IndexCountMismatchError,
    IndexEmptyError,
    QdrantVectorStore,
    VECTOR_DIM,
)


class DummyEmbedder:
    def encode(self, texts, batch_size=None, show_progress_bar=False):
        return [[0.1] * VECTOR_DIM for _ in texts]


class DummyClient:
    def __init__(self, *args, initial_count=0, existing_collections=None, **kwargs):
        self.collections = list(existing_collections or [])
        self.vector_count = initial_count
        self.upserts = []
        self.closed = False

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, collection_name=None, **kwargs):
        self.collections.append(collection_name)

    def count(self, collection_name, exact=False):
        return SimpleNamespace(count=self.vector_count)

    def upsert(self, collection_name, points):
        self.upserts.extend(points)
        self.vector_count += len(points)

    def query_points(self, collection_name, query, limit, query_filter=None):
        return SimpleNamespace(points=[SimpleNamespace(
            payload={"chunk_id": "c1", "doc_id": "doc1", "text": "dummy"},
            score=0.85,
        )])

    def close(self):
        self.closed = True


def _make_store(client=None) -> QdrantVectorStore:
    os.environ["QDRANT_PATH"] = "tmp/qdrant_test"
    os.environ["QDRANT_COLLECTION"] = "test_collection"
    os.environ.pop("QDRANT_URL", None)
    os.environ.pop("QDRANT_API_KEY", None)
    return QdrantVectorStore(
        client=client or DummyClient(),
        embedder=DummyEmbedder(),
    )


def test_qdrant_store_env_overrides():
    store = _make_store()
    assert store.collection_name == "test_collection"
    # Assert terhadap konstanta modul, bukan literal. Embedder berubah dari
    # BAAI/bge-base-en-v1.5 (768-dim) ke all-MiniLM-L6-v2 (384-dim) dan asersi
    # literal ini dulu ikut basi diam-diam. qdrant_store = sumber kebenaran.
    assert store.vector_dim == VECTOR_DIM
    store.close()


def test_qdrant_store_search_returns_results():
    store = _make_store()
    results = store.search("hello world", k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == 0.85
    store.close()


def test_require_nonempty_index_rejects_empty_without_closing_store():
    store = _make_store()

    with pytest.raises(IndexEmptyError) as exc_info:
        store.require_nonempty_index(796, consumer="startup")

    assert exc_info.value.diagnostics["vector_count"] == 0
    assert store.client.closed is False
    # The caller must still be able to seed the same store after the check.
    assert store.search("still usable", k=1)[0].chunk_id == "c1"


def test_require_nonempty_index_rejects_count_drift_without_reset():
    client = DummyClient(
        initial_count=641,
        existing_collections=["test_collection"],
    )
    store = _make_store(client)

    with pytest.raises(IndexCountMismatchError) as exc_info:
        store.require_nonempty_index(796, consumer="startup")

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["expected_vector_count"] == 796
    assert diagnostics["vector_count"] == 641
    assert diagnostics["count_matches_expected"] is False
    assert store.client.closed is False
    assert client.upserts == []


def test_require_nonempty_index_accepts_matching_count():
    client = DummyClient(
        initial_count=796,
        existing_collections=["test_collection"],
    )
    store = _make_store(client)

    diagnostics = store.require_nonempty_index(796, consumer="startup")

    assert diagnostics["vector_count"] == 796
    assert diagnostics["count_matches_expected"] is True
    assert store.client.closed is False


def test_index_chunks_without_reset_does_not_delete_existing_collection():
    client = DummyClient(
        initial_count=0,
        existing_collections=["test_collection"],
    )
    store = _make_store(client)
    chunks = [{
        "chunk_id": "c1",
        "doc_id": "doc1",
        "text": (
            "Retrieval augmented generation combines a retriever with a language "
            "model to ground each answer in external documents."
        ),
        "metadata": {},
    }]

    store.index_chunks(chunks, reset=False)

    assert client.vector_count == 1
    assert len(client.upserts) == 1
    assert store.client.closed is False


if __name__ == "__main__":
    test_qdrant_store_env_overrides()
    test_qdrant_store_search_returns_results()
    print("OK")
