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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.qdrant_store import VECTOR_DIM, QdrantVectorStore


class DummyEmbedder:
    def encode(self, texts, batch_size=None, show_progress_bar=False):
        return [[0.1] * VECTOR_DIM for _ in texts]


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.collections = []
        self.upserts = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, collection_name=None, **kwargs):
        self.collections.append(collection_name)

    def count(self, collection_name, exact=False):
        return SimpleNamespace(count=0)

    def upsert(self, collection_name, points):
        self.upserts.extend(points)

    def query_points(self, collection_name, query, limit, query_filter=None):
        return SimpleNamespace(points=[SimpleNamespace(
            payload={"chunk_id": "c1", "doc_id": "doc1", "text": "dummy"},
            score=0.85,
        )])

    def close(self):
        pass


def _make_store() -> QdrantVectorStore:
    os.environ["QDRANT_PATH"] = "tmp/qdrant_test"
    os.environ["QDRANT_COLLECTION"] = "test_collection"
    os.environ.pop("QDRANT_URL", None)
    os.environ.pop("QDRANT_API_KEY", None)
    return QdrantVectorStore(client=DummyClient(), embedder=DummyEmbedder())


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


if __name__ == "__main__":
    test_qdrant_store_env_overrides()
    test_qdrant_store_search_returns_results()
    print("OK")
