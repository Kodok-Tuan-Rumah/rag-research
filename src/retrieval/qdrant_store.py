from __future__ import annotations

import src._compat 

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional
from pathlib import Path
from urllib.parse import urlsplit

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from src.retrieval.chunk_filters import is_noise_chunk_text as _is_noise_chunk_text

CHUNK_ID_NAMESPACE = uuid.UUID('12345678-1234-5678-1234-567812345678')

COLLECTION_NAME = "rag_research"
QDRANT_PATH     = "./data/qdrant_storage"
VECTOR_DIM      = int(os.getenv("RAG_VECTOR_DIM", "384"))
BATCH_SIZE      = 128

def _is_noise_chunk(text: str) -> bool:
    """Return True jika chunk tidak layak di-index ke Qdrant."""
    return _is_noise_chunk_text(text)

@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id:   str
    text:     str
    score:    float
    metadata: dict
    retrieval_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    source_scores: Optional[dict] = None
    rrf_source_contributions: Optional[dict] = None

class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        qdrant_path: str = QDRANT_PATH,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        vector_dim: int = VECTOR_DIM,
        embedder: Optional[Any] = None,
        client: Optional[Any] = None,
    ):
        """Buat store.

        ``embedder`` dan ``client`` dapat di-inject oleh pemanggil (mis. test
        double) sehingga pemuatan model dan storage nyata dilewati sepenuhnya.
        Bila None, perilaku default (ModelRegistry + Qdrant lokal/remote)
        dipertahankan.
        """
        self.collection_name = os.getenv("QDRANT_COLLECTION", collection_name)
        self.vector_dim = vector_dim

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        qdrant_path = os.getenv("QDRANT_PATH", qdrant_path or QDRANT_PATH)

        # ── Qdrant client ──────────────────────────────────────────
        if client is not None:
            # Klien di-inject (test double): lifecycle dimiliki pemanggil,
            # jangan sentuh path lokal maupun jaringan.
            self.backend = "injected"
            self.storage_path = None
            self.endpoint = None
            self.client = client
        elif qdrant_url:
            self.backend = "remote"
            self.storage_path = None
            parsed_url = urlsplit(qdrant_url)
            host = parsed_url.hostname or "unknown-host"
            port = f":{parsed_url.port}" if parsed_url.port else ""
            self.endpoint = f"{parsed_url.scheme}://{host}{port}"
            print("[QdrantStore] Connecting to remote Qdrant")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.backend = "local"
            resolved_path = Path(qdrant_path).expanduser().resolve()
            self.storage_path = str(resolved_path)
            self.endpoint = None
            print(f"[QdrantStore] Using local Qdrant storage: {self.storage_path}")
            resolved_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(resolved_path))

        # ── Embedder: injectable agar test offline tidak memuat model ──
        if embedder is not None:
            self.embedder = embedder
            print("[QdrantStore] Using injected embedder")
        else:
            from src.retrieval.model_registry import ModelRegistry
            self.embedder = ModelRegistry.get().embedder
            print(f"[QdrantStore] Embedder reused from ModelRegistry")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            print(f"[QdrantStore] Collection '{self.collection_name}' created")
        else:
            count = self.client.count(self.collection_name).count
            print(f"[QdrantStore] Collection '{self.collection_name}' exists ({count} vectors)")

    def index_diagnostics(self) -> dict:
        """Return the exact backend/collection and vector count used by this store."""
        collection_names = {
            collection.name
            for collection in self.client.get_collections().collections
        }
        exists = self.collection_name in collection_names
        count = (
            int(self.client.count(self.collection_name, exact=True).count)
            if exists else 0
        )
        return {
            "backend": self.backend,
            "endpoint": self.endpoint,
            "storage_path": self.storage_path,
            "collection": self.collection_name,
            "collection_exists": exists,
            "vector_count": count,
            "vector_dim": self.vector_dim,
        }

    def require_nonempty_index(
        self,
        expected_corpus_chunks: int,
        consumer: str,
        query_probe: Optional[str] = None,
    ) -> dict:
        """Fail clearly when a configured retrieval corpus has no dense index."""
        diagnostics = self.index_diagnostics()
        probe_hits = self.search(query_probe, k=5) if query_probe else None
        diagnostics["probe_hit_count"] = len(probe_hits) if probe_hits is not None else None
        diagnostics["probe_chunk_ids"] = (
            [hit.chunk_id for hit in probe_hits] if probe_hits is not None else []
        )
        if expected_corpus_chunks > 0 and diagnostics["vector_count"] == 0:
            location = (
                diagnostics["storage_path"]
                if diagnostics["backend"] == "local"
                else diagnostics["endpoint"]
            )
            message = (
                f"Qdrant index is empty for {consumer}: collection="
                f"'{diagnostics['collection']}', backend={diagnostics['backend']}, "
                f"location={location}, exact_vector_count=0, "
                f"dense_probe_hits={diagnostics['probe_hit_count']}, "
                f"BM25 corpus chunks={expected_corpus_chunks}. "
                "Dense retrieval cannot run against this empty index. Verify "
                "QDRANT_URL/QDRANT_PATH/QDRANT_COLLECTION, then index the canonical "
                "chunks before starting the query service."
            )
            self.close()
            raise RuntimeError(message)
        return diagnostics

    def index_chunks(self, chunks: List[dict], reset: bool = False) -> None:
        if reset:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()

        # Filter noise sebelum embedding — hemat resource, jaga kualitas index
        original_count = len(chunks)
        chunks = [c for c in chunks if not _is_noise_chunk(c.get("text", ""))]
        noise_removed = original_count - len(chunks)
        if noise_removed > 0:
            print(f"[QdrantStore] Noise filter: {noise_removed} chunks dibuang dari {original_count} total")

        print(f"[QdrantStore] Indexing {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        all_embeddings = self.embedder.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False
        )

        points = []
        for chunk, emb in zip(chunks, all_embeddings):
            point_id = uuid.uuid5(CHUNK_ID_NAMESPACE, chunk["chunk_id"]).int >> 64
            vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id":   chunk["doc_id"],
                    "text":     chunk["text"],
                    **chunk.get("metadata", {}),
                }
            ))

        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+BATCH_SIZE]
            )
        print(f"[QdrantStore] Indexing done.")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        query_emb = self.embedder.encode([query], show_progress_bar=False)[0]
        query_emb = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        qdrant_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=filter_doc_id))])
            if filter_doc_id else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=k,
            query_filter=qdrant_filter,
        )

        return [
            RetrievalResult(
                chunk_id=r.payload["chunk_id"],
                doc_id=r.payload["doc_id"],
                text=r.payload["text"],
                score=r.score,
                metadata={
                    key: v for key, v in r.payload.items()
                    if key not in ("chunk_id", "doc_id", "text")
                },
            )
            for r in response.points
        ]
    def close(self):
        if hasattr(self, 'client'):
            try:
                self.client.close()
                print("[QdrantStore] Connection closed gracefully.")
            except Exception:
                pass


if __name__ == "__main__":
    import json, sys

    chunks_path = Path("data/processed/chunks_semantic_v2.json")
    if not chunks_path.exists():
        print(f"[ERROR] Canonical corpus {chunks_path} tidak ditemukan. Jalankan semantic_chunker.py dulu.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[CLI] Loaded {len(chunks)} chunks dari {chunks_path}")

    store = QdrantVectorStore()
    store.index_chunks(chunks, reset=True)

    # Verifikasi
    count = store.client.count(store.collection_name).count
    print(f"[CLI] Verifikasi: {count} vectors tersimpan di collection '{store.collection_name}'")
    store.close()
