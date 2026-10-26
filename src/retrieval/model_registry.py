from __future__ import annotations

import numpy as np
import threading
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.api.config import settings
from src.retrieval.reranker_rust import HybridReranker

EMBED_MODEL = settings.embed_model
RERANK_MODEL = settings.reranker_model


def _embedder_probe(embedder, model_name: str) -> None:
    """Fail loud bila embedder menghasilkan vektor kosong/NaN."""
    vec = embedder.encode(["probe sentence"], show_progress_bar=False)[0]
    arr = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"[ModelRegistry] Embedder '{model_name}' produced non-finite values")
    if float(np.linalg.norm(arr)) == 0.0:
        raise RuntimeError(
            f"[ModelRegistry] Embedder '{model_name}' produced all-zero vectors — "
            "dense retrieval would be silently broken. Set RAG_EMBED_MODEL to a "
            "working embedding model."
        )


class ModelRegistry:
    _instance = None

    def __init__(self):
        print(f"[ModelRegistry] Loading embedder: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        _embedder_probe(self.embedder, EMBED_MODEL)
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._hybrid_reranker = None
        print("[ModelRegistry] Embedder loaded. Reranker will load on first use.")

    @property
    def reranker(self):
        """Load the optional cross-encoder only when a caller requests reranking.

        Uses HybridReranker which prefers Rust (PyO3 ONNX) and falls back to
        Python sentence-transformers CrossEncoder.
        """
        if self._hybrid_reranker is None:
            with self._reranker_lock:
                if self._hybrid_reranker is None:
                    print(f"[ModelRegistry] Loading reranker: {RERANK_MODEL}")
                    self._hybrid_reranker = HybridReranker(model_name=RERANK_MODEL)
                    backend = self._hybrid_reranker.active or "none"
                    print(f"[ModelRegistry] Reranker loaded ({backend} backend).")
        return self._hybrid_reranker

    @property
    def reranker_backend(self) -> Optional[str]:
        """Return 'rust', 'python', or None. Used by observability/logging."""
        if self._hybrid_reranker is None:
            return None
        return self._hybrid_reranker.active

    def backend_name(self) -> str:
        """Human-readable backend name for log messages."""
        backend = self.reranker_backend
        return backend if backend else "none"

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance
