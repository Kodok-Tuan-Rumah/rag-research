from __future__ import annotations

import numpy as np
import threading
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.api.config import settings

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
        print("[ModelRegistry] Embedder loaded. Reranker will load on first use.")

    @property
    def reranker(self):
        """Load the optional cross-encoder only when a caller requests reranking."""
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    print(f"[ModelRegistry] Loading reranker: {RERANK_MODEL}")
                    reranker = CrossEncoder(RERANK_MODEL)
                    probe = reranker.predict([["q", "a"]], batch_size=1)
                    if not np.all(np.isfinite(np.asarray(probe, dtype=float))):
                        raise RuntimeError(
                            f"[ModelRegistry] Reranker '{RERANK_MODEL}' produced non-finite scores"
                        )
                    self._reranker = reranker
                    print("[ModelRegistry] Reranker loaded and ready for reuse.")
        return self._reranker

    @property
    def reranker_loaded(self) -> bool:
        """Report model state without triggering the optional model load."""
        return self._reranker is not None

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance
