from __future__ import annotations

import numpy as np
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
        print("[ModelRegistry] Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)
        probe = self.reranker.predict([["q", "a"]], batch_size=1)
        if not np.all(np.isfinite(np.asarray(probe, dtype=float))):
            raise RuntimeError(f"[ModelRegistry] Reranker '{RERANK_MODEL}' produced non-finite scores")
        print("[ModelRegistry] All models loaded. Will reuse for all queries.")

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance