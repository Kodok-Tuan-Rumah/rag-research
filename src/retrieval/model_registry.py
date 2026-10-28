from __future__ import annotations

import logging
import random
import threading
from typing import Optional

import numpy as np

from src.api.config import settings
from src.retrieval.reranker_rust import HybridReranker

try:
    from src.retrieval.inference_client import (
        InferenceClient,
        RemoteEmbedder,
        RemoteReranker,
    )
except Exception:  # pragma: no cover
    InferenceClient = None  # type: ignore
    RemoteEmbedder = None  # type: ignore
    RemoteReranker = None  # type: ignore

EMBED_MODEL = settings.embed_model
RERANK_MODEL = settings.reranker_model

logger = logging.getLogger(__name__)


def _load_local_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _load_local_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _embedder_probe(embedder, model_name: str) -> None:
    """Fail loud bila embedder menghasilkan vektor kosong/NaN."""
    if hasattr(embedder, "encode"):
        vec = embedder.encode(["probe sentence"], show_progress_bar=False)[0]
    elif hasattr(embedder, "embed"):
        arr = embedder.embed(["probe sentence"])
        vec = arr[0] if hasattr(arr, "__getitem__") else arr
    else:
        return
    arr = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"[ModelRegistry] Embedder '{model_name}' produced non-finite values")
    if float(np.linalg.norm(arr)) == 0.0:
        raise RuntimeError(
            f"[ModelRegistry] Embedder '{model_name}' produced all-zero vectors — "
            "dense retrieval would be silently broken. Set RAG_EMBED_MODEL to a "
            "working embedding model."
        )


class ShadowReranker:
    def __init__(self, local, remote, sample_rate: float = 0.1):
        self.local = local
        self.remote = remote
        self.sample_rate = sample_rate
        self._rng = random.Random(42)

    def predict(self, pairs, batch_size=1):
        local_scores = self.local.predict(pairs, batch_size=batch_size)
        if self._rng.random() < self.sample_rate:
            try:
                remote_scores = self.remote.predict(pairs)
                # log divergence (best effort)
                try:
                    import math

                    diff = 0.0
                    for a, b in zip(local_scores, remote_scores):
                        diff += abs(float(a) - float(b))
                    if diff > 0:
                        logger.info("shadow reranker divergence=%.4f", diff)
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("shadow remote failed: %s", exc)
        return local_scores

    @property
    def active(self):
        return getattr(self.local, "active", "shadow")

    @property
    def max_seq_length(self):
        return getattr(self.local, "max_seq_length", None)


class ModelRegistry:
    _instance = None

    def __init__(self):
        self._client = None
        backend = settings.inference_backend

        if backend in ("service", "shadow"):
            self._client = InferenceClient(
                base_url=settings.inference_service_url,
                timeout_ms=settings.inference_timeout_ms,
                connect_timeout_ms=settings.inference_connect_timeout_ms,
                max_retries=settings.inference_max_retries,
            )
            if settings.inference_required and not self._client.ready():
                raise RuntimeError("inference service is not ready")
            self.embedder = RemoteEmbedder(
                self._client,
                dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
            )
        else:
            self.embedder = _load_local_embedder(EMBED_MODEL)

        _embedder_probe(self.embedder, EMBED_MODEL)

        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._hybrid_reranker = None
        self._backend = backend
        print(f"[ModelRegistry] Embedder loaded (backend={backend}). Reranker will load on first use.")

    @property
    def reranker(self):
        if self._hybrid_reranker is None:
            with self._reranker_lock:
                if self._hybrid_reranker is None:
                    backend = self._backend
                    if backend == "service":
                        self._hybrid_reranker = RemoteReranker(
                            self._client, max_seq_length=settings.reranker_max_length
                        )
                    elif backend == "shadow":
                        local = HybridReranker(model_name=RERANK_MODEL)
                        remote = RemoteReranker(
                            self._client, max_seq_length=settings.reranker_max_length
                        )
                        self._hybrid_reranker = ShadowReranker(local=local, remote=remote)
                    else:
                        self._hybrid_reranker = HybridReranker(model_name=RERANK_MODEL)
                    print(f"[ModelRegistry] Reranker loaded ({backend} backend).")
        return self._hybrid_reranker

    @property
    def reranker_backend(self) -> Optional[str]:
        if self._hybrid_reranker is None:
            return None
        if self._backend == "service":
            return "service"
        if self._backend == "shadow":
            return "shadow"
        hr = self._hybrid_reranker
        return getattr(hr, "active", None)

    def backend_name(self) -> str:
        backend = self.reranker_backend
        return backend if backend else "none"

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance
