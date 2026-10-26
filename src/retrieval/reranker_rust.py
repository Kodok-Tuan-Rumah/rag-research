"""Rust cross-encoder reranker wrapper with Python fallback.

Tries to load the Rust PyO3 extension first. If unavailable (not built yet),
falls back to the existing sentence-transformers CrossEncoder in Python.

Interface matches the existing reranker's `predict(pairs, batch_size, max_length)`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import os
import sys

# Ensure ORT_DYLIB_PATH is set for the Rust extension (load-dynamic feature).
# Falls back to the Python onnxruntime's shared library if not already set.
if "ORT_DYLIB_PATH" not in os.environ:
    try:
        import onnxruntime
        ort_lib = os.path.join(
            os.path.dirname(onnxruntime.__file__),
            "capi",
            "libonnxruntime.so.1.31.0",
        )
        if os.path.isfile(ort_lib):
            os.environ["ORT_DYLIB_PATH"] = ort_lib
    except ImportError:
        pass

logger = logging.getLogger("rag.reranker_rust")

# Paths relative to the rag-research repo root
_RUST_MODULE_NAME = "cross_encoder_reranker"
_MODEL_DIR = Path("models")
_ONNX_MODEL_PATH = _MODEL_DIR / "model_optimized.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"


class RustCrossEncoderReranker:
    """Wrapper around the Rust cross_encoder_reranker PyO3 module."""

    def __init__(self, model_path: str = str(_ONNX_MODEL_PATH),
                 tokenizer_path: str = str(_TOKENIZER_PATH)):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._rust_module: Optional[Any] = None
        self._reranker: Optional[Any] = None
        self._available = False
        self.max_seq_length = 512

        self._try_load_rust()

    def _try_load_rust(self) -> None:
        """Attempt to import and initialize the Rust extension."""
        try:
            mod = __import__(_RUST_MODULE_NAME)
            self._rust_module = mod
            # PyO3 #[new] creates __new__, so we instantiate directly
            self._reranker = mod.PyCrossEncoderReranker()
            self._reranker.load_model(self.model_path, self.tokenizer_path)
            self._available = True
            logger.info(
                "[RerankerRust] Loaded Rust cross-encoder reranker "
                f"(model={self.model_path})"
            )
        except Exception as exc:
            logger.warning(
                "[RerankerRust] Rust extension unavailable (%s); "
                "will fall back to Python sentence-transformers", exc
            )
            self._available = False
            self._rust_module = None
            self._reranker = None

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, pairs: list, batch_size: int = 32, max_length: int = 512) -> list:
        """Score (query, text) pairs.

        Args:
            pairs: List of [query, text] pairs.
            batch_size: Ignored (Rust handles batching internally).
            max_length: Ignored (fixed at 512 in Rust).

        Returns:
            List of float relevance scores.
        """
        if not self._available:
            raise RuntimeError("Rust reranker not available")

        rust_pairs = [(str(q), str(t)) for q, t in pairs]
        scores = self._reranker.score_pairs(rust_pairs)
        return [float(s) for s in scores]

    def __call__(self, pairs: list, batch_size: int = 32, max_length: int = 512) -> list:
        return self.predict(pairs, batch_size=batch_size, max_length=max_length)


class HybridReranker:
    """Auto-selecting reranker: Rust if available, Python otherwise."""

    def __init__(self, model_name: str | None = None, prefer_rust: bool = True):
        self.model_name = model_name or settings.reranker_model
        self.prefer_rust = prefer_rust
        self._rust: Optional[RustCrossEncoderReranker] = None
        self._python_reranker: Optional[Any] = None
        self._active: Optional[str] = None  # "rust" or "python"
        self.max_seq_length = 512

        self._init()

    def _init(self) -> None:
        if self.prefer_rust:
            try:
                self._rust = RustCrossEncoderReranker()
                if self._rust.available:
                    self._active = "rust"
                    logger.info("[Reranker] Using Rust cross-encoder reranker")
                    return
            except Exception as exc:
                logger.warning(f"[Reranker] Rust init failed: {exc}")

        # Fall back to Python sentence-transformers
        # Import lazily to avoid circular import with model_registry and to
        # allow monkeypatching via model_registry.CrossEncoder in tests.
        from src.retrieval.model_registry import CrossEncoder

        try:
            self._python_reranker = CrossEncoder(self.model_name)
            probe = self._python_reranker.predict([["q", "a"]], batch_size=1)
            import numpy as np
            if not np.all(np.isfinite(np.asarray(probe, dtype=float))):
                raise RuntimeError("Python reranker produced non-finite scores")
            self._active = "python"
            logger.info("[Reranker] Using Python sentence-transformers reranker")
        except Exception as exc:
            logger.error(f"[Reranker] No reranker available: {exc}")
            self._active = None

    @property
    def active(self) -> Optional[str]:
        return self._active

    @property
    def is_rust(self) -> bool:
        return self._active == "rust"

    def predict(self, pairs: list, batch_size: int = 32, max_length: int = 512) -> list:
        """Score (query, text) pairs using the active backend."""
        if self._active == "rust" and self._rust is not None:
            return self._rust.predict(pairs, batch_size=batch_size, max_length=max_length)
        if self._active == "python" and self._python_reranker is not None:
            return [float(s) for s in self._python_reranker.predict(
                pairs, batch_size=batch_size, max_length=max_length
            )]
        # No reranker available - return neutral scores
        logger.warning("[Reranker] No backend available; returning neutral scores")
        return [0.0] * len(pairs)

    def backend_name(self) -> str:
        """Return the active backend name for logging."""
        return self._active or "none"

    def is_rust(self) -> bool:
        """Return True if the Rust backend is active."""
        return self._active == "rust"
