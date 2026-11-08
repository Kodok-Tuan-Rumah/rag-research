"""Single source of truth for the RAG corpus manifest.

All components (config, hybrid_retriever, qdrant_store CLI, api/main) read the
corpus path, chunk count, and SHA256 from this manifest instead of hardcoding
v2 vs v3. This guarantees chunker → indexer → BM25 always operate on the same
file.

Manifest: ``rag-research/data/processed/corpus_manifest.json``
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "corpus_manifest.json"


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    """Load and return the corpus manifest dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def corpus_path(path: str | Path = MANIFEST_PATH) -> Path:
    """Return the absolute path to the active corpus chunks file."""
    m = load_manifest(path)
    p = Path(m["path"])
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def corpus_chunk_count(path: str | Path = MANIFEST_PATH) -> int:
    """Return the expected chunk count from the manifest."""
    return int(load_manifest(path)["chunk_count"])


def corpus_sha256(path: str | Path = MANIFEST_PATH) -> str:
    """Return the expected SHA256 fingerprint from the manifest."""
    return str(load_manifest(path)["sha256"])


def corpus_version(path: str | Path = MANIFEST_PATH) -> str:
    """Return the active corpus version string (e.g. 'v3')."""
    return str(load_manifest(path)["corpus_version"])


# Module-level constants computed once at import time.
CORPUS_PATH = corpus_path()
CORPUS_CHUNK_COUNT = corpus_chunk_count()
CORPUS_SHA256 = corpus_sha256()
CORPUS_VERSION = corpus_version()

__all__ = [
    "load_manifest",
    "corpus_path",
    "corpus_chunk_count",
    "corpus_sha256",
    "corpus_version",
    "CORPUS_PATH",
    "CORPUS_CHUNK_COUNT",
    "CORPUS_SHA256",
    "CORPUS_VERSION",
    "MANIFEST_PATH",
]
