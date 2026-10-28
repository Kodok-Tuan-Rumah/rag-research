from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.api import main
from src.retrieval.qdrant_store import IndexCountMismatchError, IndexEmptyError


class FakeStore:
    collection_name = "test_collection"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.require_calls = []
        self.indexed = []

    def require_nonempty_index(self, *, expected_corpus_chunks, consumer, query_probe=None):
        self.require_calls.append(
            {
                "expected_corpus_chunks": expected_corpus_chunks,
                "consumer": consumer,
                "query_probe": query_probe,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def index_chunks(self, chunks, reset=False):
        self.indexed.append((chunks, reset))


def _write_chunks(path: Path, chunks: list[dict]) -> str:
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk(chunk_id: str = "c1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "text": "A complete body chunk for startup validation.",
        "metadata": {},
    }


def test_load_canonical_corpus_validates_count_and_fingerprint(tmp_path):
    path = tmp_path / "chunks.json"
    chunks = [_chunk("c1"), _chunk("c2")]
    digest = _write_chunks(path, chunks)

    loaded = main._load_canonical_corpus(
        path,
        expected_count=2,
        expected_sha256=digest,
    )

    assert loaded == chunks


def test_load_canonical_corpus_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Canonical corpus not found"):
        main._load_canonical_corpus(
            tmp_path / "missing.json",
            expected_count=2,
            expected_sha256="",
        )


def test_load_canonical_corpus_rejects_wrong_count(tmp_path):
    path = tmp_path / "chunks.json"
    digest = _write_chunks(path, [_chunk("c1")])

    with pytest.raises(ValueError, match="has 1 chunks; expected 2"):
        main._load_canonical_corpus(
            path,
            expected_count=2,
            expected_sha256=digest,
        )


def test_load_canonical_corpus_rejects_wrong_fingerprint(tmp_path):
    path = tmp_path / "chunks.json"
    _write_chunks(path, [_chunk("c1")])

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        main._load_canonical_corpus(
            path,
            expected_count=1,
            expected_sha256="0" * 64,
        )


def test_ensure_startup_index_seeds_only_an_empty_collection():
    chunks = [_chunk("c1"), _chunk("c2")]
    store = FakeStore(
        [
            IndexEmptyError("empty"),
            {"vector_count": 2, "collection": "test_collection"},
        ]
    )

    diagnostics = main._ensure_startup_index(
        store,
        chunks,
        expected_count=2,
        consumer="test startup",
    )

    assert diagnostics["vector_count"] == 2
    assert store.indexed == [(chunks, False)]
    assert [call["expected_corpus_chunks"] for call in store.require_calls] == [2, 2]
    assert store.require_calls[1]["consumer"] == "test startup (after canonical seed)"


def test_ensure_startup_index_refuses_nonempty_count_drift():
    store = FakeStore([IndexCountMismatchError("count mismatch")])

    with pytest.raises(IndexCountMismatchError, match="count mismatch"):
        main._ensure_startup_index(
            store,
            [_chunk("c1")],
            expected_count=796,
            consumer="test startup",
        )

    assert store.indexed == []
    assert store.require_calls[0]["expected_corpus_chunks"] == 796


def test_ensure_startup_index_does_not_reindex_an_already_valid_collection():
    chunks = [_chunk("c1")]
    store = FakeStore([{"vector_count": 1, "collection": "test_collection"}])

    diagnostics = main._ensure_startup_index(
        store,
        chunks,
        expected_count=1,
        consumer="test startup",
    )

    assert diagnostics["vector_count"] == 1
    assert store.indexed == []


def test_startup_source_does_not_invoke_ingestion_modules():
    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "src.ingestion.document_loader" not in source
    assert "src.ingestion.semantic_chunker" not in source
    assert "subprocess" not in source


def test_live_canonical_corpus_passes_startup_validation():
    chunks = main._load_canonical_corpus()

    assert len(chunks) == main.CANONICAL_CORPUS_CHUNKS
