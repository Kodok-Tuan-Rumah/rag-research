from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.qdrant_store import QdrantVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks_semantic_v3.json",
    )
    parser.add_argument(
        "--collection",
        default="rag_research_phase1_v3",
        help="Dedicated collection. Existing non-empty collections are never reset.",
    )
    args = parser.parse_args()
    os.environ["QDRANT_COLLECTION"] = args.collection

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunk corpus must be a non-empty JSON list")

    store = QdrantVectorStore(collection_name=args.collection)
    try:
        before = store.index_diagnostics()
        if before["vector_count"]:
            raise RuntimeError(
                f"Refusing to modify non-empty collection {args.collection!r} "
                f"({before['vector_count']} vectors). Choose a fresh collection."
            )
        store.index_chunks(chunks, reset=False)
        result = store.require_nonempty_index(
            expected_corpus_chunks=len(chunks),
            consumer="Phase 1 v3 isolated index",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
