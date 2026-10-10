# Legacy Code — Archived 2026-10-05

Files moved here during the over-engineering audit. **Nothing was deleted** — all
files are recoverable from git history.

## Why

These modules had **0 inbound imports** across the entire codebase
(`src/`, `scripts/`, `tests/`, `demo.py`), verified by AST scan. They inflated
`src/` statistics (5.2% of 5,208 LOC) while contributing nothing to the
canonical retrieval path.

Their presence was actively harmful in two cases: they hard-import packages
that are **not** in `requirements.txt`, so they could not be imported or run at
all — a trap for anyone who found them.

## Files

| File | LOC | Why it is dead | Imports cleanly? |
|---|---:|---|---|
| `retrieval/bm25_retriever.py` | 103 | `MasterHybridRetriever._bm25_search()` (line 308) is the live implementation. This module was a **duplicate** BM25 path. | Yes — but unused |
| `retrieval/embedder.py` | 127 | Legacy FAISS embedder. `retrieval_backend` defaults to `QDRANT` (`src/api/config.py`). | ❌ **No** — hard-imports `faiss`, absent from requirements |
| `evaluation/evaluator.py` | 268 | Legacy Ollama-based RAGAS judge, superseded by `evaluation/ragas_evaluator.py` (Groq). | ❌ **No** — hard-imports `ollama`, absent from requirements |
| `ingestion/dataset_builder.py` | 220 | QA dataset generation, superseded by `scripts/build_retrieval_eval_dataset.py` + `evaluation/gold_matching.py`. | Yes — but unused |

## NOT archived (deliberately)

| File | LOC | Reason kept |
|---|---:|---|
| `src/api/main.py` | 311 | 0 static imports, but it is an **ASGI entrypoint** — `uvicorn src.api.main:app`. Verified importable (`FastAPI`). Launched by `scripts/coverage_metric.py:368` and `Makefile`. |
| `src/ingestion/semantic_chunker.py` | 222 | 0 static imports, but is a **CLI entrypoint** — `Makefile:16` runs `python src/ingestion/semantic_chunker.py`, and `tests/test_semantic_chunker.py` imports it via `importlib.import_module`. |

Static import scanning cannot distinguish dead code from entrypoints. Both of
the above are live.

## BM25 duplication detail

`bm25_retriever.py` exposed `tokenize`, `build_bm25_index`, `save_bm25`,
`load_bm25`, `bm25_search`. The canonical pipeline builds its index inline in
`MasterHybridRetriever.__init__` (line ~120) and searches via `_bm25_search`
(line 308), exposed publicly as `bm25_search()` (line 325) per the Phase-1
canonical interface refactor. Two implementations of the same retrieval mode is
exactly the kind of duplication the "one canonical pipeline" rule forbids.

## Restore

```bash
git mv archive/legacy_2026_10/retrieval/bm25_retriever.py src/retrieval/
```

Note: restoring `embedder.py` or `evaluator.py` also requires adding `faiss-cpu`
or `ollama` to `requirements.txt`, otherwise the module stays unimportable.

---

*Archived as part of the over-engineering audit. Evidence:
`results/metrics/k1_overengineering_audit.md`*