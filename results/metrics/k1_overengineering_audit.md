# K1 — Over-Engineering Audit

> Date: 2026-10-09
> Scope: `src/`, `scripts/`, `tests/`, `demo.py` — 87 Python files scanned via AST.
> Method: static analysis only. No architecture change, no retrieval change.

---

## Verdict

**Partially over-engineered — but not in the abstraction layer.**

Layering in this codebase is clean and largely justified. The waste is
**dead code**: modules with zero inbound imports that inflate statistics and,
in two cases, cannot even be imported.

| Metric | Before | After archive | Delta |
|---|---:|---:|---:|
| `src/` LOC (excl. `__init__.py`) | 5,685 | 4,967 | **−718 (−12.6%)** |
| `src/` module count | 25 | 21 | −4 |
| Classes | 32 | 30 | −2 |
| Functions | 176 | 162 | −14 |
| `scripts/` LOC | 6,614 | 6,614 | 0 |
| `tests/` LOC | 1,554 | 1,554 | 0 |
| Test-to-src ratio | 0.27 | 0.31 | +0.04 |

---

## 1. Dead Code — archived

Four modules had **0 inbound imports** across `src/`, `scripts/`, `tests/`,
`demo.py`. Moved to `archive/legacy_2026_10/`. Nothing deleted — recoverable
from git.

| File | LOC | Reason dead | Importable? |
|---|---:|---|---|
| `evaluation/evaluator.py` | 268 | Legacy Ollama judge, superseded by `ragas_evaluator.py` (Groq) | ❌ `ModuleNotFoundError: ollama` |
| `ingestion/dataset_builder.py` | 220 | Superseded by `scripts/build_retrieval_eval_dataset.py` + `evaluation/gold_matching.py` | Yes, unused |
| `retrieval/embedder.py` | 127 | Legacy FAISS; `retrieval_backend` defaults to `QDRANT` | ❌ `ModuleNotFoundError: faiss` |
| `retrieval/bm25_retriever.py` | 103 | **Duplicate** BM25 implementation | Yes, unused |

**FACT.** Two of the four hard-import packages absent from `requirements.txt`
(`ollama`, `faiss`). They were unimportable and unrunnable in this environment.

**INFERENCE.** Their presence was a live trap: a new contributor finding
`embedder.py` would hit an `ImportError` and could not tell whether the cause was
a missing dependency or a broken file.

### BM25 duplication detail

`bm25_retriever.py` exposed `tokenize`, `build_bm25_index`, `save_bm25`,
`load_bm25`, `bm25_search`. The canonical pipeline builds its index inline in
`MasterHybridRetriever.__init__` (~line 120) and searches via `_bm25_search`
(line 308), exposed publicly as `bm25_search()` (line 325) per the Phase-1
canonical interface refactor.

Two implementations of one retrieval mode directly violates the "one canonical
pipeline" rule established in Phase 1.

---

## 2. Deliberately NOT archived

Static import scanning cannot distinguish dead code from **entrypoints**. Two
zero-import modules are live:

| File | LOC | Why kept |
|---|---:|---|
| `src/api/main.py` | 311 | ASGI entrypoint — `uvicorn src.api.main:app`. Verified: imports cleanly, returns `FastAPI`. Referenced by `scripts/coverage_metric.py:368`. |
| `src/ingestion/semantic_chunker.py` | 222 | CLI entrypoint — `Makefile:16` executes it. Imported by `tests/test_semantic_chunker.py` via `importlib.import_module`. |

**FACT.** Deleting either would break `make ingest` or the API server.

---

## 3. `PolicyEngine` duplication — DEFERRED to Phase K

Not touched. Touching it alters the runtime path.

| Observation | Evidence |
|---|---|
| `resolve()` (line 118) and `initial_plan()` (line 127) compute the same four sub-results | Both call `decide_query_type` → `retrieval_strategy` → `reference_policy` → `generation_policy` |
| `resolve()` has 1 caller outside the class | `scripts/run_single_query.py:316` |
| `PlanHints.allow_references` — 0 reads in `agent.py` | grep `plan.allow_references` → 0 |
| `PlanHints.max_ref_ratio` — 0 reads in `agent.py` | grep `plan.max_ref_ratio` → 0 |
| `PlanHints.query_type`, `preferred_strategy`, `expected_complexity` — 0 reads | grep → 0 |

**OBSERVATION.** `allow_references` / `max_ref_ratio` are not dead — they are
*designed but never wired*. `chunk_filters.reference_policy_allows(section,
allow_references)` accepts the flag, and `is_generation_eligible()` accepts it,
but `agent.py` never passes a value. The reference-handling policy is
unreachable from the orchestration layer.

**INFERENCE.** `initial_plan()` is load-bearing (`agent.py` depends on
`top_k_hint`, `generation_hint`, `allow_multi_query`, `allow_hyde`).
`resolve()` is redundant with it and used by exactly one script.

**UNKNOWN.** Whether reference policy is *intentionally* disabled or
*accidentally* disconnected cannot be determined from code alone.

### Phase K preconditions (required before any change)

1. Freeze and record the baseline (`k1_baseline_freeze.json`).
2. Remove `resolve()`, or keep it and collapse into `initial_plan()`.
3. Decide the `allow_references` / `max_ref_ratio` question explicitly.
4. Re-run the baseline. Metrics must be **identical** — this is a
   no-behaviour-change refactor.
5. If any metric moves, the refactor is wrong. Revert.

---

## 4. `chunker.py` — dual entry point, NOT changed

| File | LOC | Imports |
|---|---:|---:|
| `src/ingestion/chunker.py` | 468 | 1 (`tests/test_chunker.py` only) |
| `src/ingestion/semantic_chunker.py` | 222 | 0 static, but is the `Makefile` entrypoint |

Both expose `save_chunks`. Both do reference detection and semantic chunking.

**INFERENCE.** Likely a v1/v2 generational split — `semantic_chunker.py` is the
production path, `chunker.py` the original. But `chunker.py` still carries a
test, so removing it would need a test decision, not just a file move.

**HYPOTHESIS.** `chunker.py` is legacy and its test is a regression guard for
chunking behaviour that `semantic_chunker.py` no longer shares.

**UNKNOWN.** No code or artifact establishes whether `chunker.py` is still
needed. Requires a decision, not an audit.

---

## 5. Not over-engineered

| Component | LOC | Assessment |
|---|---:|---|
| `hybrid_retriever.py` | 687 | Largest file, 29 methods. This is the canonical pipeline with per-stage trace instrumentation (stage ids, score rows, latency breakdown, filter breakdown). Size is earned. |
| `chunk_filters.py` | 345 | 21 functions. All 21 reachable — the 6 with zero *external* references are used internally within the file. |
| `generator.py` | 507 | Prompt contract, confidence parsing, abstention path. Justified. |
| `controller/` (4 files, 681) | | Hand-rolled orchestration. Explicitly aligned with `PROJECT_GOALS.md` §9 Non-Goals. |
| `evaluation/gold_matching.py` | 57 | Small, single-purpose. Good. |
| `evaluation/gold_data_validator.py` | 181 | Gold integrity enforcement. Required by the evidence-first rule. |

**Unidirectional layering** — `ingestion → retrieval → generation`, with
`controller` orchestrating and `evaluation` observing. No circular imports
found. `retrieval/chunk_filters.py` uses `Any` for chunk input rather than
importing concrete types, which is why it stays decoupled.

---

## 6. `scripts/` is the largest surface

6,614 LOC across ~40 files — **larger than all of `src/`**. This is diagnostic
tooling from Phases A–J, and it is the actual output of that work.

**INFERENCE.** This is appropriate for a research repo in the middle of an
evidence-gathering campaign. It would become noise if the project moved to
production maintenance. Not flagged as a defect.

---

## Verification after archiving

| Check | Result |
|---|---|
| Test suite | 87 passed, 3 failed — identical to pre-archive |
| Residual imports of archived modules | 0 |
| `src/` import graph | unchanged |
| Retrieval parameters | untouched |
| Models | untouched |
| Gold labels | untouched |

The 3 failures are pre-existing: `numpy 2.5.3` on Python 3.14 raises
`ImportError: cannot load module more than once per process` when
`test_qdrant_store.py` and `test_semantic_chunker.py` both load `numpy.fft` in
one pytest process. Same class of bug as the protobuf `upb` issue handled by
`src/_compat.py`. Unrelated to this audit.

---

## Remaining UNKNOWNs

1. Is `chunker.py` legacy, or a live regression baseline? — no artifact says.
2. Was reference policy intentionally disabled, or accidentally disconnected?
3. Do the 18 test files cover `chunk_filters.py`'s 21 functions? Not audited
   here.
4. Test-to-src ratio 0.31 is lowish, but the suite was verified deterministic
   and network-free, which is a deliberate trade from the Phase A–J discipline.

---

## Files Changed

| File | Action |
|---|---|
| `src/retrieval/bm25_retriever.py` | → `archive/legacy_2026_10/retrieval/` |
| `src/retrieval/embedder.py` | → `archive/legacy_2026_10/retrieval/` |
| `src/evaluation/evaluator.py` | → `archive/legacy_2026_10/evaluation/` |
| `src/ingestion/dataset_builder.py` | → `archive/legacy_2026_10/ingestion/` |
| `archive/legacy_2026_10/README.md` | new — provenance + restore instructions |

**Zero code logic changed.** This was a file relocation only.

*Next: Phase K — `PolicyEngine` deduplication, gated on baseline freeze.*