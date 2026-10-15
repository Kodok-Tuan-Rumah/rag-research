# J6 — Documentation Drift Audit

> **Historical audit snapshot (2026-10-08).** The “Current System Identity”
> and action list below describe the state at that audit date; they are not a
> current inventory. README corrections in that list have since been applied.
> Current Phase 1 acceptance status is tracked in
> [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md) and
> [`results/metrics/phase1_smoke.json`](phase1_smoke.json).

## README vs Current State

| Claim in README | Current State | Classification |
|---|---|---|
| 982 chunks | 796 chunks (v2 corpus) | LEGACY |
| BGE Reranker (bge-reranker-base) | cross-encoder/ms-marco-MiniLM-L-6-v2 | LEGACY |
| llama-3.3-70b-versatile | openai/gpt-oss-20b | LEGACY |
| 55 clean QA pairs | 14 canonical queries | LEGACY |
| Python 3.12 | Python 3.14.4 | LEGACY |
| 982 vectors in Qdrant | 796 vectors | LEGACY |
| RAGAS 5-metric eval | Retrieval-only eval (no LLM) | LEGACY |
| Hybrid RRF + BGE Reranker | Hybrid RRF + ms-marco reranker | LEGACY |

## System Identity at Audit Time (verified from code/artifacts)

| Component | Current Value |
|---|---|
| Corpus | 796 chunks (`chunks_semantic_v2.json`) |
| Qdrant vectors | 796 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generator model | openai/gpt-oss-20b |
| Python version | 3.14.4 |
| Canonical dataset | 14 queries |
| Validated expansion rows | 10 (subset of canonical 14) |
| Train/holdout split | 8/6 (derived from canonical 14) |

## Original Action List — Superseded

The following actions were written for the 2026-10-08 README and are retained
for audit history. They should not be used as a current TODO list:

- 796 chunks (not 982)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (not BGE)
- openai/gpt-oss-20b (not llama-3.3-70b)
- 14 canonical queries (not 55)
- Python 3.14 (not 3.12)
- Retrieval-only eval scope (not RAGAS 5-metric)

The README now labels the old values as historical/legacy. Historical claims
remain in place for provenance rather than being deleted.
