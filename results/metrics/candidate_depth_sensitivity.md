# E3 — Candidate Depth Sensitivity

## RRF Recall@k

| k | RRF Recall@k |
|---:|---:|
| 10 | 0.8571 |
| 15 | 0.9286 |
| 20 | 0.9286 |
| 25 | 1.0000 |
| 30 | 1.0000 |
| 40 | 1.0000 |

## First Appearance Groups

| rank range | count | query_ids |
|---|---:|---|
| 1–10 | 12 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| 11–20 | 1 | q15 |
| 21–30 | 1 | q06 |
| 31–40 | 0 | — |
| beyond 40 | 0 | — |

## q06 Status

- **RRF rank:** 24
- **Current rerank depth:** 20
- **Status:** BEYOND_CURRENT_RERANK_DEPTH
- **Classification:** Candidate-depth boundary case, NOT a reranker failure.

**Note:** Whether rerank depth should be larger is a future tuning decision and remains OUT OF SCOPE.
