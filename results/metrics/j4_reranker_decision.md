# J4 — Reranker Decision Support

## Evidence (from Fase F controlled A/B)

| metric | A (no reranker) | B (reranker) | value |
|---|---:|---:|---|
| Recall@10 | 0.8571 | 0.8571 | 0 |
| MRR@10 | 0.8571 | 0.7589 | **-0.0982** |
| P50 latency | 48.43 ms | 3171.28 ms | +3122.85 ms |
| P95 latency | 62.96 ms | 4065.16 ms | +4002.20 ms |

## Three Concepts Separated

### A. Top-10 Crossing Improvement (RRF >10 → reranker ≤10)

| metric | value |
|---|---:|
| count | 0 |
| rate | 0.0 |

**No query was brought into top-10 by the reranker.**

### B. Top-10 Crossing Harm (RRF ≤10 → reranker >10)

| metric | value |
|---|---:|
| count | 0 |
| rate | 0.0 |

**No query was pushed out of top-10 by the reranker.**

### C. Within-Top-10 Demotion / MRR@10 Loss

| query_id | RRF rank | Reranker rank | RRF RR@10 | Reranker RR@10 | Delta |
|---|---:|---:|---:|---:|---:|
| q20 | 1 | 8 | 1.0 | 0.125 | **-0.875** |
| q11 | 1 | 2 | 1.0 | 0.5 | **-0.5** |
| q15 | 11 | 19 | 0 | 0 | 0 |

**MRR@10 losses:** q20 (-0.875), q11 (-0.5)
**MRR@10 gains:** None
**MRR@10 neutral:** q15 (both outside top-10)

## Latency

| metric | A (no reranker) | B (reranker) | Delta |
|---|---:|---:|---:|
| P50 | 48.43 ms | 3171.28 ms | +3122.85 ms |
| P95 | 62.96 ms | 4065.16 ms | +4002.20 ms |

**Latency measurement:** Directly measured end-to-end retrieval latency per query. Both A and B measured with same methodology.

## Classification

**RERANKER_CURRENT_SAMPLE_VALUE: LOW**

**Reasoning:**
- No Recall@10 improvement (0.8571 → 0.8571)
- Negative MRR@10 delta (-0.0982)
- High latency cost (+3.1 s p50)
- 2 within-top-10 demotions (q11, q20)

**Note:** This classification is for the current 14-query canonical sample only. It does NOT constitute a "REMOVE" decision. It provides diagnostic evidence for a later controlled ablation on a larger dataset.
