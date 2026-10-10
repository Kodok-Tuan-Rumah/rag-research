# I3 — Reranker Necessity Revisited

## Three Separate Concepts

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
| q11 | 1 | 2 | 1.0 | 0.5 | **-0.5** |
| q20 | 1 | 8 | 1.0 | 0.125 | **-0.875** |
| q15 | 11 | 19 | 0 | 0 | 0 |

**MRR@10 losses:** q11 (-0.5), q20 (-0.875)
**MRR@10 gains:** None
**MRR@10 delta:** -0.0982 (aggregate)

## Summary

| metric | value |
|---|---:|
| top10_improvement_rate | 0.0 |
| top10_crossing_harm_rate | 0.0 |
| within_top10_demotion_count | 2 (q11, q20) |
| mrr10_loss_count | 2 (q11, q20) |
| mrr10_gain_count | 0 |
| mrr10_delta | -0.0982 |

## Interpretation

**FACT:** The reranker causes no top-10 crossing harm in this sample (0 queries pushed from ≤10 to >10).

**FACT:** The reranker causes within-top-10 rank demotions (q11: 1→2, q20: 1→8) resulting in negative MRR@10 delta.

**INFERENCE:** The reranker does not provide measurable value on this dataset. It degrades MRR@10 without improving Recall@10, at a latency cost of ~3.1 s p50.

**Note:** q15's demotion (11→19) is NOT an MRR@10 loss because both positions are outside top-10 (RR@10 = 0 for both).
