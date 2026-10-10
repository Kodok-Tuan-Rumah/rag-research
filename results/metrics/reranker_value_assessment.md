# E4 — Reranker Value Assessment

## Rank Outcome Taxonomy

| classification | count | queries | definition |
|---|---:|---|---|
| IMPROVED | 0 | — | RRF rank > 10 AND reranker rank ≤ 10 |
| HARMED | 0 | — | RRF rank ≤ 10 AND reranker rank > 10 |
| DEMOTED_BUT_TOP10 | 2 | q11, q20 | RRF rank ≤ 10 AND reranker rank ≤ 10 AND reranker rank numerically worse |
| OUTSIDE_TOP10_BOTH | 1 | q15 | RRF rank > 10 AND reranker rank > 10 |
| UNCHANGED | 10 | q01,q02,q03,q04,q09,q10,q13,q14,q16,q19 | same valid rank |
| BEYOND_RERANK_DEPTH | 1 | q06 | gold was not passed to reranker |

## MRR@10 Effect

| query_id | RRF RR@10 | Reranked RR@10 | Delta | Note |
|---|---:|---:|---:|---|
| q20 | 1.0 | 0.125 | **-0.875** | RRF 1 → reranker 8 |
| q11 | 1.0 | 0.5 | **-0.5** | RRF 1 → reranker 2 |
| q06 | 0.0417 | 0 | -0.0417 | RRF 24 → not in reranker |
| q15 | 0.0909 | 0.0526 | -0.0383 | RRF 11 → reranker 19 |
| All others | — | — | 0 | no change |

**MRR@10 losses:** q20 (-0.875), q11 (-0.5), q06 (-0.0417), q15 (-0.0383)
**MRR@10 gains:** None
**MRR@10 neutral:** q01, q02, q03, q04, q09, q10, q13, q14, q16, q19

## Summary

| metric | value |
|---|---:|
| net_improvement | 0 |
| reranker_value_ratio | 0.0 |
| total_queries | 14 |
| Recall@10 RRF | 0.8571 |
| Recall@10 Reranked | 0.8571 |
| MRR@10 RRF | 0.8666 |
| MRR@10 Reranked | 0.7627 |

## Latency Measurement

**NO_DIRECT_MEASUREMENT**

No standalone no-reranker latency was measured in this phase. The current reranker-enabled latency is reported separately in E5. Timing boundaries have not been proven additive, so no subtraction is performed.
