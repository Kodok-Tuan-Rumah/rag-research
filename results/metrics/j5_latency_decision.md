# J5 — Latency Decision Support

## Current Post-Fix Measurements

Source: `results/metrics/fase_f_reranker_ablation.json`

| metric | A (no reranker) | B (reranker) |
|---|---:|---:|
| P50 | 48.43 ms | 3171.28 ms |
| P95 | 62.96 ms | 4065.16 ms |

## Budget Analysis

| metric | value |
|---|---:|
| P95 target | 3500 ms |
| B total p95 | 4065.16 ms |
| budget_remaining | -565.16 ms |
| reranker_latency_ratio | 0.9862 |
| latency_budget_remaining | -565.16 ms |

## Conclusion

**P95_BUDGET_MISSED**

The current pipeline exceeds the 3.5 s P95 target by 565.16 ms.

**Wording:** "Reranker is the dominant measured retrieval-stage latency component and the current pipeline exceeds the 3.5 s P95 target."

**Note:** This does NOT claim reranker is the only heavy function in the whole system. Broader application profiling would be needed for that claim.
