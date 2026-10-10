# E5 — Latency Budget Analysis

## Current Post-Fix Measurements

Source: `results/metrics/reranker_latency_profile.json`

| metric | value |
|---|---:|
| n_queries | 14 |
| avg_candidate_count | 20.0 |
| candidate_count_variance | 0 |
| avg_total_candidate_text_tokens | 2901.5 |
| median_total_candidate_text_tokens | 2828.5 |
| max_candidate_text_length | 3673 |
| p50_reranker_ms | 3099.63 |
| p95_reranker_ms | 4101.08 |
| min_reranker_ms | 2823.24 |
| max_reranker_ms | 4101.08 |
| p50_total_ms | 3168.16 |
| p95_total_ms | 4158.28 |

## Budget Analysis

| metric | value |
|---|---:|
| P95 target | 3500 ms |
| total p95 | 4158.28 ms |
| budget_excess_ms | 658.28 |
| reranker_latency_ratio | 0.9862 |
| latency_budget_remaining | -658.28 ms |

## Conclusion

**P95_BUDGET_MISSED**

The current pipeline exceeds the 3.5 s P95 target by 658.28 ms.

**Wording:** "Reranker is the dominant measured retrieval-stage latency component and the current pipeline exceeds the 3.5 s P95 target."

**Note:** This does NOT claim reranker is the only heavy function in the whole system. Broader application profiling would be needed for that claim.

## Correlation Analysis

- `corr_reranker_ms_vs_candidate_count`: **null** (not estimable — candidate count is constant at 20 for all 14 queries)
- `corr_reranker_ms_vs_candidate_tokens`: -0.2286 (descriptive only, n=14, no causal claim)

**Note:** At n=14, correlation is descriptive evidence only. No significance claim is made.
