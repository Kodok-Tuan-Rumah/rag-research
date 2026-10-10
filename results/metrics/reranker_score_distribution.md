# E1 — Reranker Score Distribution

## Per-Query Statistics

| query_id | gold_score | mean | median | std | min | max | gold_percentile | margin_vs_rank1 | margin_vs_top10_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q01 | 6.137 | -6.545 | -8.578 | 4.271 | -12.834 | 6.137 | 100.0 | +18.971 | +12.022 |
| q02 | 7.293 | -0.669 | -0.503 | 3.025 | -7.338 | 7.293 | 100.0 | +14.631 | +7.962 |
| q03 | 5.844 | -3.792 | -4.348 | 3.547 | -10.318 | 5.844 | 100.0 | +16.162 | +9.636 |
| q04 | -2.037 | -9.004 | -9.924 | 2.584 | -13.089 | -2.037 | 100.0 | +11.052 | +6.967 |
| q06 | — | -2.399 | -3.312 | 2.783 | -7.338 | 3.510 | — | — | — |
| q09 | 6.126 | -1.954 | -1.538 | 4.446 | -10.318 | 6.126 | 100.0 | +16.444 | +8.080 |
| q10 | 8.478 | -3.137 | -2.053 | 5.605 | -12.834 | 8.478 | 100.0 | +21.312 | +11.615 |
| q11 | 8.023 | -4.139 | -6.738 | 6.214 | -13.089 | 8.023 | 100.0 | +21.111 | +12.161 |
| q13 | 5.893 | -1.624 | -1.888 | 3.254 | -8.188 | 5.893 | 100.0 | +14.081 | +7.517 |
| q14 | 6.726 | -1.862 | -1.581 | 3.387 | -8.560 | 6.726 | 100.0 | +15.286 | +8.588 |
| q15 | -8.475 | -5.527 | -5.771 | 2.164 | -10.318 | 0.480 | 0.0 | +8.955 | +2.948 |
| q16 | 5.850 | -6.361 | -7.353 | 4.327 | -13.089 | 5.850 | 100.0 | +19.211 | +12.211 |
| q19 | 7.394 | -7.100 | -8.971 | 5.020 | -13.089 | 7.394 | 100.0 | +20.465 | +14.494 |
| q20 | -5.579 | -6.158 | -6.857 | 3.872 | -10.600 | 4.319 | 12.5 | +9.898 | +0.579 |

## Observations

**FACT:** 12/13 gold chunks that entered the reranker received positive scores. Only q15 received a negative score (-8.475).

**FACT:** q06 did not enter the reranker (RRF rank 24 > depth 20).

**OBSERVATION:** Gold chunks that enter the reranker typically score in the top percentile of their candidate set (100th percentile for 12/13 queries).

**OBSERVATION:** q15 is the only query where gold received a negative reranker score and the lowest percentile (0.0).

**OBSERVATION:** q20 gold score (-5.579) is below the candidate mean (-6.158) but above the median (-6.857), placing it at the 12.5th percentile.

## Cross-Query Descriptive Statistics

| statistic | value |
|---|---:|
| mean of per-query means | -3.847 |
| median of per-query medians | -4.348 |
| mean of per-query stds | 3.872 |
| min gold score | -8.475 (q15) |
| max gold score | 8.478 (q10) |

**Note:** Cross-query score comparisons are descriptive only. Scores are not calibrated probabilities and should not be compared across queries as relevance magnitudes.
