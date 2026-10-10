# Fase E — Reranker Behavior Analysis & Candidate Depth Study

## 1. Executive Summary

**Proven current facts:**

- RRF Recall@10 = 0.8571, MRR@10 = 0.8666
- RRF+Reranker Recall@10 = 0.8571, MRR@10 = 0.7627
- Reranker does NOT improve Recall@10 on the corrected 14-query dataset.
- Reranker does NOT improve MRR@10; it degrades MRR@10 by -0.1039.
- The MRR degradation is driven by q20 (-0.875), q11 (-0.5), and q06 (-0.0417).
- Only 1 query (q06) is outside rerank depth (RRF rank 24 > 20).
- q15 is JUDGED_CORRECT (gold relevance 0.25, no top-10 chunk strongly relevant).
- q20 is AMBIGUOUS (gold relevance 0.75, but rs_0086 is also directly about IRR).
- Current pipeline exceeds the 3.5 s P95 latency target by 658.28 ms.

## 2. Reranker Score Distribution

### Within-Query Findings

**FACT:** 12/13 gold chunks that entered the reranker received positive scores. Only q15 received a negative score (-8.475).

**FACT:** q06 did not enter the reranker (RRF rank 24 > depth 20).

**OBSERVATION:** Gold chunks that enter the reranker typically score in the top percentile of their candidate set (100th percentile for 12/13 queries).

**OBSERVATION:** q15 is the only query where gold received a negative reranker score and the lowest percentile (0.0).

**OBSERVATION:** q20 gold score (-5.579) is below the candidate mean (-6.158) but above the median (-6.857), placing it at the 12.5th percentile.

### Cross-Query Descriptive Statistics

| statistic | value |
|---|---:|
| mean of per-query means | -3.847 |
| median of per-query medians | -4.348 |
| mean of per-query stds | 3.872 |
| min gold score | -8.475 (q15) |
| max gold score | 8.478 (q10) |

**Note:** Cross-query score comparisons are descriptive only. Scores are not calibrated probabilities and should not be compared across queries as relevance magnitudes.

## 3. q20 Deep Dive

**Query:** "How is the inter-rater reliability (IRR) measured among annotators for safety assessments, and what is the average IRR score for Llama 2-Chat annotations?"

**Gold chunk:** `tier3_llama2_rs_0152`
- RRF rank = 1, rerank input rank = 1, reranker rank = 8, final rank = 8
- Reranker score = -5.579

**Top-10 Reranked Competitors:**

| rank | chunk_id | score | relevance |
|---:|---|---:|---:|
| 1 | tier3_llama2_rs_0086 | 4.319 | 0.75 |
| 2 | tier3_llama2_rs_0085 | 2.930 | 0.25 |
| 3 | tier3_llama2_rs_0151 | -3.725 | 0.50 |
| 4 | tier3_llama2_rs_0082 | -4.629 | 0.25 |
| 5 | tier3_llama2_rs_0036 | -5.073 | 0.25 |
| 6 | tier3_llama2_rs_0113 | -5.181 | 0.25 |
| 7 | tier3_llama2_rs_0038 | -5.475 | 0.25 |
| 8 | tier3_llama2_rs_0152 (gold) | -5.579 | 0.75 |
| 9 | tier3_llama2_rs_0013 | -6.111 | 0.00 |
| 10 | tier3_llama2_rs_0332 | -6.480 | 0.00 |

**Manual Judgment:** AMBIGUOUS

**Reasoning:** Gold chunk is strongly relevant (directly about IRR measurement for safety assessments). However, rs_0086 (rank 1) is also directly about IRR. The reranker demoted gold from rank 1 to 8, but both chunks are relevant. This is not a clear reranker error.

## 4. Candidate Depth Sensitivity

| k | RRF Recall@k |
|---:|---:|
| 10 | 0.8571 |
| 15 | 0.9286 |
| 20 | 0.9286 |
| 25 | 1.0000 |
| 30 | 1.0000 |
| 40 | 1.0000 |

**First appearance groups:**

| rank range | count | query_ids |
|---|---:|---|
| 1–10 | 12 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| 11–20 | 1 | q15 |
| 21–30 | 1 | q06 |
| 31–40 | 0 | — |
| beyond 40 | 0 | — |

**q06 status:** RRF rank = 24, current rerank depth = 20. Classification: BEYOND_CURRENT_RERANK_DEPTH. This is a candidate-depth boundary case, NOT a reranker failure.

## 5. Reranker Value Assessment

### Recall Effect

| Configuration | Recall@10 |
|---|---:|
| RRF | 0.8571 |
| RRF + Reranker | 0.8571 |

**No change.**

### MRR@10 Effect

| Configuration | MRR@10 |
|---|---:|
| RRF | 0.8666 |
| RRF + Reranker | 0.7627 |

**Degradation: -0.1039**

### Rank Movement

| classification | count | queries |
|---|---:|---|
| IMPROVED | 0 | — |
| HARMED | 0 | — |
| DEMOTED_BUT_TOP10 | 2 | q11, q20 |
| OUTSIDE_TOP10_BOTH | 1 | q15 |
| UNCHANGED | 10 | q01,q02,q03,q04,q09,q10,q13,q14,q16,q19 |
| BEYOND_RERANK_DEPTH | 1 | q06 |

### MRR@10 Decomposition

| query_id | RRF RR@10 | Reranked RR@10 | Delta |
|---|---:|---:|---:|
| q20 | 1.0 | 0.125 | **-0.875** |
| q11 | 1.0 | 0.5 | **-0.5** |
| q06 | 0.0417 | 0 | -0.0417 |
| q15 | 0.0909 | 0.0526 | -0.0383 |
| All others | — | — | 0 |

**Largest negative deltas:** q20 (-0.875), q11 (-0.5), q06 (-0.0417)
**Largest positive deltas:** None

### Latency Measurement Limitations

**NO_DIRECT_MEASUREMENT**

No standalone no-reranker latency was measured in this phase. Timing boundaries have not been proven additive, so no subtraction is performed.

## 6. Latency Budget Analysis

| metric | value |
|---|---:|
| P95 target | 3500 ms |
| total p95 | 4158.28 ms |
| budget_excess_ms | 658.28 |
| reranker_latency_ratio | 0.9862 |
| latency_budget_remaining | -658.28 ms |

**Conclusion:** P95_BUDGET_MISSED

**Wording:** "Reranker is the dominant measured retrieval-stage latency component and the current pipeline exceeds the 3.5 s P95 target."

## 7. Diagnostic Recommendations

The following are NOT tuning instructions:

- Perform controlled ablation to quantify reranker value on this dataset.
- Review reranker failure cases (q11, q15, q20) with full chunk text.
- Verify candidate-depth tradeoff with controlled experiments.
- Profile broader application latency to identify non-retrieval bottlenecks.

## 8. Mandatory Final Interpretation

### Rank Demotions

- q11: RRF 1 → reranker 2 (still in top-10)
- q15: RRF 11 → reranker 19 (outside top-10)
- q20: RRF 1 → reranker 8 (still in top-10)

### Current MRR@10 Losses

- q20: -0.875 (RRF 1 → reranker 8)
- q11: -0.5 (RRF 1 → reranker 2)
- q06: -0.0417 (RRF 24 → not in reranker)
- q15: -0.0383 (RRF 11 → reranker 19)

### Current Recall@10 Improvements

None. RRF Recall@10 = Reranked Recall@10 = 0.8571.

### Gold Beyond Reranker Depth

1 query: q06 (RRF rank 24 > depth 20).

### q15 Assessment

**JUDGED_CORRECT.** Gold relevance = 0.25 (weakly relevant). No top-10 competitor is strongly relevant. The reranker demotion is semantically plausible.

### q20 Assessment

**AMBIGUOUS.** Gold relevance = 0.75 (strongly relevant). Top competitor rs_0086 is also directly about IRR (0.75). The reranker demotion is not clearly wrong.

### Dominant Current Failure Mechanism

**RERANKER_DEMOTION** (3 queries: q11, q15, q20).

## 9. Remaining Unknowns

1. Whether the reranker's demotion of q20 (RRF 1 → reranker 8) reflects true cross-encoder judgment requires deeper manual review of full chunk texts.
2. Whether q15's reranker rank 19 is correct depends on whether the gold chunk's weak relevance (0.25) justifies demotion below other chunks.
3. The q06 candidate-depth boundary is a design choice, not a bug. Whether to increase rerank depth is OUT OF SCOPE.
4. No direct no-reranker latency measurement was performed; latency decomposition relies on stage-level traces.
