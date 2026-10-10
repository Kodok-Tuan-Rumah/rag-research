# Fase J — Evaluation Gate → Dataset Expansion → Confidence Readiness

## 1. Gate Status

**J0_GATE = PASS**
**J7_GATE = PASS**

All evaluation contracts are consistent. All metrics are reproducible. Gold provenance is intact. Dataset limitations are quantified. System identity is documented.

## 2. Current Evaluation Contract

| Metric | Definition | Verified |
|---|---|---|
| Recall@10 | gold retrieved in top-10 / total valid gold queries | YES |
| RR@10 | 1/rank if 1<=rank<=10, else 0 | YES |
| MRR@10 | mean(RR@10) | YES |

**Bug found and documented:** `retrieval_stage_eval_postfix.json` uses unrestricted 1/rank for MRR@10. Correct value for RRF MRR@10 = **0.8571** (not 0.8666). Fase F ablation uses correct RR@10 cutoff.

## 3. Dataset Expansion Feasibility

| metric | value |
|---|---:|
| Source rows inspected | 84 |
| Exact-span matches | 2 |
| Token-recall matches | 8 |
| Evaluation row only | 74 |
| Duplicates removed | 0 |
| Valid non-overlapping additions | **0** |

**Conclusion:** N=14 cannot be defensibly expanded from existing artifacts. New gold-labeled data must be created from additional evidence sources.

## 4. Confidence Calibration Readiness

| metric | value |
|---|---:|
| N | 14 |
| Positive samples | 12 |
| Negative samples | 2 |
| Confidence range | [0.0, 1.0] |

**Classification: CALIBRATION_EXPLORATORY_ONLY**

N=14 with only 2 negative samples is insufficient for production isotonic regression. Current status: **BLOCKED_BY_SAMPLE_SIZE**.

## 5. Statistical Sufficiency

| Test | Result |
|---|---|
| McNemar exact p-value | 0.5000 |
| Significant at α=0.05? | NO |
| BM25 95% CI | [0.7847, 1.0000] |
| RRF 95% CI | [0.6006, 0.9599] |
| CIs overlap? | YES |

**Conclusion:** N=14 is insufficient for definitive architecture decisions. The dataset cannot test RRF's hybrid value (0 dense-only gold cases).

## 6. Reranker Decision Support

| metric | value |
|---|---:|
| top10_improvement_rate | 0.0 |
| top10_crossing_harm_rate | 0.0 |
| within_top10_demotion_count | 2 (q11, q20) |
| mrr10_loss_count | 2 (q11, q20) |
| mrr10_gain_count | 0 |
| net_improvement | 0 |
| reranker_value_ratio | 0.0 |
| Recall@10 delta | 0 |
| MRR@10 delta | -0.0982 |
| P50 latency delta | +3122.85 ms |
| P95 latency delta | +4002.20 ms |

**Classification: LOW_VALUE_ON_CURRENT_SAMPLE**

**Note:** This is NOT a "REMOVE" decision. It provides diagnostic evidence for a later controlled ablation on a larger dataset.

## 7. Latency Decision Support

| metric | value |
|---|---:|
| P95 target | 3500 ms |
| B total p95 | 4065.16 ms |
| budget_remaining | -565.16 ms |
| reranker_latency_ratio | 0.9862 |

**Conclusion: P95_BUDGET_MISSED**

## 8. Documentation State

| README Claim | Current State | Classification |
|---|---|---|
| 982 chunks | 796 chunks | LEGACY |
| BGE Reranker | ms-marco-MiniLM | LEGACY |
| llama-3.3-70b | gpt-oss-20b | LEGACY |
| 55 QA pairs | 14 canonical queries | LEGACY |
| Python 3.12 | Python 3.14 | LEGACY |
| RAGAS 5-metric | Retrieval-only eval | LEGACY |

## 9. Architectural Evidence

**All architectural value assessments are restricted to the current 14-query canonical sample.**

| Component | Classification | Reason |
|---|---|---|
| BM25 | HIGH_VALUE_ON_CURRENT_SAMPLE | Recall@10 = 1.0, low latency |
| Dense | UNCERTAIN | 0 dense-only gold cases — hybrid value untestable |
| RRF | LOW_VALUE_ON_CURRENT_SAMPLE | Recall@10 < BM25, MRR@10 < BM25 |
| Reranker | LOW_VALUE_ON_CURRENT_SAMPLE | No Recall gain, MRR degradation, high latency |

**Allowed statements:**
- "BM25 is the strongest observed baseline on the current canonical sample."
- "RRF has LOW observed value on this sample."
- "Reranker has LOW observed value on this sample."
- "Dense-only contribution is currently untested because no dense-only gold cases exist."

**NOT allowed:**
- "BM25 is universally superior"
- "Dense retrieval is universally ineffective"
- "RRF is universally unnecessary"
- "Reranker is universally useless"

## 10. Remaining Unknowns

1. Whether RRF would add value on a larger dataset with dense-only gold cases.
2. Whether the reranker's demotion of q20 (RRF 1 → reranker 8) reflects true cross-encoder judgment.
3. Whether q15's reranker rank 19 is correct given its weak relevance (0.25).
4. The q06 candidate-depth boundary is a design choice, not a bug.
5. Whether a different dense model would change the conclusion (untestable here).

## 11. Next Controlled Experiment

The next phase, if authorized, should be a **controlled RRF ablation** based on the evidence established here:

- Compare BM25-only vs RRF on a larger dataset
- Include dense-only gold cases to test RRF's hybrid value
- Measure both retrieval and generation quality
- Include latency and cost analysis

**This is NOT a tuning instruction. It is a diagnostic recommendation for future work.**
