# Fase I — Dense Retrieval Failure Analysis & Dataset Sufficiency Audit

## 1. Executive Summary

**Proven current facts:**

- Dense retrieval misses q06 and q15 entirely (gold not in dense top-40).
- Both are BM25-only gold cases (BM25 finds gold, dense does not).
- RRF demotes both because dense misses leave BM25-only hits underweighted.
- Reranker provides no Recall@10 improvement and degrades MRR@10.
- N=14 is insufficient for definitive architecture decisions.
- 0 dense-only gold cases exist in this sample — RRF's hybrid value is untestable here.

**Scope:** All architectural value assessments are restricted to the current 14-query canonical sample.

## 2. Dense Retrieval Failure Analysis

### q06

| field | value |
|---|---|
| Query | "How does RAG compare to Fine-tuning (FT) in terms of its suitability for specific tasks?" |
| Word count | 15 |
| Keyword density | 0.733 |
| Domain specificity | high |
| Linguistic structure | comparative |
| Dense rank | None (miss) |
| Dense top-1 score | 0.684 |
| Gold in dense top-40 | No |
| Classification | DENSE_MISS |

### q15

| field | value |
|---|---|
| Query | "What characteristic of RAG allows it to achieve strong results without requiring expensive pre-training?" |
| Word count | 14 |
| Keyword density | 0.786 |
| Domain specificity | high |
| Linguistic structure | causal |
| Dense rank | None (miss) |
| Dense top-1 score | 0.741 |
| Gold in dense top-40 | No |
| Classification | DENSE_MISS |

### Comparison with Dense-Success Queries

| query_id | words | kw_density | dense_rank | dense_score |
|---|---:|---:|---:|---:|
| q01 | 28 | 0.643 | 1 | 0.780 |
| q02 | 22 | 0.818 | 1 | 0.758 |
| q03 | 14 | 0.643 | 1 | 0.677 |
| q04 | 9 | 0.667 | 14 | 0.325 |
| q06 | 15 | 0.733 | — | — |
| q09 | 14 | 0.714 | 1 | 0.723 |
| q10 | 9 | 0.889 | 1 | 0.694 |
| q11 | 14 | 0.786 | 1 | 0.655 |
| q13 | 17 | 0.765 | 1 | 0.684 |
| q14 | 11 | 0.818 | 1 | 0.653 |
| q15 | 14 | 0.786 | — | — |
| q16 | 14 | 0.714 | 1 | 0.713 |
| q19 | 15 | 0.667 | 1 | 0.733 |
| q20 | 23 | 0.696 | 2 | 0.682 |

**OBSERVATION:** q06 and q15 are short (14-15 words), high keyword density (0.73-0.79), high domain specificity. These characteristics overlap with dense-success queries (q03, q11, q16). No clear linguistic pattern separates dense-miss from dense-success queries.

**HYPOTHESIS:** all-MiniLM-L6-v2 (general-purpose) may not capture domain-specific semantic relationships for RAG/FT comparison queries. This is a hypothesis, not a proven fact.

**UNKNOWN:** Whether a domain-specific dense model would improve q06/q15 retrieval.

### Source Counts

| category | count | queries |
|---|---:|---|
| BM25-only gold | 2 | q06, q15 |
| Dense-only gold | 0 | — |
| Both-source gold | 12 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| Neither | 0 | — |

## 3. Dataset Sufficiency

### Statistical Power

| metric | value |
|---|---:|
| N | 14 |
| BM25 Recall@10 | 1.0000 |
| RRF Recall@10 | 0.8571 |
| Observed difference | -2/14 |
| McNemar exact p-value | 0.5000 |
| Significant at α=0.05? | NO |

### Confidence Intervals (Wilson Score, 95%)

| Stage | Recall@10 | 95% CI |
|---|---:|---|
| BM25 | 1.0000 | [0.7847, 1.0000] |
| RRF | 0.8571 | [0.6006, 0.9599] |
| Dense | 0.7857 | [0.5241, 0.9243] |

**Note:** CIs overlap substantially. No population-level superiority claim can be made from this sample.

### Missing Evaluation Categories

| category | present? | count |
|---|---|---:|
| dense-only gold cases | NO | 0 |
| BM25-only gold cases | YES | 2 |
| both-source cases | YES | 12 |
| adversarial-only cases | NO | 0 |

**Critical gap:** 0 dense-only gold cases means RRF's value for dense coverage is **untestable** with the current dataset.

### Conclusion

**Is N=14 sufficient for a definitive architecture decision?**

**NO / LIMITED.**

The current dataset can answer:
- Does RRF hurt BM25-only gold cases? → YES (2/2 observed)
- Does RRF improve rank for both-source gold cases? → YES (q04: 5→1)

The current dataset **cannot** answer:
- Does RRF add value over BM25 for dense-only coverage? → UNTESTABLE (0 cases)
- Is the BM25 vs RRF difference statistically significant? → NO (McNemar p=0.50)
- Would a different dense model change the conclusion? → UNTESTABLE

## 4. Reranker Necessity

### Three Concepts Separated

| Concept | Count | Rate |
|---|---:|---:|
| Top-10 crossing improvement (RRF >10 → reranker ≤10) | 0 | 0.0 |
| Top-10 crossing harm (RRF ≤10 → reranker >10) | 0 | 0.0 |
| Within-top-10 demotion (MRR@10 loss) | 2 (q11, q20) | — |

### MRR@10 Decomposition

| query_id | RRF RR@10 | Reranked RR@10 | Delta |
|---|---:|---:|---:|
| q20 | 1.0 | 0.125 | **-0.875** |
| q11 | 1.0 | 0.5 | **-0.5** |
| q15 | 0 | 0 | 0 |
| All others | — | — | 0 |

**MRR@10 losses:** q20 (-0.875), q11 (-0.5)
**MRR@10 gains:** None
**MRR@10 delta:** -0.0982 (aggregate)

### Latency

| metric | A (no reranker) | B (reranker) | Delta |
|---|---:|---:|---:|
| P50 | 48.43 ms | 3171.28 ms | +3122.85 ms |
| P95 | 62.96 ms | 4065.16 ms | +4002.20 ms |

**Budget:** P95 target = 3500 ms
**B total p95:** 4065.16 ms
**Budget remaining:** -565.16 ms
**Conclusion:** P95_BUDGET_MISSED

## 5. RRF vs BM25

| metric | BM25 | RRF | Delta |
|---|---:|---:|---:|
| Recall@10 | 1.0000 | 0.8571 | -0.1429 |
| MRR@10 | 0.8833 | 0.8571 | -0.0262 |

**INFERENCE:** On this sample, RRF does not provide recall or MRR value over BM25 alone. The value of RRF is limited to rank reordering (q04: 5→1, but both already in top-10).

## 6. Architectural Decision Support

| Component | Recall@10 | MRR@10 | P50 | P95 | Classification |
|---|---:|---:|---:|---:|---|
| BM25 | 1.0000 | 0.8833 | ~7 ms | ~10 ms | HIGH_VALUE_ON_CURRENT_SAMPLE |
| Dense | 0.7857 | 0.7551 | ~40 ms | ~50 ms | UNCERTAIN |
| RRF | 0.8571 | 0.8571 | ~48 ms | ~63 ms | LOW_VALUE_ON_CURRENT_SAMPLE |
| Reranker | 0.8571 | 0.7627 | ~3171 ms | ~4065 ms | LOW_VALUE_ON_CURRENT_SAMPLE |

**Note:** These classifications are for the current 14-query canonical sample only. They do NOT constitute "REMOVE" decisions. They provide diagnostic evidence for a later controlled ablation on a larger dataset.

## 7. Dataset Extension Feasibility

| Source | Rows | Valid Gold | Notes |
|---|---:|---:|---|
| failure_analysis.json | 20 | 10 (v3 validated) | 6 unmapped |
| adversarial_queries.json | 6 | 0 | No gold chunk IDs |
| ragas_phase2_postfix*.json | 45 | 0 | Question-only |
| ablation_sprint2.json | 30 | 0 | Stale gold IDs |

**Realistic non-overlapping expansion size: 0 new queries**

Unless new gold-labeled data is created, the effective evaluation size remains 14 queries.

## 8. Remaining Unknowns

1. Whether RRF would add value on a larger dataset with dense-only gold cases.
2. Whether the reranker's demotion of q20 (RRF 1 → reranker 8) reflects true cross-encoder judgment.
3. Whether q15's reranker rank 19 is correct given its weak relevance (0.25).
4. The q06 candidate-depth boundary is a design choice, not a bug.
5. Whether a different dense model would change the conclusion (untestable here).
