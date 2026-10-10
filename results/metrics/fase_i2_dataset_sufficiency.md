# I2 — Dataset Sufficiency

## Sample Size

**N = 14 canonical queries**

## Query Type Distribution

| type | count | queries |
|---|---:|---|
| factual | 7 | q01,q03,q04,q09,q10,q13,q14 |
| reasoning | 7 | q02,q06,q11,q15,q16,q19,q20 |
| general | 0 | — |
| adversarial | 0 | — |

## Difficulty Distribution

| difficulty | count | queries |
|---|---:|---|
| easy | 0 | — |
| medium | 11 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19 |
| hard | 3 | q06,q15,q20 |

## Document/Source Distribution

| doc_id | count | queries |
|---|---:|---|
| tier3_llama2 | 5 | q01,q14,q16,q19,q20 |
| tier1_rag_lewis2020 | 4 | q03,q10,q11,q15 |
| tier1_advanced_rag_survey | 2 | q02,q04 |
| tier2_large_language_models | 1 | q06 |
| tier2_self_instruct | 1 | q09 |
| tier3_attention_all_you_need | 1 | q13 |

## Source Counts

| category | count | queries |
|---|---:|---|
| BM25-only gold | 2 | q06, q15 |
| Dense-only gold | 0 | — |
| Both-source gold | 12 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| Neither | 0 | — |

## Paired Analysis: BM25 vs RRF Recall@10

| | RRF hit | RRF miss | total |
|---|---:|---:|---:|
| **BM25 hit** | 12 | **2** | 14 |
| **BM25 miss** | 0 | 0 | 0 |
| **total** | 12 | 2 | 14 |

**Discordant pairs:**
- b (BM25 hit, RRF miss) = **2** (q06, q15)
- c (BM25 miss, RRF hit) = **0**

**McNemar exact test:**
- p-value = 2 × (0.5)^2 = **0.5000**

**Interpretation:** The observed difference (2/14 = 14.3% recall loss) is NOT statistically significant at α=0.05. With only 2 discordant pairs, we cannot reject the null hypothesis that BM25 and RRF have equal Recall@10.

## Confidence Intervals (Wilson Score, 95%)

| Stage | Recall@10 | 95% CI |
|---|---:|---|
| BM25 | 1.0000 | [0.7847, 1.0000] |
| RRF | 0.8571 | [0.6006, 0.9599] |
| Dense | 0.7857 | [0.5241, 0.9243] |

**Note:** The CIs overlap substantially. BM25's lower bound (0.7847) is below RRF's point estimate (0.8571). No population-level superiority claim can be made from this sample.

## Missing Evaluation Categories

| category | present? | count |
|---|---|---:|
| dense-only gold cases | NO | 0 |
| BM25-only gold cases | YES | 2 |
| both-source cases | YES | 12 |
| adversarial-only cases | NO | 0 |

**Critical gap:** There are **0 dense-only gold cases** in this sample. This means the dataset cannot test whether RRF adds value over BM25 for dense-only coverage. The value of RRF for hybrid coverage is **untestable** with the current dataset.

## Conclusion

**Is N=14 sufficient for a definitive architecture decision?**

**NO / LIMITED.**

The current dataset can answer:
- Does RRF hurt BM25-only gold cases? → YES (2/2 observed)
- Does RRF improve rank for both-source gold cases? → YES (q04: 5→1)

The current dataset **cannot** answer:
- Does RRF add value over BM25 for dense-only coverage? → UNTESTABLE (0 cases)
- Is the BM25 vs RRF difference statistically significant? → NO (McNemar p=0.50)
- Would a different dense model change the conclusion? → UNTESTABLE
- Is the reranker's demotion of q20 correct? → AMBIGUOUS (needs larger sample)

**Scope language:** All architectural value assessments are restricted to the current 14-query canonical sample.
