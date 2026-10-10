# Fase H — RRF vs BM25 Failure Analysis & Controlled Fusion Mechanism Audit

## 1. Executive Summary

**Proven current facts:**

- BM25 Recall@10 = 1.0000 (14/14)
- RRF Recall@10 = 0.8571 (12/14)
- RRF loses 2 BM25 hits: q06 (BM25 1 → RRF 24) and q15 (BM25 6 → RRF 11)
- Both losses are BM25-only gold cases where dense misses the gold chunk
- RRF adds 0 dense-only gold coverage in this sample
- RRF improves rank for 1 query (q04: BM25 5 → RRF 1) without changing top-10 hit status
- q06 is outside rerank depth (RRF rank 24 > 20)
- q15 is JUDGED_CORRECT (gold relevance 0.25, no top-10 competitor strongly relevant)

**RRF Current Value: LOW** (on this 14-query canonical sample)

## 2. Metric Integrity

**FACT:** `retrieval_stage_eval_postfix.json` contains an incorrect MRR@10 value for RRF (0.8666 instead of 0.8571) due to unrestricted 1/rank calculation.

**FACT:** The correct RRF MRR@10 = 0.8571, verified from Fase F ablation Config A which uses proper RR@10 cutoff.

**Corrected baseline:**

| Stage | Recall@10 | MRR@10 |
|---|---:|---:|
| Dense | 0.7857 | 0.7551 |
| BM25 | 1.0000 | 0.8833 |
| RRF | 0.8571 | **0.8571** |
| RRF+Reranker | 0.8571 | 0.7627 |
| Final | 0.8571 | 0.7589 |

## 3. BM25 vs Dense vs RRF

| query_id | dense_rank | bm25_rank | rrf_rank | dense_hit | bm25_hit | rrf_hit@10 | category |
|---|---:|---:|---:|---|---|---|---|
| q01 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q02 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q03 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q04 | 14 | 5 | 1 | yes | yes | yes | BOTH_SOURCES |
| q06 | — | 1 | 24 | no | yes | no | BM25_ONLY |
| q09 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q10 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q11 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q13 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q14 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q15 | — | 6 | 11 | no | yes | no | BM25_ONLY |
| q16 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q19 | 1 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |
| q20 | 2 | 1 | 1 | yes | yes | yes | BOTH_SOURCES |

## 4. RRF Regression Queries

| query_id | BM25 rank | RRF rank | delta | dense_contrib | BM25_contrib | classification |
|---|---:|---:|---:|---:|---:|---|
| q06 | 1 | 24 | +23 | None | 0.008197 | BM25_ONLY_RRF_DEMOTION |
| q15 | 6 | 11 | +5 | None | 0.010606 | BM25_ONLY_RRF_DEMOTION |

## 5. RRF Contribution Forensics

### q06
- Dense contribution: None (dense miss)
- BM25 contribution: 0.008197 (rank 1, section boost 1.0)
- Total RRF score: 0.008197
- RRF rank: 24
- **Classification:** BM25_ONLY_WEIGHT_EFFECT

### q15
- Dense contribution: None (dense miss)
- BM25 contribution: 0.010606 (rank 6, section boost 1.4)
- Total RRF score: 0.010606
- RRF rank: 11
- **Classification:** BM25_ONLY_WEIGHT_EFFECT

## 6. Source Overlap Analysis

| category | count | RRF behavior |
|---|---:|---|
| BM25_ONLY | 2 | Both hurt (q06: 1→24, q15: 6→11) |
| DENSE_ONLY | 0 | — |
| BOTH_SOURCES | 12 | 11 neutral, 1 helped (q04) |
| NEITHER | 0 | — |

**OBSERVATION:** RRF systematically hurts BM25-only golds in this canonical sample (2/2).

**OBSERVATION:** RRF adds 0 dense-only gold coverage in this sample.

**OBSERVATION:** RRF primarily reshapes ranking rather than adding recall.

## 7. Section-Boost Observations

| query_id | gold_section | boost | implicated? |
|---|---|---:|---|
| q06 | ii. overview of rag | 1.0 | No (no boost) |
| q15 | methods | 1.4 | Partial (boost helps but insufficient) |

**OBSERVATION:** Section boost is not the primary cause of RRF regressions. The dominant mechanism is BM25-only weight effect.

## 8. q06

- BM25 rank = 1, Dense = miss, RRF rank = 24
- Rerank input = NONE
- Classification: **BEYOND_RERANK_DEPTH**
- Reranker is not responsible for q06 failure
- Filtering contribution: `removed_before_gold = 0`

## 9. q15

- BM25 rank = 6, Dense = miss, RRF rank = 11
- Rerank input rank = 11, Reranker rank = 19
- D4 judgment: gold relevance = 0.25, classification = JUDGED_CORRECT
- Both RRF (11) and reranker (19) positions are outside top-10
- This is **NOT an MRR@10 loss** — both are outside top-10

## 10. RRF Value Assessment

| metric | BM25 | RRF | Delta |
|---|---:|---:|---:|
| Recall@10 | 1.0000 | 0.8571 | -0.1429 |
| MRR@10 | 0.8833 | 0.8571 | -0.0262 |

- BM25 hits lost by RRF: 2 (q06, q15)
- Dense-only gains added by RRF: 0
- Queries where RRF improves rank without changing top-10: 1 (q04)

**INFERENCE:** On this sample, RRF does not provide recall or MRR value over BM25 alone.

## 11. Failure Taxonomy

| query_id | primary_label | secondary_label |
|---|---|---|
| q01 | NO_FAILURE | — |
| q02 | NO_FAILURE | — |
| q03 | NO_FAILURE | — |
| q04 | NO_FAILURE | RRF_HELPED |
| q06 | BM25_ONLY_RRF_DEMOTION | BEYOND_RERANK_DEPTH |
| q09 | NO_FAILURE | — |
| q10 | NO_FAILURE | — |
| q11 | BOTH_SOURCE_RRF_REORDER | — |
| q13 | NO_FAILURE | — |
| q14 | NO_FAILURE | — |
| q15 | BM25_ONLY_RRF_DEMOTION | — |
| q16 | NO_FAILURE | — |
| q19 | NO_FAILURE | — |
| q20 | BOTH_SOURCE_RRF_REORDER | — |

## 12. Architectural Interpretation

### Dominant Reason RRF Recall@10 < BM25 Recall@10

**B + C: BM25-only cases are demoted by fusion because dense candidates introduce competing RRF scores.**

When dense misses a BM25-only gold chunk:
1. The gold chunk receives only half-weight BM25 contribution.
2. Many dense+BM25 mixed chunks receive combined contributions from both sources.
3. These mixed chunks outscore the BM25-only gold chunk.
4. The gold chunk is demoted below top-10.

### RRF Value

**RRF_CURRENT_VALUE: LOW**

- RRF Recall@10 (0.8571) < BM25 Recall@10 (1.0000)
- RRF MRR@10 (0.8571) < BM25 MRR@10 (0.8833)
- 0 dense-only gold coverage added
- 2 BM25 hits lost
- Sample size N=14

### Mandatory Questions

1. **Why does BM25 Recall@10 = 1.0000 while RRF Recall@10 = 0.8571?**
   RRF loses q06 and q15 because they are BM25-only gold cases where dense misses and mixed chunks outscore them.

2. **Which exact queries are lost by RRF?**
   q06 (BM25 1 → RRF 24) and q15 (BM25 6 → RRF 11).

3. **Are those losses primarily BM25-only cases?**
   Yes. Both q06 and q15 are BM25_ONLY (dense misses gold).

4. **Does dense retrieval contribute any new gold recall in this 14-query sample?**
   No. There are 0 dense-only gold cases in this sample.

5. **How often does RRF improve ranking without improving Recall@10?**
   1 query (q04: BM25 5 → RRF 1, but both already in top-10).

6. **Does section boosting materially explain any observed regression?**
   No. q06 has boost 1.0 (no effect). q15 has boost 1.4 (helps but insufficient).

7. **Is q06 primarily a fusion problem or a candidate-depth problem?**
   Both: fusion demotes it to rank 24, and depth 20 excludes it from reranker. The root cause is fusion demotion.

8. **Does RRF add enough value over BM25 to justify its complexity in this sample?**
   No. RRF provides no recall or MRR improvement over BM25 on this sample.

9. **What evidence is still insufficient to make an architecture-level decision?**
   - N=14 is too small for statistical significance
   - No dense-only gold cases exist in this sample to test RRF's value for dense coverage
   - No controlled ablation on a larger dataset has been performed

## 13. Remaining Unknowns

1. Whether RRF would add value on a larger dataset with dense-only gold cases.
2. Whether the reranker's demotion of q20 (RRF 1 → reranker 8) reflects true cross-encoder judgment.
3. Whether q15's reranker rank 19 is correct given its weak relevance (0.25).
4. The q06 candidate-depth boundary is a design choice, not a bug.
