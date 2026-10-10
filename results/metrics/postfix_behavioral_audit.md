# Post-Fix Behavioral Audit

## Canonical Baseline

**CURRENT CORRECTED PRODUCTION BASELINE**

| Stage | Recall@10 | MRR@10 |
|---|---:|---:|
| Dense | 0.7857 | 0.7551 |
| BM25 | 1.0000 | 0.8833 |
| RRF | 0.8571 | 0.8666 |
| RRF + Reranker | 0.8571 | 0.7627 |
| Final | 0.8571 | 0.7589 |

**Note:** The previous pre-fix RRF+reranker Recall@10 = 0.9286 came from the buggy heterogeneous candidate ordering and is NOT the clean reranker baseline.

## RRF vs Dense vs BM25

| query_id | dense_rank | bm25_rank | rrf_rank | observation |
|---|---:|---:|---:|---|
| q01 | 1 | 1 | 1 | overlap |
| q02 | 1 | 1 | 1 | overlap |
| q03 | 1 | 1 | 1 | overlap |
| q04 | 14 | 5 | 1 | hybrid improves over both |
| q06 | — | 1 | 24 | BM25-only, RRF demotes |
| q09 | 1 | 1 | 1 | overlap |
| q10 | 1 | 1 | 1 | overlap |
| q11 | 1 | 1 | 1 | overlap |
| q13 | 1 | 1 | 1 | overlap |
| q14 | 1 | 1 | 1 | overlap |
| q15 | — | 6 | 11 | BM25-only, RRF demotes |
| q16 | 1 | 1 | 1 | overlap |
| q19 | 1 | 1 | 1 | overlap |
| q20 | 2 | 1 | 1 | hybrid improves over dense |

**FACT:** BM25 Recall@10 = 1.0 (14/14). Dense Recall@10 = 0.7857 (11/14). RRF Recall@10 = 0.8571 (12/14).

**OBSERVATION:** RRF is worse than BM25 for q06 (1→24) and q15 (6→11) because dense misses leave BM25-only hits underweighted.

## RRF Failure Taxonomy

| classification | count | queries |
|---|---:|---|
| NO_FAILURE | 10 | q01,q02,q03,q04,q09,q10,q13,q14,q16,q19 |
| RERANKER_DEMOTION | 3 | q11,q15,q20 |
| BEYOND_RERANK_DEPTH | 1 | q06 |
| RRF_HELPED | 1 | q04 |
| RRF_HURT | 0 | — |
| FINAL_FILTER_LOSS | 0 | — |

## Reranker MRR Impact

| query_id | RRF RR | Reranked RR | Delta | Note |
|---|---:|---:|---:|---|
| q01 | 1.0 | 1.0 | 0 | |
| q02 | 1.0 | 1.0 | 0 | |
| q03 | 1.0 | 1.0 | 0 | |
| q04 | 1.0 | 1.0 | 0 | |
| q06 | 0.0417 | 0 | -0.0417 | not in reranker |
| q09 | 1.0 | 1.0 | 0 | |
| q10 | 1.0 | 1.0 | 0 | |
| q11 | 1.0 | 0.5 | -0.5 | demoted 1→2 |
| q13 | 1.0 | 1.0 | 0 | |
| q14 | 1.0 | 1.0 | 0 | |
| q15 | 0.0909 | 0.0526 | -0.0383 | demoted 11→19 |
| q16 | 1.0 | 1.0 | 0 | |
| q19 | 1.0 | 1.0 | 0 | |
| q20 | 1.0 | 0.125 | -0.875 | demoted 1→8 |

**Largest negative deltas:**
1. q20: -0.875 (RRF 1 → reranker 8)
2. q11: -0.5 (RRF 1 → reranker 2)
3. q06: -0.0417 (RRF 24 → not in reranker)

**Largest positive deltas:** None. All deltas ≤ 0.

**INFERENCE:** The reranker does not improve MRR on this dataset. The MRR degradation is driven by q20 (large demotion), q11 (small demotion), and q06 (excluded by depth).

## q06

**FACT:**
- BM25 rank = 1
- Dense rank = none (miss)
- RRF rank = 24
- Rerank input = NONE
- Classification: **BEYOND_RERANK_DEPTH**

**OBSERVATION:** q06 is outside the current rerank depth (20). This is a candidate-depth boundary case, NOT a reranker failure.

**UNKNOWN:** Whether rerank depth should be larger is a future tuning decision and remains OUT OF SCOPE.

## q15

**FACT:**
- RRF rank = 11
- Rerank input rank = 11
- Reranker rank = 19
- Final rank = none
- Reranker score = -8.475

**D4 Manual Judgment (pre-fix candidate set):**
- Gold relevance = 0.25 (weakly relevant)
- Top-10 average relevance = 0.30
- Count top-10 ≥ 0.75 = 0
- Classification: JUDGED_CORRECT

**Post-fix candidate set:** The corrected candidate set preserves RRF order. q15 enters at position 11 and is reranked to 19. The D4 judgment remains consistent: the gold chunk does not explicitly answer the expensive-pretraining question, and no top-10 competitor is strongly relevant either.

**Classification:** JUDGED_CORRECT

## q20

**FACT:**
- RRF rank = 1
- Rerank input rank = 1
- Reranker rank = 8
- Final rank = 8
- Reranker score = -5.579

**Manual Review (post-fix candidate set):**

| rank | chunk_id | excerpt | relevance |
|---:|---|---|---:|
| 1 | tier3_llama2_rs_0086 | High IRR scores (closer to 1.0) are typically seen as better... | 0.75 |
| 2 | tier3_llama2_rs_0085 | equivalently sized Vicuna-33B and Falcon 40B models... | 0.25 |
| 3 | tier3_llama2_rs_0151 | rate model over model. Safety Evaluation of Llama 2-Chat... | 0.50 |
| 4 | tier3_llama2_rs_0082 | Llama 2-Chat is less pronounced, although obtaining more than a 60% win-rate... | 0.25 |
| 5 | tier3_llama2_rs_0036 | In order to maximize the diversity, the two responses to a given prompt... | 0.25 |
| 6 | tier3_llama2_rs_0113 | the model to exhibit unsafe behavior, i.e., perform red teaming... | 0.25 |
| 7 | tier3_llama2_rs_0038 | better/preferred by humans. Safety guidelines and more detailed information... | 0.25 |
| 8 | tier3_llama2_rs_0152 (gold) | We then asked raters to judge models for safety violations on a five-point Likert... | 0.75 |
| 9 | tier3_llama2_rs_0013 | They also appear to be on par with some of the closed-source models... | 0.00 |
| 10 | tier3_llama2_rs_0332 | A candidate must score 90% on part I to continue on to parts II and III... | 0.00 |

- Gold relevance: 0.75
- Top-10 average relevance: 0.325
- Count top-10 ≥ 0.75: 2 (rs_0086 and gold rs_0152)

**Classification:** AMBIGUOUS

**Reasoning:** The gold chunk is strongly relevant (directly about IRR measurement for safety assessments). However, rs_0086 is also directly about IRR. The reranker demoted gold from rank 1 to 8, but both chunks are relevant. This is not a clear reranker error.

## Candidate Depth Effects

| query_id | RRF rank | Rerank depth | In reranker? | Final rank |
|---|---:|---:|---|---:|
| q06 | 24 | 20 | NO | — |
| q15 | 11 | 20 | YES | 19 (out) |
| All others | ≤11 | 20 | YES | 1–8 |

**FACT:** Only q06 is outside rerank depth. All other gold chunks enter the reranker.

## Section-Boost Observations

| query_id | gold_chunk_id | section | boost | rrf_score | dense_contrib | bm25_contrib |
|---|---|---|---|---:|---:|---:|
| q01 | tier3_llama2_rs_0086 | conclusion | 1.0 | 0.016393 | 0.0082 | 0.0082 |
| q02 | tier1_advanced_rag_survey_rs_0002 | i. introduction | 1.05 | 0.017213 | 0.0086 | 0.0086 |
| q03 | tier1_rag_lewis2020_rs_0023 | methods | 1.4 | 0.022951 | 0.0115 | 0.0115 |
| q04 | tier2_large_language_models_rs_0006 | abstract | 1.1 | 0.015894 | 0.0074 | 0.0085 |
| q06 | tier1_advanced_rag_survey_rs_0028 | ii. overview of rag | 1.0 | 0.008197 | — | 0.0082 |
| q09 | tier2_self_instruct_rs_0062 | conclusion | 1.0 | 0.016393 | 0.0082 | 0.0082 |
| q10 | tier1_rag_lewis2020_rs_0027 | methods | 1.4 | 0.022951 | 0.0115 | 0.0115 |
| q11 | tier1_rag_lewis2020_rs_0026 | methods | 1.4 | 0.022951 | 0.0115 | 0.0115 |
| q13 | tier3_attention_all_you_need_rs_0008 | abstract | 1.1 | 0.018033 | 0.0090 | 0.0090 |
| q14 | tier3_llama2_rs_0043 | conclusion | 1.0 | 0.016393 | 0.0082 | 0.0082 |
| q15 | tier1_rag_lewis2020_rs_0046 | methods | 1.4 | 0.010606 | — | 0.0106 |
| q16 | tier3_llama2_rs_0007 | conclusion | 1.0 | 0.016393 | 0.0082 | 0.0082 |
| q19 | tier3_llama2_rs_0098 | conclusion | 1.0 | 0.016393 | 0.0082 | 0.0082 |
| q20 | tier3_llama2_rs_0152 | conclusion | 1.0 | 0.016261 | 0.0081 | 0.0082 |

**OBSERVATION:** Section boosts are applied multiplicatively. Methods chunks get 1.4×, abstract 1.1×, introduction 1.05×, conclusion 1.0×. This affects RRF scores but does not change the relative order of chunks from the same section.

## Remaining Unknowns

1. Whether the reranker's demotion of q20 (RRF 1 → reranker 8) reflects true cross-encoder judgment requires deeper manual review of the full chunk texts.
2. Whether q15's reranker rank 19 is correct depends on whether the gold chunk's weak relevance (0.25) justifies demotion below other chunks.
3. The q06 candidate-depth boundary is a design choice, not a bug. Whether to increase rerank depth is OUT OF SCOPE.

## Final Conclusion

1. **Does reranker improve Recall@10?** NO. RRF Recall@10 = 0.8571, Reranked Recall@10 = 0.8571. No change.
2. **Does reranker improve MRR@10?** NO. RRF MRR@10 = 0.8666, Reranked MRR@10 = 0.7627. Degradation of -0.1039.
3. **Which queries cause the MRR degradation?** q20 (-0.875), q11 (-0.5), q06 (-0.0417).
4. **How many gold queries are outside rerank depth?** 1 (q06).
5. **Is q15 likely reranker error?** NO. JUDGED_CORRECT. Gold is only weakly relevant (0.25).
6. **Is q20 likely reranker error?** AMBIGUOUS. Gold is strongly relevant (0.75) but rs_0086 is also directly about IRR.
7. **Is there any remaining candidate-ordering inconsistency?** NO. The D7 fix ensures reranker input order = RRF order after filtering.
8. **Dominant current failure mechanism?** RERANKER_DEMOTION (3 queries: q11, q15, q20).
