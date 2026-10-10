# Per-query Regression / Recovery

```json
{
  "count_improved": 0,
  "count_regressed": 3,
  "count_unchanged": 10,
  "mean_rank_delta": 1.2308,
  "median_rank_delta": 0,
  "total": 13
}
```

| query_id | gold_chunk_id | dense_rank | bm25_rank | rrf_rank | reranker_rank | final_status | failure_stage | Δrrf | Δrerank |
|---|---|---|---|---|---|---|---|---|---|
| q01 | tier3_llama2_rs_0086 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q02 | tier1_advanced_rag_survey_rs_0002 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q03 | tier1_rag_lewis2020_rs_0023 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q04 | tier2_large_language_models_rs_0006 | 14 | 5 | 1 | 1 | in | NO_FAILURE | -4 | 0 |
| q06 | tier1_advanced_rag_survey_rs_0028 | None | 1 | 24 | None | out | RERANKER_REGRESSION | None | None |
| q09 | tier2_self_instruct_rs_0062 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q10 | tier1_rag_lewis2020_rs_0027 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q11 | tier1_rag_lewis2020_rs_0026 | 1 | 1 | 1 | 2 | in | NO_FAILURE | 0 | 1 |
| q13 | tier3_attention_all_you_need_rs_0008 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q14 | tier3_llama2_rs_0043 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q15 | tier1_rag_lewis2020_rs_0046 | None | 6 | 11 | 19 | out | TOP_K_CUTOFF | None | 8 |
| q16 | tier3_llama2_rs_0007 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q19 | tier3_llama2_rs_0098 | 1 | 1 | 1 | 1 | in | NO_FAILURE | 0 | 0 |
| q20 | tier3_llama2_rs_0152 | 2 | 1 | 1 | 8 | in | NO_FAILURE | 0 | 7 |
