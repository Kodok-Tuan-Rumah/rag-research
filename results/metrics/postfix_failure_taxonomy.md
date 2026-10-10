# Post-Fix Failure Taxonomy

| query_id | gold_chunk_id | classification | evidence |
|---|---|---|---|
| q01 | tier3_llama2_rs_0086 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q02 | tier1_advanced_rag_survey_rs_0002 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q03 | tier1_rag_lewis2020_rs_0023 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q04 | tier2_large_language_models_rs_0006 | NO_FAILURE | dense=14, bm25=5, rrf=1 (RRF_HELPED), reranker=1, final=1 |
| q06 | tier1_advanced_rag_survey_rs_0028 | BEYOND_RERANK_DEPTH | dense=miss, bm25=1, rrf=24 (>20), reranker=NONE, final=out |
| q09 | tier2_self_instruct_rs_0062 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q10 | tier1_rag_lewis2020_rs_0027 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q11 | tier1_rag_lewis2020_rs_0026 | RERANKER_DEMOTION | dense=1, bm25=1, rrf=1, reranker=2, final=2 (still in top-10) |
| q13 | tier3_attention_all_you_need_rs_0008 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q14 | tier3_llama2_rs_0043 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q15 | tier1_rag_lewis2020_rs_0046 | RERANKER_DEMOTION | dense=miss, bm25=6, rrf=11, reranker=19, final=out |
| q16 | tier3_llama2_rs_0007 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q19 | tier3_llama2_rs_0098 | NO_FAILURE | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q20 | tier3_llama2_rs_0152 | RERANKER_DEMOTION | dense=2, bm25=1, rrf=1, reranker=8, final=8 (still in top-10) |

## Summary

| classification | count |
|---|---:|
| NO_FAILURE | 10 |
| RERANKER_DEMOTION | 3 |
| BEYOND_RERANK_DEPTH | 1 |
| RRF_HELPED | 1 |
| RRF_HURT | 0 |
| FINAL_FILTER_LOSS | 0 |
