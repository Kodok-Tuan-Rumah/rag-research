# RRF Fusion vs BM25 Movement

| query | gold_chunk | dense_rank | bm25_rank | rrf_rank | delta | classification |
|---|---|---|---:|---:|---:|---|
| q01 | tier3_llama2_rs_0086 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q02 | tier1_advanced_rag_survey_rs_0002 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q03 | tier1_rag_lewis2020_rs_0023 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q04 | tier2_large_language_models_rs_0006 | 14 | 5 | 1 | -4 | FUSION_HELPED |
| q06 | tier1_advanced_rag_survey_rs_0028 | None | 1 | 24 | 23 | FUSION_REGRESSION |
| q09 | tier2_self_instruct_rs_0062 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q10 | tier1_rag_lewis2020_rs_0027 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q11 | tier1_rag_lewis2020_rs_0026 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q13 | tier3_attention_all_you_need_rs_0008 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q14 | tier3_llama2_rs_0043 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q15 | tier1_rag_lewis2020_rs_0046 | None | 6 | 11 | 5 | FUSION_REGRESSION |
| q16 | tier3_llama2_rs_0007 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q19 | tier3_llama2_rs_0098 | 1 | 1 | 1 | 0 | FUSION_NEUTRAL |
| q20 | tier3_llama2_rs_0152 | 2 | 1 | 1 | 0 | FUSION_NEUTRAL |
