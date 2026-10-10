# Gold Loss by Stage

| query_id | gold_chunk_id | in_rrf | in_reranked | in_final | lost_stage | loss_reason | filter_rule |
|---|---|---:|---:|---:|---|---|---|
| q01 | tier3_llama2_rs_0086 | True | True | True | - | - | ok |
| q02 | tier1_advanced_rag_survey_rs_0002 | True | True | True | - | - | ok |
| q03 | tier1_rag_lewis2020_rs_0023 | True | True | True | - | - | ok |
| q04 | tier2_large_language_models_rs_0006 | True | True | True | - | - | ok |
| q06 | tier1_advanced_rag_survey_rs_0028 | True | True | True | - | - | ok |
| q09 | tier2_self_instruct_rs_0062 | True | True | True | - | - | ok |
| q10 | tier1_rag_lewis2020_rs_0027 | True | True | True | - | - | ok |
| q11 | tier1_rag_lewis2020_rs_0026 | True | True | True | - | - | ok |
| q13 | tier3_attention_all_you_need_rs_0008 | True | True | True | - | - | ok |
| q14 | tier3_llama2_rs_0043 | True | True | True | - | - | ok |
| q15 | tier1_rag_lewis2020_rs_0046 | True | True | False | filter | ok | ok |
| q16 | tier3_llama2_rs_0007 | True | True | True | - | - | ok |
| q19 | tier3_llama2_rs_0098 | True | True | True | - | - | ok |
| q20 | tier3_llama2_rs_0152 | True | True | True | - | - | ok |

- retrieval/rrf gold present: 14/14 (1.0000)
- reranked gold present: 14/14 (1.0000)
- final gold present: 13/14 (0.9286)
