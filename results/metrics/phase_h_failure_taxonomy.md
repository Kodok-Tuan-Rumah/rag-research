# H9 — Failure Taxonomy

## Complete Taxonomy

| query_id | primary_label | secondary_label | evidence |
|---|---|---|---|
| q01 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q02 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q03 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q04 | NO_FAILURE | RRF_HELPED | dense=14, bm25=5, rrf=1 (hybrid improves over both) |
| q06 | BM25_ONLY_RRF_DEMOTION | BEYOND_RERANK_DEPTH | bm25=1, dense=miss, rrf=24 (>20) |
| q09 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q10 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q11 | BOTH_SOURCE_RRF_REORDER | — | dense=1, bm25=1, rrf=1, reranker=2 (demoted but still top-10) |
| q13 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q14 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q15 | BM25_ONLY_RRF_DEMOTION | — | bm25=6, dense=miss, rrf=11 (outside top-10) |
| q16 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q19 | NO_FAILURE | — | dense=1, bm25=1, rrf=1, reranker=1, final=1 |
| q20 | BOTH_SOURCE_RRF_REORDER | — | dense=2, bm25=1, rrf=1, reranker=8 (demoted but still top-10) |

## Summary

| primary_label | count | queries |
|---|---:|---|
| NO_FAILURE | 10 | q01,q02,q03,q04,q09,q10,q13,q14,q16,q19 |
| BM25_ONLY_RRF_DEMOTION | 2 | q06, q15 |
| BOTH_SOURCE_RRF_REORDER | 2 | q11, q20 |
| DENSE_MISS | 0 | — |
| BM25_MISS | 0 | — |
| RRF_TOP10_LOSS | 2 | q06, q15 |
| BEYOND_RERANK_DEPTH | 1 | q06 |
| RERANKER_DEMOTION | 2 | q11, q20 |
| FINAL_FILTER_LOSS | 0 | — |
