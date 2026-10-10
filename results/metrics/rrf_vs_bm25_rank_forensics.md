# H1 — Full BM25 vs Dense vs RRF Rank Forensics

## Complete Table

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

## Source Categories

| category | count | queries |
|---|---:|---|
| BM25_ONLY | 2 | q06, q15 |
| DENSE_ONLY | 0 | — |
| BOTH_SOURCES | 12 | q01,q02,q03,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| NEITHER | 0 | — |

## RRF Movement Classification

| query_id | bm25_rank | rrf_rank | movement | classification |
|---|---:|---:|---|---|
| q04 | 5 | 1 | -4 | RRF_HELPED |
| q06 | 1 | 24 | +23 | RRF_HURT + RRF_LOST_TOP10 |
| q15 | 6 | 11 | +5 | RRF_HURT + RRF_LOST_TOP10 |
| All others | 1 | 1 | 0 | RRF_NEUTRAL |

## Summary

| classification | count | queries |
|---|---:|---|
| RRF_HELPED | 1 | q04 |
| RRF_NEUTRAL | 11 | q01,q02,q03,q09,q10,q11,q13,q14,q16,q19,q20 |
| RRF_HURT | 2 | q06, q15 |
| RRF_LOST_TOP10 | 2 | q06, q15 |
