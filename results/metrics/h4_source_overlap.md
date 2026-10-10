# H4 — Source Overlap Analysis

## Per-Query Source Overlap

| query_id | dense_hit | bm25_hit | source_overlap |
|---|---|---|---|
| q01 | yes | yes | both |
| q02 | yes | yes | both |
| q03 | yes | yes | both |
| q04 | yes | yes | both |
| q06 | no | yes | bm25_only |
| q09 | yes | yes | both |
| q10 | yes | yes | both |
| q11 | yes | yes | both |
| q13 | yes | yes | both |
| q14 | yes | yes | both |
| q15 | no | yes | bm25_only |
| q16 | yes | yes | both |
| q19 | yes | yes | both |
| q20 | yes | yes | both |

## RRF Behavior by Source Category

| category | count | RRF behavior |
|---|---:|---|
| bm25_only | 2 | Both hurt (q06: 1→24, q15: 6→11) |
| dense_only | 0 | — |
| both | 12 | 11 neutral (rank 1→1), 1 helped (q04: 5→1) |
| neither | 0 | — |

## Answers

1. **Does RRF systematically hurt BM25-only golds?** OBSERVATION: Yes, in this canonical sample (2/2 BM25-only golds were hurt).
2. **Does RRF behave differently when both sources retrieve gold?** OBSERVATION: Yes — 11/12 both-source cases are neutral, 1/12 helped.
3. **Does dense coverage appear beneficial or harmful to gold rank?** OBSERVATION: Dense coverage is beneficial when gold is in both sources (q04 improved). Dense coverage is harmful when gold is BM25-only (q06, q15 demoted).
4. **Is there evidence that RRF primarily reshapes ranking rather than adding recall?** OBSERVATION: Yes — RRF adds 0 dense-only gold coverage in this sample. All RRF gains come from reordering existing BM25/dense hits.

**Note:** These observations are from the current 14-query canonical sample only. No generalization beyond this dataset is made.
