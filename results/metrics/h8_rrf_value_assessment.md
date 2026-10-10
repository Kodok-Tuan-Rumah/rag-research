# H8 — RRF Value Assessment

## Recall@10 Comparison

| Stage | Recall@10 |
|---|---:|
| Dense | 0.7857 |
| BM25 | 1.0000 |
| RRF | 0.8571 |

**RRF recall delta vs BM25:** 0.8571 - 1.0000 = **-0.1429**

## MRR@10 Comparison (corrected)

| Stage | MRR@10 |
|---|---:|
| Dense | 0.7551 |
| BM25 | 0.8833 |
| RRF | 0.8571 |

**RRF MRR delta vs BM25:** 0.8571 - 0.8833 = **-0.0262**

## BM25 Hits Lost by RRF

| query_id | BM25 rank | RRF rank | lost top-10? |
|---|---:|---:|---|
| q06 | 1 | 24 | YES |
| q15 | 6 | 11 | YES |

**Count:** 2

## Dense-Only Gains Added by RRF

| query_id | dense rank | RRF rank | gained top-10? |
|---|---:|---:|---|
| — | — | — | — |

**Count:** 0 (no dense-only gold cases in this sample)

## Rank Improvements Without Top-10 Change

| query_id | BM25 rank | RRF rank | improved? | top-10 status |
|---|---:|---:|---|---|
| q04 | 5 | 1 | YES | both in top-10 |

**Count:** 1 (q04: BM25 5 → RRF 1, but both already in top-10)

## Summary

On the current 14-query canonical evaluation set:

- RRF Recall@10 (0.8571) is **below** BM25 Recall@10 (1.0000).
- RRF MRR@10 (0.8571) is **below** BM25 MRR@10 (0.8833).
- RRF loses 2 BM25 hits (q06, q15).
- RRF adds 0 dense-only gold coverage.
- RRF improves rank for 1 query (q04) without changing top-10 hit status.

**INFERENCE:** On this sample, RRF does not provide recall or MRR value over BM25 alone. The value of RRF is limited to rank reordering (q04) without top-10 improvement.
