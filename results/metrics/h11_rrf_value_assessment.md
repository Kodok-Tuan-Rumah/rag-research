# H11 — RRF Current Value Assessment

## Evidence Summary

| metric | BM25 | RRF | Delta |
|---|---:|---:|---:|
| Recall@10 | 1.0000 | 0.8571 | -0.1429 |
| MRR@10 | 0.8833 | 0.8571 | -0.0262 |

## Counts

| metric | value |
|---|---:|
| RRF regressions | 2 (q06, q15) |
| RRF improvements | 1 (q04: rank 5→1, but both in top-10) |
| Dense-only gold coverage added by RRF | 0 |
| BM25 hits lost by RRF | 2 |

## Assessment

**RRF_CURRENT_VALUE: LOW**

**Reasoning:**
- RRF Recall@10 is below BM25 Recall@10 on this sample.
- RRF MRR@10 is below BM25 MRR@10 on this sample.
- RRF adds 0 dense-only gold coverage.
- RRF improves rank for 1 query (q04) without changing top-10 hit status.
- The only RRF benefit observed is rank reordering, not recall improvement.

**Note:** This assessment is for the current 14-query canonical sample only. It does not constitute a general proof that BM25 is superior to RRF. It provides decision support for a later controlled ablation.
