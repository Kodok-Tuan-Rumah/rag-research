# H2 — RRF Regression Queries

## Primary Failure Set

| query_id | BM25 rank | RRF rank | delta | dense_contrib | BM25_contrib | classification |
|---|---:|---:|---:|---:|---:|---|
| q06 | 1 | 24 | +23 | None | 0.008197 | BM25_ONLY_RRF_DEMOTION |
| q15 | 6 | 11 | +5 | None | 0.010606 | BM25_ONLY_RRF_DEMOTION |

## Notes

- Both regressions are **BM25-only** gold cases (dense misses).
- q06: BM25 rank 1 → RRF rank 24 (lost top-10)
- q15: BM25 rank 6 → RRF rank 11 (lost top-10)
- q20 is NOT included: BM25=1, RRF=1 (no regression)
