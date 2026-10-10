# H3 — RRF Contribution Forensics

## q06

| field | value |
|---|---|
| dense_contrib | None (dense miss) |
| bm25_contrib | 0.008197 |
| section | ii. overview of rag |
| section_boost | 1.0 |
| total_rrf_score | 0.008197 |
| rrf_rank | 24 |

**Explanation:** q06 gold is BM25-only. BM25 rank 1 contributes `0.5 * 1.0 / (60 + 1 + 1) = 0.008065`. With section boost 1.0, total = 0.008197. Many dense+BM25 mixed chunks have higher combined scores, pushing gold to rank 24.

**Classification:** BM25_ONLY_WEIGHT_EFFECT

## q15

| field | value |
|---|---|
| dense_contrib | None (dense miss) |
| bm25_contrib | 0.010606 |
| section | methods |
| section_boost | 1.4 |
| total_rrf_score | 0.010606 |
| rrf_rank | 11 |

**Explanation:** q15 gold is BM25-only. BM25 rank 6 contributes `0.5 * 1.4 / (60 + 6 + 1) = 0.010606`. Section boost 1.4 helps but is insufficient to overcome dense competition.

**Classification:** BM25_ONLY_WEIGHT_EFFECT

## Summary

Both RRF regressions are explained by **BM25_ONLY_WEIGHT_EFFECT**: when dense misses a BM25-only gold, the BM25 half-weight contribution is too small to compete with dense+BM25 mixed chunks.
