# H6 — q06 Forensic

## Current Status

| field | value |
|---|---|
| BM25 rank | 1 |
| Dense rank | none (miss) |
| RRF rank | 24 |
| Rerank input | NONE |
| Rerank depth | 20 |
| Classification | BEYOND_RERANK_DEPTH |

## Chain

```
BM25 rank 1
→ weighted + section-boosted RRF
→ rank 24
→ rerank depth 20
→ not passed to reranker
```

## Explanation

- q06 is **BM25_ONLY** (dense misses gold).
- BM25 rank 1 contributes only half-weight RRF score (0.008197).
- Many dense+BM25 mixed chunks have higher combined RRF scores.
- RRF rank 24 exceeds rerank depth 20.
- Reranker is **not responsible** for q06 failure.

## Filtering Contribution

- `removed_before_gold = 0` (no low-quality chunks removed before gold in RRF list)
- Filtering does not explain the rank movement from 1 to 24.

## Conclusion

**FACT:** q06 is outside the current rerank depth. This is a candidate-depth boundary case, NOT a reranker failure.
