# H7 — q15 Forensic

## Current Status

| field | value |
|---|---|
| BM25 rank | 6 |
| Dense rank | none (miss) |
| RRF rank | 11 |
| Rerank input rank | 11 |
| Reranker rank | 19 |
| Final rank | none |
| Reranker score | -8.475 |

## D4 Manual Judgment

- Gold relevance = 0.25 (weakly relevant)
- Top-10 average relevance = 0.30
- Count top-10 ≥ 0.75 = 0
- Classification = **JUDGED_CORRECT**

## Post-Fix Behavior

- RRF rank 11 is already outside top-10.
- Reranker demotion (11 → 19) does not change the final outcome.
- This is **NOT an MRR@10 loss** — both RRF and reranker positions are outside top-10.

## Classification

**RRF rank regression + outside-top10 case**

The reranker demotion is observed but does not affect the final result because the gold chunk was already outside top-10 after RRF.
