# J3 — Statistical Sufficiency

## BM25 vs RRF Paired Analysis

| | RRF hit | RRF miss | total |
|---|---:|---:|---:|
| **BM25 hit** | 12 | **2** | 14 |
| **BM25 miss** | 0 | 0 | 0 |
| **total** | 12 | 2 | 14 |

**Discordant pairs:**
- b (BM25 hit, RRF miss) = **2** (q06, q15)
- c (BM25 miss, RRF hit) = **0**

## McNemar Exact Test

- p-value = 2 × (0.5)^2 = **0.5000**
- Significant at α=0.05? **NO**

## Confidence Intervals (Wilson Score, 95%)

| Stage | Recall@10 | 95% CI |
|---|---:|---|
| BM25 | 1.0000 | [0.7847, 1.0000] |
| RRF | 0.8571 | [0.6006, 0.9599] |
| Dense | 0.7857 | [0.5241, 0.9243] |

**Note:** CIs overlap substantially. BM25's lower bound (0.7847) is below RRF's point estimate (0.8571). No population-level superiority claim can be made from this sample.

## Coverage Limitations

| category | present? | count |
|---|---|---:|
| dense-only gold cases | NO | 0 |
| BM25-only gold cases | YES | 2 |
| both-source cases | YES | 12 |
| adversarial-only cases | NO | 0 |

**Critical gap:** 0 dense-only gold cases means RRF's hybrid value is **untestable** with the current dataset.

## Conclusion

**Is N=14 sufficient for a definitive architecture decision?**

**NO / LIMITED.**

The current dataset can answer:
- Does RRF hurt BM25-only gold cases? → YES (2/2 observed)
- Does RRF improve rank for both-source gold cases? → YES (q04: 5→1)

The current dataset **cannot** answer:
- Does RRF add value over BM25 for dense-only coverage? → UNTESTABLE (0 cases)
- Is the BM25 vs RRF difference statistically significant? → NO (McNemar p=0.50)
- Would a different dense model change the conclusion? → UNTESTABLE
