# H0 — Metric Integrity Reconciliation

## MRR@10 Definition

```
RR@10(rank) = 1/rank  if 1 <= rank <= 10
            = 0        if rank > 10 or rank is None
```

## Per-Query RR@10 Verification (RRF stage)

| query_id | rrf_rank | RR@10 |
|---|---:|---:|
| q01 | 1 | 1.0 |
| q02 | 1 | 1.0 |
| q03 | 1 | 1.0 |
| q04 | 1 | 1.0 |
| q06 | 24 | 0 |
| q09 | 1 | 1.0 |
| q10 | 1 | 1.0 |
| q11 | 1 | 1.0 |
| q13 | 1 | 1.0 |
| q14 | 1 | 1.0 |
| q15 | 11 | 0 |
| q16 | 1 | 1.0 |
| q19 | 1 | 1.0 |
| q20 | 1 | 1.0 |

**Sum = 12.0, Mean = 12.0/14 = 0.8571**

## Discrepancy Found

| Source | RRF MRR@10 | Correct? |
|---|---:|---|
| `retrieval_stage_eval_postfix.json` | 0.8666 | **NO** — uses unrestricted 1/rank |
| `fase_f_reranker_ablation.json` (Config A) | 0.8571 | **YES** — uses RR@10 cutoff |

**Root cause:** `retrieval_stage_eval_postfix.json` computes MRR@10 as `mean(1/rank)` without cutoff at 10. This includes:
- q06: 1/24 = 0.0417 (should be 0)
- q15: 1/11 = 0.0909 (should be 0)

**Correct RRF MRR@10 = 0.8571** (verified from Fase F ablation Config A).

## Reconciliation

| Stage | Recall@10 | MRR@10 (correct) | MRR@10 (postfix artifact) |
|---|---:|---:|---:|
| Dense | 0.7857 | 0.7551 | 0.7551 |
| BM25 | 1.0000 | 0.8833 | 0.8833 |
| RRF | 0.8571 | **0.8571** | 0.8666 (wrong) |
| RRF+Reranker | 0.8571 | 0.7627 | 0.7627 |
| Final | 0.8571 | 0.7589 | 0.7589 |

## Conclusion

**FACT:** The `retrieval_stage_eval_postfix.json` artifact contains an incorrect MRR@10 value for RRF (0.8666 instead of 0.8571) due to unrestricted 1/rank calculation.

**FACT:** The correct RRF MRR@10 = 0.8571, verified from Fase F ablation Config A which uses proper RR@10 cutoff.

**OBSERVATION:** The difference (0.8666 - 0.8571 = 0.0095) comes from including q06 (1/24) and q15 (1/11) contributions that should be zero.

**ACTION:** Use Fase F ablation values as the canonical baseline. The postfix artifact MRR@10 for RRF should be corrected to 0.8571.
