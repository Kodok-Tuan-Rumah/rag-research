# Fase F — Controlled Reranker Ablation

## 1. Executive Summary

**Proven current facts:**

- Reranker does NOT improve Recall@10 (0.8571 → 0.8571, delta = 0).
- Reranker does NOT improve MRR@10 (0.8571 → 0.7589, delta = **-0.0982**).
- Reranker adds ~3123 ms p50 latency (48.43 ms → 3171.28 ms).
- 0 queries improved, 0 harmed, 2 demoted-but-top10, 12 unchanged.
- Net improvement = 0, reranker value ratio = 0.0.

## 2. Ablation Setup

| | Configuration A | Configuration B |
|---|---|---|
| Reranker | OFF (`rerank=False`) | ON (`rerank=True`) |
| Pipeline | RRF → filter → final | RRF → filter → reranker → final |

## 3. Controlled Variables

Both configurations used:
- Same 14 canonical queries
- Same gold dataset
- Same query order
- Same corpus (`chunks_semantic_v2.json`)
- Same dense model (`all-MiniLM-L6-v2`)
- Same BM25
- Same RRF parameters (k=60, weights 0.5/0.5)
- Same section boosts
- Same candidate_k (40)
- Same filtering
- Same top_k (10)

Only independent variable: `reranker = OFF vs ON`.

## 4. Metrics Comparison

| Metric | A (no reranker) | B (reranker) | Delta |
|---|---:|---:|---:|
| Recall@10 | 0.8571 | 0.8571 | 0 |
| MRR@10 | 0.8571 | 0.7589 | **-0.0982** |
| Mean latency | 48.62 ms | 3240.74 ms | +3192.12 ms |
| P50 latency | 48.43 ms | 3171.28 ms | +3122.85 ms |
| P95 latency | 62.96 ms | 4065.16 ms | +4002.20 ms |

**Latency population:** End-to-end retrieval latency per query (dense + BM25 + RRF + filter + reranker if enabled). No generator latency included.

## 5. Per-Query Analysis

| query_id | A_rank | B_rank | A_RR@10 | B_RR@10 | delta_RR@10 | classification |
|---|---:|---:|---:|---:|---:|---|
| q01 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q02 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q03 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q04 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q06 | — | — | 0 | 0 | 0 | UNCHANGED |
| q09 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q10 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q11 | 1 | 2 | 1.0 | 0.5 | **-0.5** | DEMOTED_BUT_TOP10 |
| q13 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q14 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q15 | — | — | 0 | 0 | 0 | UNCHANGED |
| q16 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q19 | 1 | 1 | 1.0 | 1.0 | 0 | UNCHANGED |
| q20 | 1 | 8 | 1.0 | 0.125 | **-0.875** | DEMOTED_BUT_TOP10 |

## 6. MRR@10 Decomposition

| query_id | RRF RR@10 | Reranked RR@10 | Delta | Note |
|---|---:|---:|---:|---|
| q20 | 1.0 | 0.125 | **-0.875** | RRF 1 → reranker 8 |
| q11 | 1.0 | 0.5 | **-0.5** | RRF 1 → reranker 2 |
| q06 | 0 | 0 | 0 | both outside top-10 |
| q15 | 0 | 0 | 0 | both outside top-10 |
| All others | — | — | 0 | no change |

**Largest negative deltas:** q20 (-0.875), q11 (-0.5)
**Largest positive deltas:** None

## 7. Reranker Value Assessment

| metric | value |
|---|---:|
| count_improved | 0 |
| count_harmed | 0 |
| count_demoted_but_top10 | 2 |
| count_unchanged | 12 |
| net_improvement | 0 |
| reranker_value_ratio | 0.0 |
| Recall@10 delta | 0 |
| MRR@10 delta | -0.0982 |

### A. Behavior-Count Value

0 queries improved, 0 harmed, 2 demoted-but-top10, 12 unchanged. Net improvement = 0.

### B. Metric Value

- Recall@10: no change (0.8571 → 0.8571).
- MRR@10: degradation of -0.0982 (0.8571 → 0.7589).

**INFERENCE:** The reranker does not provide measurable value on this dataset. It degrades MRR@10 without improving Recall@10, at a latency cost of ~3.1 s p50.

## 8. Latency Comparison

| metric | A (no reranker) | B (reranker) | Delta |
|---|---:|---:|---:|
| P50 | 48.43 ms | 3171.28 ms | +3122.85 ms |
| P95 | 62.96 ms | 4065.16 ms | +4002.20 ms |

**Budget analysis:**

| metric | value |
|---|---:|
| P95 target | 3500 ms |
| B total p95 | 4065.16 ms |
| budget_remaining | -565.16 ms |
| reranker_latency_ratio | 0.9862 |

**Conclusion:** P95_BUDGET_MISSED

**Wording:** "Reranker is the dominant measured retrieval-stage latency component and the current pipeline exceeds the 3.5 s P95 target."

## 9. q06 / q11 / q15 / q20

### q06

- A: rank = none (RRF rank 24 > top_k 10)
- B: rank = none (RRF rank 24 > rerank depth 20)
- Classification: **BEYOND_RERANK_DEPTH** (both configs)
- Note: q06 is outside rerank depth in both configurations. This is a candidate-depth boundary case, NOT a reranker failure.

### q11

- A: rank = 1
- B: rank = 2
- Delta: -0.5 MRR@10
- Classification: DEMOTED_BUT_TOP10
- Note: Reranker demoted gold from rank 1 to 2, but it remains in top-10.

### q15

- A: rank = none (RRF rank 11 > top_k 10)
- B: rank = none (RRF rank 11 → reranker rank 19)
- Classification: UNCHANGED (both outside top-10)
- Note: q15's RRF rank 11 is already outside top-10. The reranker demotion (11 → 19) does not change the final outcome. This is NOT an MRR@10 loss.

### q20

- A: rank = 1
- B: rank = 8
- Delta: -0.875 MRR@10
- Classification: DEMOTED_BUT_TOP10
- Note: Reranker demoted gold from rank 1 to 8, but it remains in top-10.

## 10. Limitations

- **N=14:** Small sample size. No statistical significance claim.
- **Current canonical dataset:** Results apply only to the 14-query canonical set.
- **Latency directly comparable:** Both configurations measured end-to-end retrieval latency with same methodology. No generator latency included.
- **No population differences:** Both A and B ran on same queries, same corpus, same models.

## 11. Diagnostic Recommendation

The following are NOT tuning instructions:

- Retain reranker as optional component pending further investigation.
- Perform controlled future ablation on larger dataset to confirm these findings.
- Inspect reranker failure cases (q11, q20) with full chunk text.
- Investigate latency optimization for reranker stage.

## 12. Final Interpretation

1. **Does reranker improve Recall@10?** NO (0.8571 → 0.8571).
2. **Does reranker improve MRR@10?** NO (0.8571 → 0.7589, delta -0.0982).
3. **Which queries cause MRR degradation?** q20 (-0.875), q11 (-0.5).
4. **How many gold queries outside rerank depth?** 1 (q06).
5. **Is q15 likely reranker error?** NO — both A and B are outside top-10. Reranker demotion does not change outcome.
6. **Is q20 likely reranker error?** AMBIGUOUS — gold is strongly relevant but rs_0086 is also directly about IRR.
7. **Remaining candidate-ordering inconsistency?** NO — D7 fix ensures reranker input order = RRF order.
8. **Dominant current failure mechanism?** RERANKER_DEMOTION (q11, q20) + BEYOND_RERANK_DEPTH (q06).

---

> **SUPERSEDED 2026-10-10.** This ablation was measured on code that predates
> commit `50796bf` (rerank candidate-selection fix). On the same 14-query v1 set,
> current code gives `config_B` Recall@10 = **0.9286**, not 0.8571. q15 is
> promoted from RRF rank 11 to reranked rank 1. Determinism confirmed across
> three runs, and a fresh run of the same script reproduces 0.9286/0.8452.
>
> Do not use this file as a baseline. See
> [`k1_q15_reconciliation.md`](k1_q15_reconciliation.md) and
> [`fase_f_reranker_ablation_replica_20261010.json`](fase_f_reranker_ablation_replica_20261010.json).
> The JSON artifact itself is retained unmodified as the historical record.
