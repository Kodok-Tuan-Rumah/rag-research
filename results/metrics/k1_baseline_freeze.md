# K1 Task 6 — Baseline Freeze

> Date: 2026-10-10
> Purpose: satisfy `PROJECT_GOALS.md` §14 Change Safety Protocol — record the retrieval
> baseline BEFORE any runtime change (Tasks 7, 11b).
> Scope: retrieval only. No LLM call. Deterministic.

---

## Reproduce

```bash
# v2 — the script's default dataset (13 queries)
python3 scripts/run_retrieval_eval.py --out results/metrics/k1_baseline_freeze.json

# v1 — canonical 14-query set
python3 scripts/run_retrieval_eval.py \
    --dataset data/evaluation/retrieval_eval_dataset.json \
    --out results/metrics/k1_baseline_freeze_v1_14q.json
```

---

## Baseline A — v1, canonical 14 queries

`data/evaluation/retrieval_eval_dataset.json` · top_k=10 · rerank=True

| Stage | Recall@5 | Recall@10 | MRR@10 | Avg cand | Avg gold rank | Failures |
|---|---:|---:|---:|---:|---:|---|
| `dense_only` | 0.7857 | 0.7857 | 0.7500 | 40.0 | 2.17 | 11 ok / 1 rank / 2 miss |
| `bm25_only` | 0.9286 | **1.0000** | **0.8833** | 40.0 | 1.64 | 13 ok / 1 rank |
| `rrf_fused` | 0.8571 | 0.8571 | **0.8571** | 40.0 | 3.36 | 12 ok / 2 rank |
| `rrf_reranked` | **0.9286** | **0.9286** | 0.8452 | 20.0 | 1.23 | 13 ok / 1 miss |
| `final_filtered` | **0.9286** | **0.9286** | 0.8452 | 10.0 | 1.23 | 13 ok / 1 miss |

## Baseline B — v2, 13 queries (script default)

`data/evaluation/retrieval_eval_dataset_v2.json` · top_k=10 · rerank=True

| Stage | Recall@5 | Recall@10 | MRR@10 | Failures |
|---|---:|---:|---:|---|
| `dense_only` | 0.8462 | 0.8462 | 0.8077 | 11 ok / 1 rank / 1 miss |
| `bm25_only` | 1.0000 | 1.0000 | 0.9385 | 13 ok |
| `rrf_fused` | 0.9231 | 0.9231 | 0.9231 | 12 ok / 1 rank |
| `rrf_reranked` | 0.9231 | 0.9231 | 0.8333 | 12 ok / 1 miss |
| `final_filtered` | 0.9231 | 0.9231 | 0.8333 | 12 ok / 1 miss |

Both datasets: reranking raises Recall@10 and lowers MRR@10. That trade-off is the
only systematic stage-to-stage movement, and it holds on both sets.

---

## Tooling defect found while freezing

`scripts/run_retrieval_eval.py` read `item["gold_chunk_ids"]` directly, so it could
only evaluate the v2/v3 schema. Passing the v1 canonical dataset raised
`KeyError: 'gold_chunk_ids'`.

`src/evaluation/gold_schema.py` already provides `dataset_records()` and
`normalize_gold_chunk_ids()` for exactly this, and the script already used
`dataset_records()`. It only bypassed the gold-ID normaliser.

**Fix applied:** use `normalize_gold_chunk_ids(item)` at both read sites.

**Proof of no measurement change:** v2 was re-run after the edit. All 15 metrics
(5 stages × recall@5/recall@10/mrr@10) and the 13-query ID set were byte-identical.
See `k1_baseline_freeze.json` (before) vs `k1_baseline_freeze_v2_verify.json` (after).

---

## Discrepancy against the documented historical baseline

`README.md` reports for the 14-query set:

| Stage | README (historical) | Measured now | Δ |
|---|---:|---:|---:|
| Dense | 0.7857 | 0.7857 | 0 |
| BM25 | 1.0000 | 1.0000 | 0 |
| RRF | 0.8571 | 0.8571 | 0 |
| RRF + Reranker | 0.8571 | **0.9286** | **+0.0714** |
| Final (filtered) | 0.8571 | **0.9286** | **+0.0714** |

(Recall@10. MRR@10 differs in the 4th decimal for Dense: 0.7551 vs 0.7500.)

**Root cause isolated to a single query — q15.**

Per-query trace, current code:

```
rrf_ids            n=40  gold_pos=11
rerank_input_ids   n=20  gold_pos=11
reranked_ids       n=20  gold_pos=1     <-- cross-encoder promotes 11 -> 1
filtered_ids       n=19  gold_pos=1
final_ids          n=10  gold_pos=1
```

q15 is one of the two BM25-only gold cases. It sits at RRF rank 11 — outside
top-10 but **inside** the 20-candidate rerank window. The cross-encoder promotes
it to rank 1, and it survives the strict filter.

The historical artifact `results/metrics/fase_f_reranker_ablation.json` records
q15 as `A_rank=None, B_rank=None` for both configs, and R@10 = 0.8571 for both.

**UNKNOWN.** That artifact predates the D7 candidate-selection fix, and its rank
extraction appears to observe a different point in the pipeline than
`trace["reranked_ids"]`. The two measurements are not directly comparable until
that is resolved. No claim is made here about which is correct for the historical
configuration.

**What is not in doubt:** the numbers in this document are reproducible from the
current code, on the current corpus, with the current index.

---

## Coverage limits

- Retrieval only. Says nothing about answer correctness — see
  `k1_generation_stability_ab.md`, which found generation has no measurement
  coverage at all.
- N=14 and N=13. Per `j3_statistical_sufficiency.md`, McNemar exact p=0.50 and
  Wilson CIs overlap. These figures cannot separate the stages statistically.
- `bm25_only` Recall@5 = 0.9286 but Recall@10 = 1.0000 on v1: BM25 finds every
  gold within 10, and misses one within 5. q06 is that query.
- `rrf_fused` has two `ranking_failure` rows on v1 (q06 at rank 24, q15 at rank 11),
  which is the previously documented RRF regression pair.

---

## Gate for subsequent work

Per §14, any of the following must be compared against these numbers, and any
degradation must be reverted:

| Candidate change | Expected effect on this baseline |
|---|---|
| Task 7 — `retry_attempted` guard in `RETRIEVE_AGAIN` | none (orchestration-layer only, no retrieval change) |
| Task 11b — generator format contract | none (generation-layer only) |
| Any change to RRF weights / candidate_k / rerank depth / filter thresholds | **would move these numbers — requires a fresh decision, not a refactor** |

Test suite at freeze time: **161 passed, 0 failed**.