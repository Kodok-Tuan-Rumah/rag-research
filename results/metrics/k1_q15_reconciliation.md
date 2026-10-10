# C2 — Reconciling the q15 Discrepancy

> Date: 2026-10-10
> Question: `k1_baseline_freeze.md` measured `rrf_reranked` Recall@10 = 0.9286 where
> the README records 0.8571. Is the baseline wrong?
> Answer: **No. The baseline is correct and reproducible. The older numbers are stale.**

---

## Finding

The discrepancy was isolated to one query and is explained by code state that no
longer exists. It is not a measurement error in the current baseline.

### FACT 1 — Two independent scripts agree exactly on current code

v1 canonical set, top_k=10, rerank=True:

| Source | fused R@10 / MRR@10 | reranked R@10 / MRR@10 |
|---|---|---|
| `scripts/run_retrieval_eval.py` | 0.8571 / 0.8571 | 0.9286 / 0.8452 |
| `scripts/fase_f_reranker_ablation.py` (re-run) | 0.8571 / 0.8571 | 0.9286 / 0.8452 |

Different scripts, different retrieval entry points (`search_with_trace` vs
`search`), same numbers.

### FACT 2 — Current behaviour is deterministic

Three consecutive runs each:

```
q15  gold in rerank input: True    reranked rank: 1    (all 3 runs)
q06  gold in rerank input: False   reranked rank: None (all 3 runs)
```

q06 is excluded because its RRF rank is 24 and the rerank depth is 20. q15 sits
at RRF rank 11, inside the window, and the cross-encoder promotes it to rank 1.

### FACT 3 — The historical artifact predates the candidate-selection fix

| Artifact | mtime |
|---|---|
| `fase_f_reranker_ablation.json` | 2026-10-05 00:36 |
| commit `50796bf` (adds `candidates[:20]`, the D7 fix) | 2026-10-07 00:34 |

Two days apart. The artifact was produced before the fix was committed.

---

## Why the two old numbers disagree with each other

`results/metrics/d7_candidate_selection_fix_report.md` records its own before/after:

| | q06 | q15 | reranked R@10 |
|---|---|---|---:|
| D7 before fix | reranked to 1 (rescued) | reranked to 16 | 0.9286 |
| D7 after fix | excluded | reranked to 19 | 0.8571 |
| **measured today** | **excluded** | **reranked to 1** | **0.9286** |

Today's per-query pattern does not match either recorded state: it matches the
D7 *after* row for q06 but the D7 *before* row for q15.

### Ruled out: the pre-fix candidate sort

The pre-fix code sorted rerank candidates with
`key=lambda x: x.get("retrieval_score", 0.0) ... reverse=True)[:20]`, which is the
D7 bug. The D7 audit itself shows q15 was **promoted** to candidate index 2 by
that sort, not demoted:

| query | origin | has `retrieval_score`? | sort key | RRF rank |
|---|---|---|---:|---:|
| q06 | BM25 dict | yes | 0.008197 | 24 |
| q15 | BM25 dict | yes | 0.010606 | 11 |

RRF scores decrease monotonically with rank (q15 at rank 11 scores 0.010606,
below ranks 1–10). Sorting by that key descending therefore reproduces RRF order
and cannot move q15 out of the top 20. Simulated against the live trace and
confirmed.

So the sort is not the explanation for the historical q15 result.

### UNKNOWN

The working tree on 2026-10-05 cannot be reconstructed from git. The artifact
carries no timestamp field, and the two relevant commits are nine seconds apart,
so commit order does not establish run order. The exact state that produced
0.8571 is not recoverable.

---

## Resolution

1. `k1_baseline_freeze.md` stands. It is reproducible, deterministic, and agreed
   upon by two independent scripts. Every "metrics identical" §14 comparison made
   since is valid.

2. `0.8571` for `rrf_reranked` / `final_filtered` is **stale** wherever it
   appears — the README historical baseline table and the D7 fix report's
   *after* column both need marking as pre-`50796bf`.

3. `fase_f_reranker_ablation.json` and `d7_candidate_selection_fix_report.md`
   are retained as an audit trail of a state that no longer exists. They are not
   comparable to current numbers and must not be used as a baseline.

4. Replica artifact written to
   `fase_f_reranker_ablation_replica_20261010.json`. The original was not
   overwritten.

---

## Note on the q06 behaviour

D7's post-fix prediction for q06 holds exactly: "correctly excluded from the
reranker input because its RRF rank (24) exceeds the rerank depth (20)". That
part of the fix report remains valid and is confirmed by three runs today.

The disagreement is confined to q15, whose aggregate contribution moves
`rrf_reranked` between 0.8571 (12/14) and 0.9286 (13/14).

---

## Caveats

- One query explains the whole aggregate gap. With N=14 and McNemar exact
  p=0.50, a single query moving in or out of the top-10 changes Recall@10 by
  7.14 points. Per `j3_statistical_sufficiency.md`, this cannot support any
  architectural conclusion in either direction.
- Determinism was checked on 2 queries × 3 runs. It was not swept across the full
  dataset.

## Reproduce

```bash
python3 scripts/run_retrieval_eval.py \
    --dataset data/evaluation/retrieval_eval_dataset.json \
    --out results/metrics/k1_baseline_freeze_v1_14q.json

# replica of the Fase F ablation, writing to a new file
python3 - <<'PY'
import pathlib, sys; sys.path.insert(0, ".")
import scripts.fase_f_reranker_ablation as m
m.OUT_JSON = pathlib.Path("results/metrics/fase_f_reranker_ablation_replica_20261010.json")
m.main()
PY
```