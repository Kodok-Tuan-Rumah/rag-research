# K1 Task 11c — Generation Stability Triage

> Date: 2026-10-10
> Method: observation only. No runtime parameter, model, threshold, or gold label was changed.
> Cost: 6 triage calls + 22 follow-up calls = 28 generation calls.
> Raw: `k1_generation_triage.json/.md`, `k1_confounder_topk8.json`

---

## Headline — and a correction to Task 11a

**RRF-only is not broken.** Across three different query types at the policy's
own `top_k`, it answered 3/3. The Task 11a result of 0/15 was **confounded by a
hardcoded `top_k=5`**.

### Triage — 3 queries from the v1 canonical set, both ranking modes

| Query | query_type | top_k | max_tokens | RRF-only | RRF+rerank |
|---|---|---:|---:|---:|---:|
| q01 — Gwet's AC1/2 inter-rater reliability | factual | 5 | 256 | **1/1** | **1/1** |
| q03 — FEVER class label mapping | reasoning | 10 | 768 | **1/1** | **1/1** |
| q15 — RAG characteristic without pre-training | general | 8 | 512 | **1/1** | 0/1 |
| | | | | **3/3** | **2/3** |

Overall 5/6 answered. The single failure was `ERROR_JSON_VALIDATE_FAILED` on
q15 **with reranking enabled** — a format failure, not a grounding failure.

This is the opposite of the Task 11a pattern, where reranking was the rescuing
configuration.

---

## The confounder

Task 11a hardcoded `top_k=5` for every query. `PolicyEngine` assigns `top_k` per
query class: factual 5, general 8, reasoning 10. The measured query
`"What is Hybrid Search in RAG?"` is classified `general`, so Task 11a
deliberately gave it fewer chunks than the runtime would.

A/B on that query, changing only `top_k`:

| top_k | RRF-only | RRF+rerank |
|---:|---:|---:|
| **5** | **0/6** | 10/10 |
| **8** | **4/6** | **6/6** |

Both interventions independently rescue it. Either more chunks, or better
selection within the same budget.

---

## Findings

### FACT 1 — RRF-only answers 3/3 at policy top_k

Three query classes, three different `top_k` values (5, 8, 10) and three
different `max_tokens` values (256, 512, 768). All answered.

### FACT 2 — Task 11a's 0/15 is attributable to `top_k=5`, not to RRF-only

Same query, same model, same mode, `top_k` 5 → 0/6 and 8 → 4/6.

### FACT 3 — Reranking still helps within a fixed budget

At `top_k=5`, reranking moves 0/6 → 10/10. At `top_k=8`, 4/6 → 6/6. Reranking
changes *which* chunks fill the budget, not how many.

### FACT 4 — The format failure is independent of ranking mode

`ERROR_JSON_VALIDATE_FAILED` appeared in both Task 11a (rerank=False, 6/15) and
this triage (rerank=True, 1/6). It is a property of
`response_format={"type":"json_object"}` meeting a reasoning model, not of the
retrieval path. It will not be fixed by changing ranking configuration.

### FACT 5 — The 3-query triage does not reproduce the Task 11a pattern

q15 — the BM25-only gold case that motivated choosing it — failed **with**
reranking and succeeded **without**. The predicted direction reversed.

---

## Interpretation

**INFERENCE.** Task 11a's ranking conclusion was an artifact. With one confound
unguarded, the 0/15 vs 10/10 gap was measured at a `top_k` the runtime does not
use for that query class.

**INFERENCE.** The reranker does something real that ranking metrics did not
capture: at a fixed chunk budget it selects chunks the model can ground on. That
is a *generation-quality* effect, invisible to Recall@10 and MRR@10.

**HYPOTHESIS.** RRF top-5 and reranked top-5 differ enough in grounding quality
that the model abstains on the former. Testable by diffing the two chunk sets
against `ground_truth` — not done here.

**UNKNOWN.** Whether `top_k=5` for `general` queries is a policy error.
`PolicyEngine` gives general 8, so the runtime is not using 5 here. But the API
accepts `top_k` 1..20 from callers, so a caller passing 5 for a general query
would hit the 0/6 regime. That is a reachable production state.

---

## Consequences for prior work

| Item | Status |
|---|---|
| `k1_generation_stability_ab.md` headline ("rerank 0% → 100%") | **Superseded.** Retained for audit trail; the ranking conclusion does not hold at policy top_k. |
| Task 11b (format contract) priority | **Unchanged — still needed.** FACT 4 shows it is ranking-independent. |
| B3 (`try/except` in `Agent.run`) priority | **Unchanged — still needed.** |
| `reranker_default_enabled: false` (K7.13) | **No longer contradicted.** The counter-signal was a measurement artifact. |
| Task 7 (retry guard) | **Unblocked.** No longer competing with a P0. |

---

## Remaining real defect

The format failure is now the one finding that survives every control:

| Config | HTTP 400 rate |
|---|---|
| Task 11a, RRF-only, top_k=5 | 6/15 |
| Triage, RRF+rerank, policy top_k | 1/6 |
| Controlled probe, short context | 0/10 |
| Controlled probe, production context | 6/10 |

It appears whenever context is large, regardless of ranking mode. That is
`generator.py:387` sending `response_format` to a reasoning model, and
`generator.py:394` raising it as `RuntimeError` instead of degrading to abstain.

This is what Task 11b should address. It remains a correctness fix, not tuning.

---

## Scope limits

- 1 iteration per triage cell. Enough to detect a 0% or 100% regime, not enough
  to estimate a rate to two decimals.
- 3 queries of 14. The 2x2 follow-up used one query.
- The `top_k` finding is established on one query only. It should be confirmed on
  q15 before being generalised.
- Nothing here speaks to Recall@10 or MRR@10; those are frozen in
  `k1_baseline_freeze.md`.

---

## Reproduce

```bash
# triage
python3 scripts/diagnose_generation_flakiness.py \
    --dataset data/evaluation/retrieval_eval_dataset.json \
    --query-ids q01,q03,q15 --iterations 1 --modes rrf_only,rerank \
    --out-prefix results/metrics/k1_generation_triage

# top_k A/B on the Task 11a query
for tk in 5 8; do
  python3 scripts/diagnose_generation_flakiness.py \
      --query "What is Hybrid Search in RAG?" --top-k $tk --iterations 6 \
      --out-prefix results/metrics/k1_confounder_topk$tk
done
```