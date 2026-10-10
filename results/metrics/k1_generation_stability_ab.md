# K1 Task 11a — Generation Stability: A/B on ranking mode

> Date: 2026-10-10
> Method: observation only. No retrieval parameter, threshold, model, or gold label changed.
> Cost: 25 generation API calls + 20 probe calls.
> Raw: `k1_generation_flakiness.json/.md` (rerank=False), `k1_generation_flakiness_rerank.json/.md` (rerank=True)

---

## Headline

**Ranking mode changes the generation answer rate from 0% to 100% on the same query.**

| Config | Iterations | ANSWERED | HTTP 400 `json_validate_failed` | Abstain (model chose) |
|---|---:|---:|---:|---:|
| `rerank=False` — RRF-only (**runtime default**) | 15 | **0 (0.0%)** | 6 (40%) | 9 (60%) |
| `rerank=True` — RRF + cross-encoder | 10 | **10 (100%)** | 0 (0%) | 0 (0%) |

Query: `What is Hybrid Search in RAG?` · top_k=5 · query_type=`general` · max_tokens=512 · temperature=0.2 · model `openai/gpt-oss-20b`

Identical in every other respect. Retrieval is deterministic here (fixed chunks, one process). The only difference is whether the cross-encoder reranks before filtering.

---

## Findings

### FACT 1 — RRF-only produced no answer at all across 15 identical calls

Outcomes: `ERROR_JSON_VALIDATE_FAILED` ×6, `ABSTAIN_LLM_STATUS` ×9, `ANSWERED` ×0.

The `run_single_query.py` CLI path, which runs `reranker=True`, did return `ANSWERED` in the same session (observed confidence 0.74–0.76). The two paths therefore do not agree.

### FACT 2 — `response_format={"type": "json_object"}` is incompatible with this reasoning model

Controlled probe, same prompt, same 2688-char context, 10 calls each:

| Config | HTTP 400 | Abstain | Answered |
|---|---:|---:|---:|
| with `response_format=json_object` | 6/10 | 4/10 | 0/10 |
| without `response_format` | **0/10** | 10/10 | 0/10 |

Two distinct Groq error messages observed, same class: `Failed to validate JSON` and `Failed to generate JSON`.

`generator.py:387` sends this parameter. `generator.py:394` raises the resulting HTTP 400 as `RuntimeError` rather than degrading to an abstain.

### FACT 3 — Dropping `response_format` removes the 400 but exposes JSON truncation

Without `response_format`, the raw output is truncated mid-string at `max_tokens=512`:

```
[PARSE_ERROR] LLM output is not valid JSON: Unterminated string starting at:
line 1 column 11 (char 10)
raw='{"answer":"Hybrid Search in Retrieval-Augmented Generation (RAG) refers to a
retrieval strategy that combines multiple search modalities and query expansions...'
```

The model begins a long answer and runs out of budget before closing the JSON. `_parse_llm_output` cannot recover it.

### FACT 4 — `RAGResponse` does not carry the abstain reason

`_make_abstain_response` accepts `reason`, logs it, and discards it. The returned object exposes only `status="INSUFFICIENT_CONTEXT"`. Callers cannot distinguish "model refused", "empty output", and "sanity check rejected" without parsing logs.

This is a diagnosability gap against `PROJECT_GOALS.md` §4 ("structured failure classification", "no silent failures").

### FACT 5 — `Agent.run` has no exception handling

`_execute_generation` is awaited at `agent.py:234` inside no `try` block. A `RuntimeError` from FACT 2 propagates to the caller.

Comparison:

| Path | Catches the `RuntimeError`? |
|---|---|
| `scripts/run_single_query.py:683` | yes → `_make_abstain_response` |
| `src/api/main.py:391` | yes → `failed_safe_abstain` |
| **`src/controller/agent.py`** | **no** |

---

## Interpretation

**INFERENCE.** Under RRF-only, the chunk set reaching the generator is either not grounding-relevant or is ordered such that the model will not answer from it. Reranking changes which chunks survive, and with reranked chunks the model answers 10/10.

**INFERENCE.** Two of the three failure modes are format failures, not reasoning failures. The model demonstrably *can* answer this query — call 2 of the raw probe produced a correct, grounded answer with `confidence_score: 0.8` and `[Source 1]`.

**HYPOTHESIS.** `response_format=json_object` conflicts with reasoning-model output. The 60% abstentions under RRF-only may partly be a coping behaviour: emit minimal schema-valid JSON with `INSUFFICIENT_CONTEXT` rather than risk a format violation. Removing `response_format` removed the 400s but did not restore answers, which weakens this hypothesis — abstention rose to 100%.

**UNKNOWN.** Whether the RRF-only abstention is specific to this query or general. One query, 25 calls.

---

## Scope limits — read before acting

- **N=1 query.** Both configurations were measured on `What is Hybrid Search in RAG?`. A single query cannot establish a system-wide answer rate.
- **This is not a reranker ranking-metric result.** It says nothing about Recall@10 or MRR@10. It concerns generation stability only.
- **It does not contradict the K7.13 decision on its own.** K7.13 judged reranking value on ranking metrics over 13 queries, where reranker value was LOW. This measurement uses a different outcome variable (generation success) and N=1.
- No claim is made that reranking "improves" the system. The finding is that removing it coincided with generation failure on one query.

---

## Consequences for existing decisions

`README.md` and `PROJECT_GOALS.md` §3.1 state `reranker_default_enabled: false` (RRF-only default), adopted from the K7.13 record.

**This measurement is evidence that was not available when that default was set.** It does not overturn it, but it is a counter-signal that should be recorded before the decision is treated as settled.

---

## Recommended next step

Not a fix yet. Required first:

1. **Replicate on a second and third query.** If RRF-only also gives 0% on other queries, this is systemic. If not, it is query-specific.
2. **Decide the format contract.** Options:
   - drop `response_format` and raise `max_tokens` so JSON can close (cost: more tokens, latency already over budget)
   - keep `response_format` and treat HTTP 400 as an abstain rather than a `RuntimeError`
   - switch off `json_object` mode for reasoning models only
3. **Add a regression test** before editing `generator.py`.
4. **Propagate `reason` onto `RAGResponse`** so abstain causes are machine-readable.

Items 2–4 touch the generator runtime path and need the Task 6 baseline recorded first per `PROJECT_GOALS.md` §14.

---

## Reproduce

```bash
python3 scripts/diagnose_generation_flakiness.py --iterations 15 \
    --out-prefix results/metrics/k1_generation_flakiness

python3 scripts/diagnose_generation_flakiness.py --iterations 10 --rerank \
    --out-prefix results/metrics/k1_generation_flakiness_rerank
```