# K1 Task 11a — Generation Flakiness Measurement

> Observation only. No runtime parameter, model, threshold, or gold label was changed.
> One generation API call per iteration.

## Config

| Field | Value |
|---|---|
| Query | `What is Hybrid Search in RAG?` |
| top_k / rerank | 5 / False |
| query_type | general |
| max_tokens / temperature | 512 / 0.2 |
| Model | `openai/gpt-oss-20b` |
| Chunks retrieved | 5 |
| Iterations | 15 |

## Result

| Metric | Value |
|---|---:|
| Answered | 0/15 |
| Failed / abstained | 15/15 |
| **Answer rate** | **0.0000** |
| **Flaky-or-fail rate** | **1.0000** |
| Latency median | 1127.0 ms |
| Latency p95 | 4467.2 ms |

## Outcome distribution

| Terminal path | Count |
|---|---:|
| `ABSTAIN_LLM_STATUS` | 9 |
| `ERROR_JSON_VALIDATE_FAILED` | 6 |

## Interpretation

`ERROR_JSON_VALIDATE_FAILED` occurred. `generator.py:387` sends `response_format={"type": "json_object"}` to a reasoning model; with production-sized context the model occasionally emits output that fails Groq's JSON validation (HTTP 400). `generator.py:394` raises this as `RuntimeError` rather than handling it as a graceful abstain.

**Non-determinism observed**: 15 of 15 identical calls did not answer. `PROJECT_GOALS.md` §10.4 requires metric variance <= 3%; observed variance on this single query exceeds that by a wide margin. A repeat measurement is required before drawing a final rate, because a small sample makes the estimate unstable.

## Scope limits

- Single query, single configuration. This is **not** a system-wide rate.
- Deterministic components (retrieval, RRF, filtering) were shown stable in earlier phases; the variance here sits in the LLM call, not in retrieval.
- `PROJECT_GOALS.md` §2.3 sets a False Abstain Rate target of <= 10%. A single-query sample cannot be compared against that target directly.

## Next

Fix is deliberately NOT proposed yet. Required first:
1. Confirm which terminal path dominates across more iterations.
2. Decide the contract — should a JSON-validation failure abstain gracefully, or retry once without `response_format`?
3. Add a regression test before changing `generator.py`.
