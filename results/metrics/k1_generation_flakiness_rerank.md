# K1 Task 11a — Generation Flakiness Measurement

> Observation only. No runtime parameter, model, threshold, or gold label was changed.
> One generation API call per iteration.

## Config

| Field | Value |
|---|---|
| Query | `What is Hybrid Search in RAG?` |
| top_k / rerank | 5 / True |
| query_type | general |
| max_tokens / temperature | 512 / 0.2 |
| Model | `openai/gpt-oss-20b` |
| Chunks retrieved | 5 |
| Iterations | 10 |

## Result

| Metric | Value |
|---|---:|
| Answered | 10/10 |
| Failed / abstained | 0/10 |
| **Answer rate** | **1.0000** |
| **Flaky-or-fail rate** | **0.0000** |
| Latency median | 719.1 ms |
| Latency p95 | 1673.6 ms |

## Outcome distribution

| Terminal path | Count |
|---|---:|
| `ANSWERED` | 10 |

## Interpretation

## Scope limits

- Single query, single configuration. This is **not** a system-wide rate.
- Deterministic components (retrieval, RRF, filtering) were shown stable in earlier phases; the variance here sits in the LLM call, not in retrieval.
- `PROJECT_GOALS.md` §2.3 sets a False Abstain Rate target of <= 10%. A single-query sample cannot be compared against that target directly.

## Next

Fix is deliberately NOT proposed yet. Required first:
1. Confirm which terminal path dominates across more iterations.
2. Decide the contract — should a JSON-validation failure abstain gracefully, or retry once without `response_format`?
3. Add a regression test before changing `generator.py`.
