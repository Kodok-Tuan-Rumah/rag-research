# K1 Task 11a — Generation Flakiness Measurement

> Observation only. No retrieval parameter, model, threshold, or gold label was changed.
> Model `openai/gpt-oss-20b`. 4 generation call(s).

## Result

| Metric | Value |
|---|---:|
| Answered | 2/4 |
| **Answer rate** | **0.5000** |
| query_type | general |
| top_k / chunks | 8 / 8 |
| max_tokens / temperature | 512 / 0.2 |
| Latency median | 1081.4 ms |

## Outcome distribution

| Terminal path | Count |
|---|---:|
| `ANSWERED` | 2 |
| `ERROR_JSON_VALIDATE_FAILED` | 1 |
| `ABSTAIN_LLM_STATUS` | 1 |

## Overall

- Calls: **4**
- Answered: **2**
- **Overall answer rate: 0.5000**

## Scope limits

- Sample is small by design. It is a triage signal, not a system-wide rate.
- Says nothing about Recall@10 or MRR@10; those are frozen separately in `k1_baseline_freeze.md`.
- Any comparison with the README historical baseline must state the dataset version and ranking mode, because neither is implicit.

## Reproduce

```bash
python3 scripts/diagnose_generation_flakiness.py \
    --dataset data/evaluation/retrieval_eval_dataset.json \
    --query-ids (all) --iterations 4 \
    --out-prefix results/metrics/k1_confounder_topk8
```
