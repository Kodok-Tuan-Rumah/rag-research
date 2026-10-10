# Dead Scripts — Archived 2026-10-10

**Nothing was deleted.** Every file is recoverable from git history via
`git mv archive/dead_scripts_2026_10/<path> <original path>`.

## Why these are here, separately from `legacy_2026_10/`

`archive/legacy_2026_10/` holds modules with **0 inbound imports**.
These eight scripts were imported by nobody either, but they fail for a
different and more specific reason: **the data they read no longer exists.**

Labeled data was migrated from `data/processed/` to `data/evaluation/` during
the K1 work. Eight scripts still point at the old locations, so they raise
`FileNotFoundError` on the first read. They are not "unreferenced", they are
**unrunnable**, which is a stronger statement and a different failure mode.

## Files

| Original path | Reads (all missing) | Produced |
|---|---|---|
| `audit_chunk_id_consistency.py` | `ground_truth_qa{,_clean,_rebuilt,_v2}.json`, `holdout_eval{,_v2}.json`, `train_eval{,_v2}.json` | console report |
| `experiments/ablation_per_category.py` | `ground_truth_qa.json`, `adversarial_queries.json`, `multihop_queries.json`, `paraphrase_queries.json` | `results/metrics/ablation_per_category.json` **(kept)** |
| `experiments/ablation_runner.py` | `ground_truth_qa.json` | `results/metrics/ablation_sprint2.json` **(kept)** |
| `scripts/merge_dataset.py` | `ground_truth_qa.json`, `adversarial_queries.json`, `multihop_queries.json`, `paraphrase_queries.json` | `ground_truth_qa_v2.json` |
| `scripts/remap_ground_truth_chunk_ids.py` | `holdout_eval_v{2,3}.json`, `train_eval_v{2,3}.json` | remapped splits |
| `scripts/run_eval_ragas.py` | `ground_truth_qa.json` | `results/metrics/ragas_phase2_*.json` **(kept)** |
| `scripts/split_dataset.py` | `ground_truth_qa.json`, `holdout_eval.json`, `train_eval.json` | train/holdout splits |
| `scripts/stress_test_20.py` | `ground_truth_qa_rebuilt.json`, `multihop_queries.json`, `paraphrase_queries.json` | `results/debug/stress_test_summary.json` **(kept)** |
| `scripts/rebuild_gold_labels.py` | `ground_truth_qa.json` (gone) | `ground_truth_qa_rebuilt.json` (gone) |
| `scripts/split_rebuilt_dataset.py` | `ground_truth_qa_rebuilt.json` (gone) | train/holdout splits (gone) |
| `scripts/build_eval_dataset.py` | — | `data/processed/ground_truth_qa_v2.json` (gone) |
| `scripts/build_ground_truth.py` | — | `data/processed/ground_truth_qa.json` (gone) |

`build_eval_dataset.py` and `build_ground_truth.py` are archived for a different
reason than the rest: they are not broken, they are **superseded**. Both write
`data/processed/ground_truth_qa*.json`, the pre-migration label files that no
longer exist. `scripts/build_retrieval_eval_dataset.py` produces the canonical
`data/evaluation/` labels instead, and `scripts/build_ground_truth.py` is its
predecessor.

Artifacts marked **(kept)** still exist in the repo. They are historical
evidence produced when the input data was intact, and they are retained. See
`results/metrics/k2_ragas_legacy_lineage.md` for why the RAGAS ones can no
longer be reproduced.

## Deliberately NOT archived

| File | Reason kept |
|---|---|
| `scripts/run_single_query.py` | Also reads `data/processed/ground_truth_qa.json`, but **only behind the `--from_gt` flag** (line 1049). Its default path, `--query`, and `--dataset` modes all work. One dead flag does not make the tool dead. |
| `scripts/run_eval.py` | Was dead for the same reason until 2026-10-10, when it was rewritten against `data/evaluation/` through `dataset_records()`. This is the replacement path. |
| `scripts/coverage_metric.py`, `scripts/run_eval_phase2.py` | Read a missing file but are multi-stage diagnostics with their own evidence trail. Left in place pending a separate decision; not claimed as runnable. |

## The rule this exposed

Nothing detected that these eight were unrunnable. The test suite does not
execute scripts, the CI job pointed at a *different* directory, and 221 green
tests said nothing about any of it.

The replacement gate is `scripts/check_ci_contracts.py`, which validates every
labeled dataset in `data/evaluation/` against the corpus. It would not have
caught these eight directly — they read paths outside `data/evaluation/` — but
it establishes the habit: a file that cannot find its data is a defect that
gets reported, not a trap that waits to be stepped in.

---

*Archived as part of the codebase-alignment audit. Evidence:
`results/metrics/k2_ragas_legacy_lineage.md`,
`scripts/check_ci_contracts.py`.*