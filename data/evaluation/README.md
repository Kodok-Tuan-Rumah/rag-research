# `data/evaluation/` — Dataset Retrieval dan Acceptance

> **Purpose distinction:** “canonical” below means canonical for retrieval
> metric comparisons. Phase 1 has a separate frozen acceptance set for full
> generation, abstention, runtime contracts, and human review; it is not part
> of the retrieval dataset family and must not be pooled into retrieval scores.
> See [`phase1_smoke_set_v1.json`](phase1_smoke_set_v1.json) and
> [`docs/evaluation_guide.md`](../../docs/evaluation_guide.md).

Read this before pointing an evaluation script at a file here. All five
datasets validate against the corpus (`scripts/check_ci_contracts.py`), but
they are **not interchangeable** — they use different gold-label schemas, and
one of them is the reference every recorded metric was produced with.

## Which one to use

| File | n | Gold field | Use it for |
|---|---:|---|---|
**`retrieval_eval_dataset_v2.json`** | 13 | `ground_truth` + `gold_chunk_ids` | **Canonical.** Every baseline in `results/metrics/k1_baseline_freeze.json` was measured on this file, and `scripts/run_retrieval_eval.py` defaults to it |
`retrieval_eval_dataset.json` | 14 | `gold_answer` + `gold_chunk_id` | Legacy v1. Superset of v2 by one query (q15). Kept because some artifacts name it |
`retrieval_eval_dataset_v3.json` | 10 | `gold_answer` + `gold_chunk_id` | Exploratory. Different query ids (`esti*`), not derived from v1 |
`retrieval_train_eval.json` | 8 | legacy | §10.1 train split |
`retrieval_holdout_eval.json` | 6 | legacy | §10.1 holdout split — the only file §10.1 permits for final reported metrics |

**v2 is the only one with a version envelope**, recording `dataset_version`,
`sha256` of itself and of both corpora, and the files it was built from. That is
what makes it the one a metric can be pinned to.

```
corpus_chunks_sha256     4375a26668366674473a8917a3d0bc4ce7a2e53ace913457b80ff09be3481ba6
corpus_documents_sha256  9048aca17d69289a7174edc3d938ea2d79f03896d3900deba76f9fb6c7f0dc8a
```

`scripts/check_ci_contracts.py` verifies both on every CI run. If they ever
mismatch, every metric recorded against this corpus is invalid — regenerate the
dataset or restore the corpus from git.

## The two gold schemas do not match

| v1 / v3 / train / holdout | v2 |
|---|---|
`gold_chunk_id` (single string) | `gold_chunk_ids` (list) |
`gold_answer` | `ground_truth` |

`normalize_gold_chunk_ids()` accepts **exactly one** of the two, and raises on
anything else. Three consequences that have already cost time:

- `gold_data_validator` also accepts the legacy aliases `source_chunk`,
  `source_chunk_id`, `chunk_id`, `gold_chunk` so old datasets can still be
  audited. **`normalize_gold_chunk_ids` does not.** A row can pass validation and
  then abort `run_retrieval_eval.py`, which calls it with no `try/except`.
  `check_ci_contracts.py` tests the evaluator's contract separately for exactly
  this reason.
- `run_eval.py` (`_ground_truth`) reads `ground_truth` then falls back to
  `gold_answer`, so it handles both.
- Always pass `dataset_records()` when loading. v1/v3/train/holdout are bare
  JSON arrays; v2 is `{"records": [...], ...envelope}`.

## Data integrity rules

`src/evaluation/gold_data_validator.py` — run it after editing any file here:

```bash
python -m src.evaluation.gold_data_validator \
  --dataset data/evaluation/retrieval_eval_dataset_v2.json \
  --chunks data/processed/chunks_semantic_v2.json \
  --require-answer
```

Exit code is non-zero if any row is an `evaluation_data_error`. The validator
takes an exact-string membership check; the evaluator uses
`gold_matching.chunk_ids_match`, which also accepts a `_sub` suffix in either
direction. Those are not the same rule — a gold id that the validator accepts
can still be scored differently by the evaluator.

A gold id absent from the corpus is no longer silently counted as a retrieval
miss. It is classified `evaluation_data_error` and excluded from the metric
denominator, so a labelling mistake can neither lower nor raise a score.

## `data/adversarial/` is not here

`data/adversarial/adversarial_queries.json` holds query-shaped rows with
`should_abstain` and no reference answers. It is used by
`run_retrieval_eval.py --include-adversarial`. It is **not** a labeled dataset
and cannot be fed to RAGAS — the previous CI job did exactly that and could
never pass.

## Corpus dependency

All five datasets were labeled against `data/processed/chunks_semantic_v2.json`
(796 chunks, ids `{doc_id}_rs_0000`). That file is produced by
`src/ingestion/semantic_chunker.py`. The legacy `src/ingestion/chunker.py`
produces a **different** corpus (582 chunks, ids `{doc_id}_s0000`, references
stripped) and writes to `chunks_legacy_chunker.json`. Do not point anything here
at that file.

---

*Labels are verified against the source papers by manual comparison
(`verification_method` in the v2 envelope). No label was produced by running a
benchmark or a RAGAS evaluation.*
