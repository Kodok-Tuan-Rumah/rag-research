# Phase 1 Re-acceptance Investigation and Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the active v1/v2/v3 Phase 1 dataset migration, reproduce the two known answerable failures, and produce either an evidence-backed Phase 1 pass or a documented fail without changing labels merely to make the gate green.

**Architecture:** Treat the current v1 report as historical evidence, validate the untracked v3 dataset/corpus as the candidate acceptance target, make runner/test/report paths explicit, then run the full 20-case path and review all 15 answerable cases. Only after the gate result is known should the status docs be updated and committed.

**Tech Stack:** Python 3.14, pytest, Qdrant, Groq, `scripts/phase1_smoke.py`, `scripts/run_single_query.py`, Markdown status docs.

**Spec:** [`PROJECT_GOALS.md`](../../../PROJECT_GOALS.md#phase-1--pipeline-stabilization), [`docs/evaluation_guide.md`](../../evaluation_guide.md), current report `results/metrics/phase1_smoke.json`.

> **P0 execution status (2026-10-22):** Tasks 1–4 complete. Focused Phase 1 tests pass; the v3 collection holds 642 vectors. `results/metrics/phase1_smoke_v3.json` exists with `runtime_pass: true` but `human_review_complete: false`; Task 5 review and the gate check remain. See the SDD ledger for exact evidence.

---

## Global Constraints

- Do not edit `results/metrics/phase1_smoke.json` or any historical `human_review` labels to force a pass.
- Do not modify v1 or v2 datasets in place. If labels change, create a new version with provenance and preserve the old files.
- Review all 15 answerable cases. A pass requires at least 18/20 correct final decisions and **zero** confident-wrong answers; all failed reviews need a valid category.
- Do not claim `phase1_pass: true` until `--check-report` exits `0` and prints `phase1_pass: true`.
- Use explicit `--dataset` and `--check-report` paths. Do not rely on the runner default while the v1/v2/v3 migration is unresolved.
- Keep the report, dataset fingerprint, status docs, and any runtime fix in one atomic change set. Update the root submodule pointer only after the submodule commit exists.
- Do not start Phase 2 evaluator work until the Phase 1 target and result are settled.

## Current Evidence Snapshot (2026-10-22)

| Item | Current state |
|---|---|
| `results/metrics/phase1_smoke.json` | v1 report, `runtime_pass: true`, `human_review_complete: true`, `18/20`, `confident_wrong_answers: 1`, `quality_pass: false`, `phase1_pass: false` |
| `scripts/phase1_smoke.py` | Defaults to `phase1_smoke_set_v3.json`; defaults output to `results/metrics/phase1_smoke_v3.json` |
| `results/metrics/phase1_smoke_v3.json` | Missing |
| `data/evaluation/phase1_smoke_set_v3.json` | Untracked, 20 cases, 15 answerable, 5 abstain, expects 642 vectors |
| `data/processed/chunks_semantic_v3.json` | Untracked, 642 chunks, SHA-256 `f3068181943f850762a8e66856b802f56c440ce41f192bb03ac12dfb54a3754b` |
| `Makefile` `phase1-check` | Points to missing `results/metrics/phase1_smoke_v3.json` |
| `tests/test_phase1_smoke.py` | Currently fails: expects `phase1-smoke-v1`, loader returns `phase1-smoke-v3` |
| Status docs | Corrected to the v1 report; they must not be committed until the v3 decision is settled |
| `p1_answer_14` | v1 gold `STAR` is wrong; v3 gold is `NPPrompt` with evidence chunk `tier2_self_instruct_rs_0037` |
| `p1_answer_15_attention` | v1 had `false_abstention`; v3 uses the specific self-attention question and gold chunk `tier3_attention_all_you_need_rs_0008` |

**Current test evidence:**

```text
$ python3 -m pytest tests/test_phase1_smoke.py -q
F.........
1 failed, 9 passed
AssertionError: assert 'phase1-smoke-v3' == 'phase1-smoke-v1'
```

## Review Focus

These are the failure modes most likely to invalidate a claimed pass:

1. **Dataset drift:** runner defaults to v3 while the report and docs describe v1.
2. **Corpus drift:** Qdrant collection and BM25 corpus can silently come from v2 while gold IDs point to v3.
3. **Gold-label drift:** `p1_answer_14` in v1 rewards `STAR` even though the source text names `NPPrompt`.
4. **False abstention:** `p1_answer_15_attention` may still be rejected after a correct retrieval decision.
5. **Reranker illusion:** `use_reranker: true` with null scores makes the method label look active when it is not.
6. **Provider failure classification:** JSON-validation failures may be counted as clean abstentions.

---

## Decision Gate D1: Choose the acceptance target

**Recommendation:** use v3 as the candidate acceptance target because the working tree already defaults to it, it maps gold IDs to the 642-chunk corpus, and it corrects `p1_answer_14`. Keep v1 and v2 frozen as historical evidence.

- [ ] Confirm with the human owner that v3 is the intended target; if not, stop and switch the plan back to the chosen version.
- [ ] Record the decision in this plan and in `docs/evaluation_guide.md`.
- [ ] Do not edit the v1 report or relabel v1 data to make it pass.

**Acceptance:** the chosen dataset version, corpus path, Qdrant collection, and report path are written down explicitly in the plan.

---

### Task 1: Freeze the current state before any remediation

**Files:** none.

- [ ] Capture the current report summary and hashes:

```bash
git -C rag-research rev-parse HEAD
git -C rag-research status --short
sha256sum rag-research/results/metrics/phase1_smoke.json
python3 - <<'PY'
import json
d = json.load(open('rag-research/results/metrics/phase1_smoke.json'))
print(json.dumps(d['summary'], indent=2))
PY
```

- [ ] Capture the v3 dataset/corpus identity:

```bash
sha256sum rag-research/data/processed/chunks_semantic_v3.json
python3 - <<'PY'
import json
d = json.load(open('rag-research/data/evaluation/phase1_smoke_set_v3.json'))
print(d['dataset_version'], len(d['records']), d['expected_qdrant_vectors'])
PY
```

**Acceptance:** the old report and the candidate v3 inputs are identifiable by commit/path/hash before any new run.

---

### Task 2: Make the test and command surface agree on v3

**Files:**
- Modify: `tests/test_phase1_smoke.py`
- Modify: `scripts/phase1_smoke.py` (docstring commands only; behavior already defaults to v3)
- Modify: `Makefile` (only if the target path changes)
- Modify: `docs/evaluation_guide.md` (commands and dataset path after D1)
- Modify: `data/evaluation/README.md` (dataset pointer after D1)

**Interfaces:**
- `load_cases(Path("data/evaluation/phase1_smoke_set_v3.json"))` must return `dataset_version == "phase1-smoke-v3"` and 20 records.
- The report path used by `make phase1-check` must exist by the end of Task 5.

- [ ] Update `test_phase1_acceptance_set_is_frozen_and_balanced` to assert `phase1-smoke-v3`, 20 cases, 15 answerable, 5 abstain, and `expected_qdrant_vectors == 642`.
- [ ] Update the `phase1_smoke.py` docstring examples to use `results/metrics/phase1_smoke_v3.json` and the explicit v3 dataset path.
- [ ] Update `docs/evaluation_guide.md` and `data/evaluation/README.md` to name v3 as the candidate target, not v1/v2 as the current run target.
- [ ] Run:

```bash
cd rag-research
python3 -m pytest tests/test_phase1_smoke.py -q
```

Expected: `10 passed`.

**Acceptance:** the test suite no longer contradicts the runner default, and every documented command points at an existing target.

---

### Task 3: Validate v3 dataset, gold IDs, and corpus hash without calling an LLM

**Files:**
- Read: `data/evaluation/phase1_smoke_set_v3.json`
- Read: `data/processed/chunks_semantic_v3.json`
- Test: `tests/test_phase1_smoke.py`

- [ ] Run the loader's validation path:

```bash
cd rag-research
python3 - <<'PY'
from pathlib import Path
from scripts.phase1_smoke import load_cases
d = load_cases(Path('data/evaluation/phase1_smoke_set_v3.json'))
print(d['dataset_version'], len(d['records']), d['expected_qdrant_vectors'])
PY
```

Expected: `phase1-smoke-v3 20 642`.

- [ ] Assert every answerable `gold_chunk_id` and every `gold_evidence_chunk_ids` entry exists in the v3 corpus.
- [ ] Assert `p1_answer_14` has `gold_answer == "NPPrompt"` and its evidence chunk contains `NPPrompt`, not a claim that `STAR` is the answer.
- [ ] Assert `p1_answer_15_attention` has its specific self-attention question and evidence chunk `tier3_attention_all_you_need_rs_0008`.
- [ ] Run the focused test:

```bash
cd rag-research
python3 -m pytest tests/test_phase1_smoke.py -q
```

Expected: pass.

**Acceptance:** v3 has no missing gold chunks and the two known data defects are pinned by tests.

---

### Task 4: Index the v3 corpus into an isolated Qdrant collection

**Files:**
- Use: `scripts/index_phase1_corpus.py`
- Read: `data/processed/chunks_semantic_v3.json`

- [ ] Create or reuse the isolated collection:

```bash
cd rag-research
python3 scripts/index_phase1_corpus.py \
  --chunks data/processed/chunks_semantic_v3.json \
  --collection rag_research_phase1_v3
```

Expected: `vector_count == 642` for collection `rag_research_phase1_v3`.

- [ ] Verify that the default/current collection is not silently used:

```bash
grep -RIn "QDRANT_COLLECTION" scripts/phase1_smoke.py scripts/run_single_query.py src/retrieval/qdrant_store.py
```

Expected: the acceptance command sets `QDRANT_COLLECTION=rag_research_phase1_v3`.

**Acceptance:** the v3 run has an isolated index with exactly 642 vectors.

---

### Task 5: Run the full v3 Phase 1 acceptance and review all 15 answerable cases

**Files:**
- Write: `results/metrics/phase1_smoke_v3.json`
- Review: `results/metrics/phase1_smoke_v3.json`

- [ ] Run with both corpus selectors explicit:

```bash
cd rag-research
QDRANT_COLLECTION=rag_research_phase1_v3 \
BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json \
python3 scripts/phase1_smoke.py \
  --dataset data/evaluation/phase1_smoke_set_v3.json \
  --output results/metrics/phase1_smoke_v3.json
```

Expected: 20 runtime rows, `runtime_pass: true`, review pending.

- [ ] Review all answerable rows:

```bash
cd rag-research
python3 scripts/phase1_smoke.py \
  --review-report results/metrics/phase1_smoke_v3.json
```

- [ ] Check the gate:

```bash
cd rag-research
python3 scripts/phase1_smoke.py \
  --check-report results/metrics/phase1_smoke_v3.json
echo "EXIT_CODE=$?"
```

**Acceptance:** a complete v3 report exists with all 15 answerable cases reviewed. Do not claim pass before the checker exits `0`.

---

### Task 6: Triage any remaining answerable failures

**Files:**
- Read: `src/generation/generator.py`
- Read: `src/controller/agent.py`
- Read: `scripts/run_single_query.py`
- Test: `tests/test_phase1_guardrails.py`, `tests/test_agent.py`

- [ ] For each failed row, print `status`, `confidence_decision`, `generation_attempts`, `abstain_reason`, `chunks_used`, `prompt_chunk_ids`, and `generation_contexts`.
- [ ] For `p1_answer_15_attention`, verify whether the final answer's material claims are actually supported by the chunks sent to the generator. If the answer is a correct definition but rejected by `check_answer_evidence`, add a focused regression test with the v3 gold chunk text.
- [ ] For `p1_answer_14`, verify the v3 answer names `NPPrompt`; if it names `STAR`, the old v1 question/gold is still leaking into the run path.
- [ ] For any provider JSON-validation failure, inspect `generator.py`'s error-to-abstain mapping and add a test that distinguishes a clean abstention from a provider error.
- [ ] For any reranker row with `use_reranker: true`, assert whether `reranker_scores` are non-null and whether the reranker latency is plausible. A null score set means the stage did not run.

**Acceptance:** every failure has either a root-cause note plus regression test, or a documented data-label correction in a new dataset version.

---

### Task 7: Re-run and finalize the acceptance evidence

**Files:**
- Modify: `results/metrics/phase1_smoke_v3.json` only through a fresh run/review.
- Modify: `README.md`, `PROJECT_GOALS.md`, `PROJECT_CONTEXT.md`, `docs/evaluation_guide.md`.

- [ ] Re-run Task 5 after fixes, then run `--check-report` again.
- [ ] If pass: update the docs to the v3 report values and remove the v1 report as the current target.
- [ ] If fail: keep `phase1_pass: false`, document the exact failing cases and root causes, and do not start Phase 2.
- [ ] Run:

```bash
cd rag-research
python3 scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke_v3.json
git diff --check
python3 -m pytest tests/test_phase1_smoke.py -q
```

**Acceptance:** the report, docs, test suite, and Makefile all point to the same dataset/report; a pass claim is supported by exit `0`.

---

### Task 8: Commit atomically and update the root submodule pointer

**Files:**
- Commit in `rag-research`: dataset version, corpus/chunk metadata if needed, scripts/tests, report, status docs.
- Commit in the root workspace: updated `rag-research` pointer.

- [ ] Review `git -C rag-research status --short` and stage only the Phase 1 change set.
- [ ] Commit with a message that names the dataset version and report path.
- [ ] Update the root submodule pointer only after the submodule commit exists.
- [ ] Re-run the checker from the committed submodule state.

**Acceptance:** a fresh checkout can reproduce the reported result using the committed dataset, corpus, report, and docs.

---

## Deferred: Phase 2

Do not start this until Phase 1 has a settled result.

- [ ] Inspect `scripts/run_eval_phase2.py`: it calls `agent.run(query)` and then `retriever.search(query, top_k=10)` again for `retrieved_ids`. Verify that Recall@10/MRR@10 use the chunks actually used by the agent, or remove the duplicate retrieval.
- [ ] Do not use Phase 2 results to retroactively replace the frozen Phase 1 acceptance result.
