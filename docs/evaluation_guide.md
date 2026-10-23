# Evaluation Guide

This guide separates retrieval benchmarking from the Phase 1 runtime and answer
quality gate. A retrieval score is not a substitute for generation, abstention,
or human review.

## Evaluation tracks

| Track | Dataset | What it measures | Main entrypoint |
|---|---|---|---|
| Retrieval benchmark | `data/evaluation/retrieval_eval_dataset_v2.json` | Retrieval ranking against labeled gold chunks (13-query K7 benchmark) | `python scripts/run_retrieval_eval.py` |
| Phase 1 acceptance | `data/evaluation/phase1_smoke_set_v3.json` | Full runtime contract, answer/abstention decisions, and reviewed grounding (20 cases: 15 answerable, 5 abstain) | `python scripts/phase1_smoke.py --dataset data/evaluation/phase1_smoke_set_v3.json` |
| Phase 2 pipeline metrics | `data/evaluation/retrieval_eval_dataset_v4.json` | Same-run answer, retrieval, abstention, and failure metrics; v4 is currently an incomplete expansion candidate | `python scripts/run_eval_phase2.py` |
| Optional answer-quality judge | A pipeline report produced by `run_eval.py` | RAGAS metrics; separate from the deterministic Phase 2 gate | `python scripts/run_eval.py` |

The Phase 1 set is versioned independently. Do not merge its outcomes into
retrieval metrics or change its labels to make a run pass. If its content needs
to change, create a new dataset version and preserve the previous report and
dataset provenance.

## Phase 1 acceptance workflow

1. Ensure Qdrant is reachable with the intended collection and corpus, and
   provide the Groq API credential through the project's configured environment.
   For v3, both retrieval paths must read the same snapshot:
   `QDRANT_COLLECTION=rag_research_phase1_v3` and
   `BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json` (642 chunks).
2. Run the complete acceptance set with generation enabled:

   ```bash
   QDRANT_COLLECTION=rag_research_phase1_v3 \
   BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json \
   python scripts/phase1_smoke.py \
     --dataset data/evaluation/phase1_smoke_set_v3.json \
     --output results/metrics/phase1_smoke_v3.json
   ```

   The runner archives an existing report before replacing it. The report
   records dataset identity, runtime gates, per-case outputs, and the summary.
3. Review every answerable case interactively:

   ```bash
   python scripts/phase1_smoke.py --review-report results/metrics/phase1_smoke_v3.json
   ```

   Review may be resumed. Mark `answer_correct=yes` only when the answer
   responds to the question and contains the essential gold facts. Mark
   `grounded=yes` only when every material claim is supported by retrieved
   context; the presence of a gold chunk alone is insufficient. Record the
   reviewer identifier. For any incorrect or ungrounded answer, add at least
   one failure category and useful notes. Do not mark a response correct merely
   because its decision status was `ANSWERED`.
4. Check the complete report:

   ```bash
   python scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke_v3.json
   ```

   Phase 1 passes only when all runtime/dataset/Qdrant gates pass, all required
   answerable cases are reviewed, at least 18/20 final decisions are correct,
   and there are zero wrong or ungrounded answers with effective `GENERATE`
   decisions. Failed reviews must include a failure reason. The checker output
   and report are the evidence; do not infer a pass from a successful process
   exit during the generation run alone.
5. Run the five `sanity_check_answer()` guardrail scenarios (short answer,
   verbatim chunk copy, high overlap, normal grounded answer, and empty answer)
   through the project's test suite when verifying implementation changes.
   These focused checks supplement the acceptance report and do not replace it.
6. Update `PROJECT_GOALS.md` and the status summaries from the checker output,
   whether it passes or fails. Claim a pass only when the report records
   `phase1_pass: true`. Preserve the report and dataset fingerprint.

## Current recorded status

### v3 candidate target — not yet accepted

The Phase 1 runner and `make phase1-check` now target the v3 dataset and the
642-chunk semantic v3 corpus. `results/metrics/phase1_smoke_v3.json` has not
been produced yet, so there is no v3 acceptance evidence. Do not claim a Phase 1
pass until the 20-case run is complete, all 15 answerable cases are reviewed,
and `--check-report` exits `0` with `phase1_pass: true`.

### Historical v1 report (2026-10-22)

The archived 2026-10-15 report passed the frozen v1 acceptance gate; it does
not establish production readiness or confidence calibration. A later v1
full-generation run was completed on 2026-10-22 against the v1 dataset and the
796-chunk v2 corpus fingerprints. Its 20 runtime rows pass, including the live
Qdrant collection/vector-count checks, and review of all 15 answerable cases is
complete (`human_review_complete: true`). The reviewed result is
`correct_final_decisions: 18/20` with `confident_wrong_answers: 1` against
`maximum_confident_wrong: 0`, so that report records `quality_pass: false` and
`phase1_pass: false`.

The v1 answerable failures are `p1_answer_14` (`answer_correct: false`,
`grounded: false`, categories `answer_incorrect` and `unsupported_claim`; the
reviewer notes the original gold may be wrong) and `p1_answer_15_attention`
(`answer_correct: false`, category `false_abstention`, result status
`INSUFFICIENT_CONTEXT`). `p1_answer_03` is currently reviewed as correct. The
v1 report also records `use_reranker: true` while every `reranker_scores` value
across all 20 records is null and reranker latency is sub-millisecond; do not
claim the reranker stage was effectively exercised. This v1 report is
uncommitted and remains historical evidence for the v1 dataset; it is not the
v3 acceptance result. Re-run and complete acceptance after any code or corpus
change that could affect runtime outputs.

## Phase 2 evaluation workflow

The deterministic Phase 2 evaluator is `scripts/run_eval_phase2.py`. It reads
versioned dataset envelopes through `dataset_records()`, validates labels
against the selected corpus before model calls, and calculates retrieval scores
from the chunk IDs returned by the same FastAPI `/generate` request used to
produce the answer. Runtime errors remain per-query report rows and count against
decision accuracy; they are not silently dropped. The optional LLM-judge path
in `scripts/run_eval.py` is separate.

Inspect the dataset and corpus without loading model clients or calling an LLM:

```bash
python scripts/run_eval_phase2.py --validate-only
```

Run a small API sample to inspect the report shape (requires the runtime
dependencies, Qdrant, and Groq credentials; requests may incur provider cost):

```bash
python scripts/run_eval_phase2.py --limit 3
```

Run the complete Phase 2 evaluation only after the dataset meets the size,
category, label, and category-balance gates:

```bash
python scripts/run_eval_phase2.py --require-phase2-gate
```

The checked-in v4 dataset currently has 112 rows and is missing the multihop
category and the 20% category floor. It is therefore useful for schema/data
validation and incremental work, but it is not yet eligible for a Phase 2 pass.

## Phase 2 entry

Start Phase 2 only after every Phase 1 exit criterion in
[`PROJECT_GOALS.md`](../PROJECT_GOALS.md#phase-1--pipeline-stabilization) passes.
Phase 2 expands evaluation infrastructure and labeled coverage; it does not
retroactively replace the frozen Phase 1 acceptance result.
