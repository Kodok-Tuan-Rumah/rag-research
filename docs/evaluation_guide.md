# Evaluation Guide

This guide separates retrieval benchmarking from the Phase 1 runtime and answer
quality gate. A retrieval score is not a substitute for generation, abstention,
or human review.

## Evaluation tracks

| Track | Dataset | What it measures | Main entrypoint |
|---|---|---|---|
| Retrieval benchmark | `data/evaluation/retrieval_eval_dataset_v2.json` | Retrieval ranking against labeled gold chunks (13-query K7 benchmark) | `python scripts/run_retrieval_eval.py` |
| Phase 1 acceptance | `data/evaluation/phase1_smoke_set_v1.json` | Full runtime contract, answer/abstention decisions, and reviewed grounding (20 cases: 15 answerable, 5 abstain) | `python scripts/phase1_smoke.py` |

The Phase 1 set is versioned independently. Do not merge its outcomes into
retrieval metrics or change its labels to make a run pass. If its content needs
to change, create a new dataset version and preserve the previous report and
dataset provenance.

## Phase 1 acceptance workflow

1. Ensure Qdrant is reachable with the intended collection and corpus, and
   provide the Groq API credential through the project's configured environment.
2. Run the complete acceptance set with generation enabled:

   ```bash
   python scripts/phase1_smoke.py
   ```

   The runner archives an existing report before replacing it. The report
   records dataset identity, runtime gates, per-case outputs, and the summary.
3. Review every answerable case interactively:

   ```bash
   python scripts/phase1_smoke.py --review-report results/metrics/phase1_smoke.json
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
   python scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke.json
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
6. Update `PROJECT_GOALS.md` and the status summaries only after the report
   records `phase1_pass: true`. Preserve the report and dataset fingerprint.

## Current recorded status

The checked-in `results/metrics/phase1_smoke.json` contains 20 cases, full
generation, complete human review, and records `runtime_pass: true`,
`quality_pass: true`, and `phase1_pass: true`. The checker was rerun and returned
the same passing result on 2026-10-15. This passes the frozen Phase 1 acceptance
gate; it does not establish production readiness or confidence calibration.
A follow-up spot check was suggested for grounding judgments on
`p1_answer_03` and `p1_answer_15_attention`, because some answer claims may be
broader than evidence included in the prompt. If review changes either
annotation, rerun `--check-report` and update the recorded status from its output.
Re-run and complete acceptance after any code or corpus change that could affect
runtime outputs.

## Phase 2 entry

Start Phase 2 only after every Phase 1 exit criterion in
[`PROJECT_GOALS.md`](../PROJECT_GOALS.md#phase-1--pipeline-stabilization) passes.
Phase 2 expands evaluation infrastructure and labeled coverage; it does not
retroactively replace the frozen Phase 1 acceptance result.
