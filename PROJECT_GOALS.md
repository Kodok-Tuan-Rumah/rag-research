# RAG SYSTEM — ENGINEERING GOALS & PRODUCTION SPEC
> **Version:** 2.2
> **Repo:** `rag-research`  
> **Status:** Historical Phase 1 snapshot passed on 2026-10-15. The v3 current-runtime run completed on 2026-10-26 and passed runtime gates, but has four provider errors due to exhausted Groq quota and has not had human review (`quality_pass: false`, `phase1_pass: false`). Phase 1 remains open; Phase 2 work is not an acceptance substitute.
> **Last Updated:** 2026-10-26

> **Current evidence note:** The K7.10–K7.13 retrieval audits use the 13-query `data/evaluation/retrieval_eval_dataset_v2.json`. This is distinct from the older 14-query `data/evaluation/retrieval_eval_dataset.json` summarized in the historical README baseline. The 13-query K7 set does not satisfy the Phase 2 target of at least 200 labeled samples, and its results do not establish production readiness or confidence calibration.

> **Phase 1 evidence:** The archived 2026-10-15 report records a passed acceptance for the older frozen set. The current candidate is `data/evaluation/phase1_smoke_set_v3.json` against the 642-chunk v3 corpus and isolated Qdrant collection. The working-tree report `results/metrics/phase1_smoke_v3.json` completed all 20 runtime rows (`runtime_pass: true`), but four rows are provider errors following Groq token-quota exhaustion. Human review is pending (`human_review_complete: false`), so no final quality score is available and `quality_pass`/`phase1_pass` are false. See [`docs/evaluation_guide.md`](docs/evaluation_guide.md).

---

## CORE PRINCIPLE



> **If retrieval is wrong, no architecture will save generation.**  
> Data quality supersedes model complexity. Every time.

---

## TABLE OF CONTENTS

1. [System Objective](#1-system-objective)
2. [Success Criteria — Measurable](#2-success-criteria--measurable)
3. [Architecture & Component Contracts](#3-architecture--component-contracts)
4. [Failure Mode Registry](#4-failure-mode-registry)
5. [Confidence & Calibration Spec](#5-confidence--calibration-spec)
6. [Latency Budget](#6-latency-budget)
7. [Roadmap — Phase-Gated Execution](#7-roadmap--phase-gated-execution)
8. [Production Readiness Checklist](#8-production-readiness-checklist)
9. [Non-Goals](#9-non-goals)
10. [Evaluation Integrity Constraints](#10-evaluation-integrity-constraints)
11. [Grounding Strictness Specification](#11-grounding-strictness-specification)
12. [Confidence System Constraints](#12-confidence-system-constraints)
13. [Load & Throughput Behavior](#13-load--throughput-behavior)
14. [Change Safety Protocol](#14-change-safety-protocol)
15. [Dataset & Index Versioning](#15-dataset--index-versioning)
16. [Adversarial Robustness](#16-adversarial-robustness)
17. [Proposed Product Differentiators — "Forensic AI"](#17-proposed-product-differentiators--forensic-ai)

---

## 1. System Objective

Build a **reliable, failure-aware Retrieval-Augmented Generation system** for research-domain QA that:

- Maximizes answer correctness (Exact Match + semantic groundedness)
- Enforces a **hard abstention policy** when retrieved context is insufficient
- Provides **structured failure observability** at every pipeline stage
- Operates within defined latency and cost envelopes per query

This is not a chatbot. This is an **engineering-grade QA system** where answer reliability is the primary constraint.

| Reranker latency bottleneck | High | CPU-only constraint — cannot resolve without GPU | Phase 5 |

---

## 2. Success Criteria — Measurable

### 2.1 Answer Quality

| Metric | Target | Measurement Protocol | Status |
|---|---|---|---|
| Exact Match (EM) | ≥ 70% | Token-level F1 after normalization (lowercase, strip punct, remove articles). NOT raw string match. | `IMPLEMENTED` — see below |
| Semantic F1 | ≥ 0.75 | BERTScore F1 against ground truth, averaged over eval set | `TODO` — needs `bert-score` |
| Faithfulness | ≥ 0.88 | RAGAS faithfulness — NLI entailment between answer and retrieved chunks | `RUNNABLE` — never run on the canonical set |
| Hallucination Rate | 0% | Definition: answer contains claim not entailed by any retrieved chunk. Measured via NLI (≥0.5 entailment threshold). Evaluated on full eval set per release. | `RUNNABLE` — never run on the canonical set |
| Answer Relevancy | ≥ 0.88 | RAGAS answer relevancy metric | `RUNNABLE` — never run on the canonical set |

**Status vocabulary change (2026-10-11).** `DONE` previously meant "implemented",
but these metrics were labelled `DONE` while RAGAS could not be imported at all
— `langchain-community` 0.4.2 had removed the module ragas needs, and nothing
in the suite imported it. `IMPLEMENTED` means the code exists and is tested.
`RUNNABLE` additionally means the code can execute today; it does **not** mean
a number has been produced. No row here has a current measurement.

**Exact Match (EM).** Implemented in `scripts/run_eval_phase2.py` as
`normalize()`, `exact_match()` and `token_f1()`, with all 20 normalization cases
covered by `tests/test_em_normalization.py`. It was blocked from running only by
a dataset path pointing at `data/processed/holdout_eval_v2.json`, which the
migration to `data/evaluation/` removed; that path now resolves to the canonical
v2 file. EM has never been measured.

A second, non-compliant implementation existed in
`scripts/run_single_query.py`: `answer.strip().lower() == ground_truth.strip().lower()`,
which violates the mandatory normalization rules below and scores 0 for answers
that are in fact exact. It now calls `run_eval_phase2.exact_match`.

**`token_f1` is not Semantic F1.** It is bag-of-token overlap and is the
measurement protocol for EM above. It cannot stand in for BERTScore: it scores 0
for any paraphrase, which is the case Semantic F1 exists to catch.

**EM Normalization Rules (mandatory):**
```
normalize(text):
  lowercase
  strip punctuation
  remove articles: ["a", "an", "the"]
  collapse whitespace
```

### 2.2 Retrieval Quality

| Metric | Target | Measurement Protocol | Status |
|---|---|---|---|
| Recall@10 | ≥ 90% | % of queries where gold chunk appears in top-10 retrieved. k=10 is fixed. | `WIP` |
| MRR@10 | ≥ 0.70 | Mean Reciprocal Rank at k=10 over eval set | `WIP` |
| Retrieval Miss Rate | ≤ 10% | 1 - Recall@10 | `WIP` |
| Ranking Error Rate | ≤ 15% | % of queries where gold chunk rank > 3 despite being retrieved. Historical measurement protocol: hybrid_rrf, top_k=10, candidate_k setting=50 | `WIP` |
| Chunk Diversity (intra-list) | ≥ 0.30 | Average pairwise cosine distance between top-5 retrieved chunks. Detects duplicate-dominated results. | `TODO` |

### 2.3 Reliability

| Metric | Target | Measurement Protocol | Status |
|---|---|---|---|
| Abstain Accuracy | ≥ 90% | Requires labeled eval set: each query tagged `should_answer` or `should_abstain`. Abstain accuracy = correct abstain decisions / total abstain-labeled queries. | `TODO` |
| False Abstain Rate | ≤ 10% | % of `should_answer` queries that were incorrectly abstained | `TODO` |
| No Confident Wrong Answers | Hard constraint | Any query where confidence ≥ threshold AND answer is factually wrong = **critical failure**. Zero tolerance. | `WIP` |

> **Note on Abstain Labels:** Ground truth abstain labels must be created manually or semi-automatically. Rule: if gold answer cannot be derived from any chunk in the index, label = `should_abstain`.

### 2.4 Performance

| Metric | Target | Notes |
|---|---|---|
| End-to-end latency (P95) | ≤ 3.5s | No-retry path target; revised from the superseded `<2s` aspiration. See [Section 6](#6-latency-budget). Not yet verified end-to-end against the current runtime model. |
| Retrieval latency (P95) | ≤ 600ms | Qdrant hybrid search + BM25 |
| Reranker latency (P95) | ≤ 700ms | Target only; current code reranks at most 20 candidates and reranker is opt-in. K7.11 CPU stage P95 was 1,003.52ms at @5 and 3,498.03ms at @20, so this target was not met in that run. |
| LLM generation latency (P95) | ≤ 1.8s | Target only; current configured model is `openai/gpt-oss-20b`. Re-measure on deployment hardware/provider. |
| Retry overhead (max) | ≤ 2.0s | Max 1 retry cycle. See retry contract in Section 3.4. |

### 2.5 Cost Efficiency

| Lever | Target | Implementation |
|---|---|---|
| Token budget per query | ≤ 2,000 input tokens | Context budgeting in `generator.py` |
| Chunk filtering | Layered score, content-quality, strict/fallback, and generation-time score gates | Implemented; see Section 3.3 for stage-specific rules |
| Retry cap | Max 1 retry per query | Hard limit — no infinite loops |
| Multi-query expansion | Disabled by default | Only enable when EM < 60% and MRR < 0.60 on eval |

---

## 3. Architecture & Component Contracts

### 3.1 Current Runtime Baseline (Source-Verified 2026-10-08)

```yaml
retrieval_strategy: hybrid_rrf
api_top_k_policy_defaults: factual=5, general=8, reasoning=10
api_top_k_request_override: 1..20
candidate_k_setting: 50   # typical computed pool is 40 when top_k=10
rrf_k: 60
embedding_model: sentence-transformers/all-MiniLM-L6-v2
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
reranker_default_enabled: false
reranker_input_cap: 20
hyde: disabled
multi_query: disabled
```

The cross-encoder is available as an explicit opt-in; it is not the production ranking default. The configured `candidate_k` setting is an upper bound used by `_choose_candidate_k`; with `top_k=10`, the current code typically retrieves/fuses 40 candidates, and reranking is capped at 20. Older ablation claims below are historical evidence, not this runtime configuration.

### 3.2 Query-Type Classifier

**Purpose:** Route query to correct retrieval config.  
**Classes:** `factual` | `reasoning` | `general`  
**Implementation:** Current runtime uses `PolicyEngine` keyword heuristics for factual/reasoning/general routing; classifier accuracy has not been established by the K7 retrieval audits.

| Class | Retrieval Config | Rationale |
|---|---|---|
| `factual` | precision mode — top_k=5, strict filtering | High precision needed, references allowed |
| `reasoning` | diversity mode — top_k=10, relaxed filtering | Need broader context coverage |
| `general` | balanced — top_k=8, standard filtering | Default fallback |

**Contract:**
- Classifier accuracy target: ≥ 85% on eval set (measured separately)
- Current routing is a deterministic keyword heuristic and does not produce a classifier-confidence score; a `<0.6` confidence fallback is not implemented. Unmatched queries route directly to `general`.
- Misclassification impact must be tracked as a failure subtype: `classification_error`

### 3.3 Retrieval Pipeline

Current runtime flow (reranking is opt-in):

```
Query
  │
  ├── Dense retrieval (embedding, Qdrant)  ──┐
  └── BM25 retrieval                       ──┤
                                             ▼
                                    Weighted RRF (k=60)
                                             │
                                    Score/noise prefilter
                                             │
                           Optional cross-encoder (max 20 inputs)
                                             │
                         Strict filter → minimum-filter fallback
                                             │
                         Context diversity/character limit
                                             │
                             Confidence decision → LLM
```

The configured candidate pool setting is 50; for `top_k=10`, `_choose_candidate_k` typically yields 40. Filtering can reduce the returned result count, so `top_k` is an upper bound rather than a guarantee. Reranking is disabled by default and receives at most 20 candidates when requested.

#### Stage-specific filtering gates

The runtime does not use one global minimum chunk length of 80 characters. Filtering is layered, and a chunk rejected at a stage is not available to later stages:

| Stage | Source rule | Effect |
|---|---|---|
| Post-RRF score gate | Keep `retrieval_score >= settings.min_score_threshold` (default `0.0`; configurable) | Removes low-scoring fused candidates before content checks. |
| Pre-reranker noise gate | `is_low_quality_chunk`: reject fewer than 10 words, text ending in `:`, or more than 10 commas with fewer than 15 words | Candidate removed before optional reranking and strict filtering. |
| Strict retrieval filter | Require at least 12 words, acceptable start/completeness, and not a fragment/title/author; reference chunks also require the query policy to allow references | Primary final-result path. The completeness heuristic accepts terminal punctuation or a sufficiently substantive clause. |
| Minimum-filter fallback | Used only when the strict path yields no final chunks; rejects empty text, disallowed references, title/author lines, and fragments, while relaxing the informative/length requirements | Can return lower-quality, shorter chunks; if this pool is also empty, retrieval abstains. |
| Generator score gate | Separately rejects when top score is `<= -3.0`; then stops adding ordered candidates when a score is `< -3.0` or the adjacent score gap exceeds `settings.score_gap_threshold` (default `15.0`) | Further limits the chunks used to build the generation context; this is not the retrieval `0.0` threshold. |

The retrieval score threshold is configured in `src/api/config.py`; the content predicates and strict/minimum rules are in `src/retrieval/chunk_filters.py`, and their ordering/fallback is in `src/retrieval/hybrid_retriever.py`. The generation-time score gate is implemented separately in `src/generation/generator.py`. Thresholds and predicates above describe current defaults and source behavior, not calibrated relevance guarantees.

### 3.4 Self-Reflection & Retry Contract

**This is not an infinite loop. It is a single bounded retry.**

```
Attempt 1: Standard retrieval → confidence decision
  │
  └── IF (decision = REJECT OR retrieval is unstable):
        Attempt 2 (ONE retry only):
          - requested top_k grows by 1.5x (integer truncation, at least +1,
            capped at 30)
          - dense/BM25 weights change from 0.5/0.5 to 0.3/0.7
          - pass min_score_threshold = -0.1
            (currently inert on the RRF score scale; see correction below)
          │
          └── If still rejected/unstable → abstain; never retry again
```

The retry occurs before generation. It does not retry a generated answer after
the post-generation evidence check. The API and Agent both allow one retry and
use the same retrieval adjustments; the retry is triggered by REJECT or an
`unstable_retrieval` signal.

**Hard constraints on retry:**
- Max iterations: **2** (initial + 1 retry)
- Max additional LLM calls from retry: **1**
- Max latency addition from retry: **2.0s**
- If the second attempt remains rejected or unstable: **ABSTAIN**

### 3.5 Context Control

- `reference_ratio`: max fraction of context that can be reference-section text (default: 0.3)
- `context_length_budget`: max tokens fed to LLM (default: 1,800 tokens)
- `sanity_check_answer()`: detects reference-dump answers (answer ≈ chunk verbatim)

---

## 4. Failure Mode Registry

Every query must produce a structured failure classification. No silent failures.

| Failure Type | Definition | Trigger Condition | Response Action | Logged |
|---|---|---|---|---|
| `retrieval_miss` | Gold chunk not in top-k results | Recall@10 = 0 for this query | Retry with BM25 boost | ✓ |
| `ranking_error` | Gold chunk retrieved but rank > 3 | Gold chunk in top-10 but not top-3 | Adjust reranker / top_k | ✓ |
| `filtering_error` | Good chunk dropped by quality filter | Post-filter chunk count < 3 | Relax filtering threshold | ✓ |
| `generation_fail` | LLM fails to produce structured output | JSON parse error or empty response | Retry with simplified prompt | ✓ |
| `classification_error` | Query misrouted by query-type classifier | Detected via answer quality signal | Log for classifier retraining | ✓ |
| `confidence_miss` | System abstains on answerable query | False abstain (should_answer but abstained) | Adjust confidence threshold | ✓ |
| `hallucination` | Answer contains claim not in context | NLI entailment < 0.5 for any claim | **Hard block — never surface to user** | ✓ |
| `latency_breach` | Query exceeds P95 latency target | End-to-end > 3.5s | Log + degrade gracefully (skip reranker) | ✓ |

**Per-query log schema (JSON):**
```json
{
  "query_id": "uuid",
  "query_text": "...",
  "query_type": "factual|reasoning|general",
  "retrieval_scores": [...],
  "reranker_scores": [...],
  "chunks_used": [...],
  "confidence_score": 0.0,
  "failure_type": "none|retrieval_miss|...",
  "answer_status": "answered|abstained|failed",
  "latency_ms": {
    "retrieval": 0,
    "reranker": 0,
    "generation": 0,
    "total": 0
  },
  "retry_triggered": false
}
```

---

## 5. Confidence & Calibration Spec

### 5.1 Current State
The confidence engine **exists but is not calibrated**. This is the single highest-priority blocker. All abstain/answer decisions are currently based on uncalibrated heuristic scores.

### 5.2 Calibration Protocol (Required Before Phase 3 Exit)

**Step 1 — Collect raw confidence vs actual correctness pairs**
```
For each query in eval set:
  raw_confidence = system confidence score (0–1)
  actual_correct = 1 if EM > 0 else 0
  → save (raw_confidence, actual_correct)
```

**Step 2 — Fit calibration model**
- Method: **Isotonic Regression** (preferred over Platt scaling for non-monotonic scores)
- Library: `sklearn.isotonic.IsotonicRegression`
- Minimum sample size: 100 labeled queries (aim for 200+)

**Step 3 — Evaluate calibration**
- Plot reliability diagram (predicted prob vs actual accuracy per bin)
- Measure **ECE (Expected Calibration Error)** — target: ECE < 0.10
- If ECE ≥ 0.10: re-examine confidence signal composition

**Step 4 — Set threshold from calibrated scores**
- Default threshold: 0.65 calibrated confidence
- Tune via F1 of abstain/answer decision on held-out set

### 5.3 Confidence Signal Composition (Pre-Calibration)

Current raw confidence is composed of:
```
raw_confidence = w1 * reranker_score_top1
              + w2 * context_overlap_score
              + w3 * generation_logprob (if available)
```
Weights `w1, w2, w3` must be validated against actual correctness — not assumed.

### 5.4 What Calibrated Confidence Unlocks
- Trustworthy abstain/answer decision boundary
- Meaningful confidence in query-level logs
- Foundation for Phase 4 adaptive responses

---

## 6. Latency Budget

### 6.1 Why the Original 2s Target Was Wrong

Original spec: retrieval (500ms) + generation (1500ms) = 2000ms. This left **0ms** for reranking, query classification, context assembly, logging, and retry. The original budget was mathematically impossible with all components active.

### 6.2 Revised Budget (Realistic)

| Component | P50 | P95 | Notes |
|---|---|---|---|
| Query classification | 20ms | 50ms | Heuristic-based |
| Dense retrieval (Qdrant) | 80ms | 200ms | Hybrid search |
| BM25 retrieval | 40ms | 100ms | |
| RRF fusion | 10ms | 20ms | CPU |
| Cross-encoder reranking | 300ms | 700ms | Historical planning budget; current reranker cap is 20 and K7.11 exceeded this P95 target on CPU. |
| Chunk filtering + assembly | 20ms | 60ms | |
| LLM generation (Groq) | 600ms | 1,800ms | Planning target; current configured model is `openai/gpt-oss-20b`; remeasure with current provider/model. |
| Logging | 10ms | 30ms | async |
| **Total (no retry)** | **~1,080ms** | **~2,960ms** | |
| **Total (with 1 retry)** | **~2,160ms** | **~4,960ms** | |

These component values are budget estimates from the earlier architecture, not measurements of the current runtime. K7.11 measured only retrieval stages on CPU (not API-to-answer latency or Groq generation): median retrieval totals were 53.31ms (RRF-only), 574.72ms (reranker@5), and 3,095.61ms (reranker@20). Its P95 retrieval totals were 113.33ms, 1,095.07ms, and 3,549.29ms, respectively. Use those results only for the measured 13-query workload and machine.

**P95 target: ≤ 3.5s (no retry path)**  
**P95 target with retry: ≤ 5.5s (acceptable — retry is exceptional path)**

### 6.3 Degradation Strategy (Latency Spike)

If retrieval + reranking > 800ms:
1. Skip reranker → use RRF scores directly
2. Reduce candidate_k to 30
3. Log `latency_degradation` event

This must be implemented as a circuit breaker, not a manual toggle.

---

## 7. Roadmap — Phase-Gated Execution

**Rule: Do not start the next phase until exit criteria of current phase are all green.**

---

### Phase 1 — Pipeline Stabilization
**Status: historical snapshot passed (2026-10-15); v3 current-runtime acceptance is INCOMPLETE (2026-10-26)**

**Goal:** Every query produces either a valid answer or a clean abstain. No crashes, no silent failures, no infinite loops.

**Exit Criteria (all must pass):**
- [x] The 20-case candidate `data/evaluation/phase1_smoke_set_v3.json` completes via full generation with no unhandled exception. Generation must not be disabled.
- [x] The set contains 15 answerable cases and 5 `should_abstain` cases; answerable gold evidence resolves to the v3 canonical corpus and the five abstain cases have no gold answer/chunk.
- [x] Runtime output for every case has status `ANSWERED` or `INSUFFICIENT_CONTEXT`, confidence finite and within `[0.0, 1.0]`, at most 2 generation attempts, complete §4 JSON log fields, and complete retrieval/generation latency fields.
- [x] The run verifies the live Qdrant collection/path, exact vector count (642 for v3), and a non-empty dense probe. Dense/BM25 source and stage counts are recorded.
- [x] Runtime/pipeline/provider exceptions are either prevented or explicitly classified and logged; no unhandled exception or `None` failure state escapes to the API.
- [ ] Human review confirms at least 18/20 correct final decisions: answerable cases must be answered correctly and grounded in retrieved evidence; abstain cases must return a reasoned abstention. No wrong or ungrounded answer may have the effective `GENERATE` confidence decision. Review is pending; four rows in the current run are provider errors and require a clean rerun before review.
- [x] `sanity_check_answer()` passes its five manual cases: short answer, verbatim copy, high overlap, normal grounded answer, and empty answer.

**Acceptance procedure:** Run the v3 workflow in [`docs/evaluation_guide.md`](docs/evaluation_guide.md), using `QDRANT_COLLECTION=rag_research_phase1_v3` and `BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json`. The report path is `results/metrics/phase1_smoke_v3.json`; prior reports are archived. Review the fresh run with `python scripts/phase1_smoke.py --review-report results/metrics/phase1_smoke_v3.json`, then run `python scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke_v3.json`. The checker exits successfully only when runtime, dataset, Qdrant, complete review, failure reasons, 18/20, and zero-confident-wrong gates all pass. A run with `SMOKE_STOP_BEFORE_GEN` is not acceptance evidence.

**Current runtime run (2026-10-26):** `results/metrics/phase1_smoke_v3.json` completed all 20 runtime rows against `rag_research_phase1_v3` (642 vectors), with `runtime_pass: true`. Four records are classified as provider errors after Groq quota exhaustion; `human_review_complete: false`, `quality_pass: false`, and `phase1_pass: false`. The next valid acceptance step is to rerun after quota reset, then review the resulting report. A gold-blind sample pack is available at `results/metrics/phase1_blind_review_v3.json` for independent spot-checking; it does not replace reviewing all answerable cases.

---

### Phase 2 — Evaluation Infrastructure
**Status: `IN PROGRESS` — evaluator/data-contract work; acceptance gates remain open**

**Goal:** Automated evaluation loop that gives a reliable signal on system quality.

**Exit Criteria (all must pass):**
- [ ] Versioned evaluation dataset contains ≥ 200 labeled samples across all 4 query categories (lexical, paraphrase, multihop, adversarial), with no query category below the 20% floor in §10.3
- [ ] Every row has a unique `query_id`, `question`, boolean `should_abstain`, category, provenance, and `verification_status` (`human_verified` or `span_verified`); answerable rows have a gold answer and exactly one supported gold-chunk field (`gold_chunk_id` or `gold_chunk_ids`), while abstain rows have no gold answer/chunk
- [ ] `scripts/run_eval_phase2.py` reads the versioned dataset schema and runs the chosen RAG execution path once per query; RAGAS remains a separate optional evaluator in `scripts/run_eval.py`
- [ ] Phase 2 report outputs EM, token F1, Recall@10, MRR@10, abstain accuracy, false abstain rate, false answer rate, pipeline error rate, and failure distribution, with denominators and per-category summaries
- [ ] No query category falls below 20% of the dataset; question category and difficulty labels are tracked as distinct fields
- [ ] EM normalization function unit-tested with ≥ 10 edge cases
- [ ] Eval results are versioned and stored (JSON + timestamp), including dataset and corpus fingerprints, code revision/dirty state, runtime configuration, and per-query records

---

### Phase 3 — Confidence Calibration
**Status: `TODO` — blocked by Phase 2`**

**Goal:** Confidence score is statistically meaningful, not heuristic.

**Exit Criteria (all must pass):**
- [ ] Collected ≥ 200 (raw_confidence, actual_correct) pairs from Phase 2 eval
- [ ] Isotonic regression calibration fitted and serialized to `models/confidence_calibrator.pkl`
- [ ] Reliability diagram plotted and saved
- [ ] ECE < 0.10 on held-out 20% of calibration data
- [ ] Abstain threshold set to calibrated value (default: 0.65, adjusted from reliability diagram)
- [ ] False abstain rate on eval set ≤ 10%
- [ ] False confident wrong answer rate on eval set = 0%

---

### Phase 4 — Adaptive Failure Response
**Status: `TODO` — blocked by Phase 3`**

**Goal:** System adjusts retrieval strategy based on observed failure patterns.

**Implementation Note:** Adaptation is **per-session config override**, not a globally mutable state. Persistent global adaptation requires a state store — do not implement without one.

| Failure Type | Adaptation | Scope |
|---|---|---|
| `retrieval_miss` rate > 15% on a query category | Increase BM25 weight for that category | Session config |
| `ranking_error` rate > 20% | Reduce top_k, increase reranker strictness | Session config |
| `filtering_error` rate > 25% | Raise min_score threshold | Session config |
| `classification_error` detected | Flag for classifier retraining | Log only (offline fix) |

**Exit Criteria:**
- [ ] Failure rate distribution tracked per query category in eval output
- [ ] At least one failure type shows measurable improvement after adaptation (≥5% reduction in failure rate)
- [ ] Adaptation changes logged with before/after metrics

---

### Phase 5 — Optimization & Scalability
**Status: `TODO` — blocked by Phase 4`**

**Goal:** System meets latency and cost targets under load.

**Exit Criteria:**
- [ ] P95 latency ≤ 3.5s verified under 10 concurrent requests
- [ ] Latency degradation circuit breaker implemented and tested
- [ ] Token usage per query ≤ 2,000 input tokens on average (measured over eval set)
- [ ] Reranker caching implemented for repeated queries (LRU cache, size=512)
- [ ] Load test: 50 queries/minute for 5 minutes with ≤ 5% failure rate

---

## 8. Production Readiness Checklist

System is **production-ready** when ALL of the following are true:

### Answer Quality
- [ ] EM ≥ 70% on held-out eval set (≥ 200 samples)
- [ ] Faithfulness ≥ 0.88 on held-out eval set
- [ ] Zero confirmed hallucinations on eval set (NLI-verified)

### Reliability
- [ ] Abstain accuracy ≥ 90% on labeled eval set
- [ ] False abstain rate ≤ 10%
- [ ] Confident wrong answer rate = 0%
- [ ] Every query produces structured log — no silent failures

### Calibration
- [ ] Confidence calibration fitted with ECE < 0.10
- [ ] Abstain threshold set from empirical calibration data

### Performance
- [ ] P95 end-to-end latency ≤ 3.5s under normal load
- [ ] Retry loop hard-capped at 2 iterations — verified in code
- [ ] Latency degradation circuit breaker active

### Observability
- [ ] Per-query JSON log with all fields in Section 4 schema
- [ ] Failure type distribution queryable from logs
- [ ] Eval script runs automatically (CI or cron)

### Retrieval
- [ ] Recall@10 ≥ 90% on eval set
- [ ] MRR@10 ≥ 0.70 on eval set
- [ ] No duplicate-dominated retrieval results (intra-list diversity ≥ 0.30)

---

## 9. Non-Goals

| Not Building | Reason |
|---|---|
| Chatbot / multi-turn conversation | Out of scope — this is single-turn QA |
| HyDE (Hypothetical Document Embeddings) | Confirmed domain drift — worse retrieval. Do not re-enable. |
| Multi-query expansion (active) | +0.17 MRR at +3400ms cost — not viable until latency budget allows |
| Generic LLM wrapper | This system must be reliable, not flexible |
| Real-time index updates | Out of scope for v1 |
| Global mutable adaptive state | Requires state store not yet built — Phase 4 uses session-level overrides only |

---

## 10. Evaluation Integrity Constraints

### 10.1 Dataset Split (Mandatory)

Dataset MUST be split into:

- train_eval (used for tuning)
- holdout_eval (never used during development)

Constraint:
- Final reported metrics MUST use holdout_eval only

---

### 10.2 Data Leakage Prevention

For each (query, chunk):

- compute similarity(query, chunk_text)

Constraint:
- If similarity > 0.9 → sample MUST be excluded

Rationale:
Prevents trivial retrieval where query ≈ chunk text.

---

### 10.3 Query Difficulty Distribution

Dataset MUST contain:

- easy (direct lookup)
- medium (paraphrase)
- hard (multi-hop / reasoning)
- adversarial

Constraint:
- No category < 20% of dataset

---

### 10.4 Metric Stability Check

Run evaluation multiple times (≥3 runs)

Constraint:
- Metric variance ≤ 3%

If violated:
→ system is unstable (likely due to LLM randomness)

## 11. Grounding Strictness Specification

### 11.1 Span-Level Grounding (Required)

For each answer:

- Each claim MUST map to at least one span in retrieved chunks

Validation:
- substring match OR semantic similarity ≥ 0.75

If not satisfied:
→ answer MUST be rejected

---

### 11.2 Attribution Coverage

Define:

coverage = (# supported sentences) / (total sentences in answer)

Constraint:
- coverage ≥ 0.8

If coverage < 0.8:
→ answer downgraded or abstained

---

### 11.3 Multi-Chunk Consistency

If answer uses multiple chunks:

- No conflicting claims allowed

Detection:
- NLI contradiction between supporting chunks

If conflict detected:
→ abstain

| `semantic_drift` | Retrieved chunks relevant but do not answer the query intent | High similarity but low answer relevancy | Adjust query expansion / reranker |
| `context_conflict` | Retrieved chunks contradict each other | NLI contradiction across chunks | Abstain |
| `position_bias` | Relevant chunk exists but ignored due to low rank | Gold chunk rank > 5 but present | Increase reranker weight |
| `index_drift` | Index embedding outdated vs corpus | Retrieval quality drops over time | Re-embed corpus |
| `prompt_sensitivity` | Output changes significantly with minor prompt variation | High answer variance | Stabilize prompt template |

## 12. Confidence System Constraints

### 12.1 Feature Validity

Each feature MUST satisfy:

- monotonic relation with correctness OR
- explicitly modeled as non-linear

If not:
→ feature MUST be removed

---

### 12.2 Feature Interaction Check

System MUST detect conflicting signals:

Examples:
- high reranker_score + low overlap
- low reranker_score + high overlap

If conflict detected:
→ confidence MUST be penalized

---

### 12.3 Entropy-Based Uncertainty

Define:

entropy = entropy of retrieval score distribution

Constraint:
- high entropy → low confidence

---

### 12.4 Query-Type Calibration

Confidence MUST be calibrated separately for:

- factual
- reasoning
- general

## 13. Load & Throughput Behavior

### 13.1 Backpressure Strategy

When system load increases:

- candidate_k reduced (50 → 30 → 20)
- reranker disabled if latency > threshold
- context budget reduced (1800 → 1200 tokens)

---

### 13.2 Priority Policy

If system overloaded:

- factual queries prioritized over reasoning
- retry disabled under high load

---

### 13.3 Graceful Degradation

System MUST degrade in this order:

1. Disable reranker
2. Reduce candidate_k
3. Reduce context size
4. Force abstain

Never:
- return low-confidence answer

## 14. Change Safety Protocol

Any system modification MUST follow:

1. Run full eval BEFORE change
2. Apply change
3. Run full eval AFTER change
4. Compare metrics

Constraint:
- If any metric degrades > 3%
  → change MUST be reverted

---

### 14.1 Canary Evaluation

Before full rollout:

- test on 10% subset

If failure rate increases:
→ rollback immediately

## 15. Dataset & Index Versioning

Each experiment MUST log:

- dataset_version
- embedding_model_version
- chunking_strategy_version

Constraint:
- Results without version metadata are INVALID

## 16. Adversarial Robustness

System MUST handle:

- prompt injection in retrieved chunks
- irrelevant high-similarity chunks
- malicious document content

---

### 16.1 Defense Mechanisms

- strip instructions from chunks
- ignore system-level instructions in context
- enforce answer-only-from-facts policy


System must handle:

- paraphrase queries (semantic drift)
- multi-hop reasoning queries
- distractor-heavy context
- out-of-distribution queries

Metrics:
- EM per category
- Failure distribution per category

Constraint:
No category may have EM < 40%

If any of the following occurs:

- hallucination detected
- confident wrong answer
- repeated retrieval_miss (>3 times in session)

System must:

1. Force abstain
2. Return structured error response
3. Log CRITICAL event

Any system modification must follow:

1. Run full eval BEFORE change
2. Apply change
3. Run full eval AFTER change
4. Compare:
   - EM
   - Recall@10
   - MRR
   - Hallucination rate

If any metric degrades > 3%:
→ change MUST be reverted

Known limitations:

- Cannot answer if answer not present in indexed corpus
- Sensitive to chunking quality
- Cross-encoder latency bottleneck
- Confidence depends on dataset quality

System does NOT:
- perform true reasoning beyond context
- guarantee correctness without supporting chunk



## 17. Proposed Product Differentiators — "Forensic AI"

> **Status: PROPOSED — not implemented, not committed, and not a phase-gate
> change.** Added 2026-10-26. This section records a product-direction
> proposal. None of the capabilities below exist in the current runtime unless
> a subsection explicitly says otherwise. Do not use them in README claims,
> demos, sales material, or evaluation reports until the acceptance evidence in
> §17.3–§17.7 exists. The Phase 1–5 gates in §7 and the production checklist in §8 remain
> authoritative, and Phase 1 is still open.

### 17.1 Positioning

Generic "chat with PDF" systems optimize fluent answers. This proposal moves
the external value proposition from **generative AI** toward **forensic AI**:

> Do not optimize for how fluently the system speaks. Optimize for how
> defensibly it can account for every sentence.

The differentiators are evidence provenance, explicit conflict reporting,
calibrated uncertainty, and resistance to distractor documents. They are
product capabilities built on top of the existing reliability work, not a
replacement for it. A citation feature is worthless if retrieval is wrong; a
confidence percentage is dangerous if it is uncalibrated.

### 17.2 Feature portfolio

| ID | Differentiator | Intended user value | Current state | Gate |
|---|---|---|---|---|
| F1 | Citation-to-sentence attribution | Every answer sentence links to a specific source sentence; clicking highlights it in the PDF | **Not implemented.** §11.1/§11.2 define the validation rule, but provenance metadata and a citation API do not exist | Requires re-ingestion/provenance work; after Phase 2 |
| F2 | Evidence conflict detection | Surfaces contradictory evidence instead of blending it into one confident answer | **Not implemented.** §11.3 specifies NLI contradiction handling, but no NLI model or contradiction eval set exists | Requires F1-style citation substrate plus a labeled contradiction set; Phase 3/4 |
| F3 | Calibrated confidence certificate | A statistically calibrated answer-level confidence with documented methodology | **Not implemented.** Confidence is heuristic and uncalibrated (§5.1); Phase 3 is the existing plan | Blocked by Phase 2's ≥200 labeled samples; Phase 3 |
| F4 | Adversarial noise filtering | Ignores bibliography-only, title-only, keyword-stuffed, and near-duplicate distractor chunks | **Partial foundation only.** Layered heuristic gates exist in §3.3; adversarial robustness is not measured | Phase 2 adversarial slice; rule changes after baseline |

### 17.3 F1 — Citation-to-Sentence Attribution (Deep Linking)

**Product behaviour.** Each sentence in an answer carries one or more citations.
Selecting a citation opens the source PDF at the cited page and highlights the
supporting sentence. The user does not have to search the document for the
claim.

**Technical approach.** Extend §11.1 span-level grounding into a user-visible
contract:

1. Ingest and persist provenance for every chunk: `doc_id`, PDF page number,
   `char_start`/`char_end`, optional `bbox`, and a stable sentence identifier.
2. Map each generated claim to at least one retrieved span using substring
   match or semantic similarity ≥ 0.75 (§11.1).
3. Return a structured `citations` array keyed by answer sentence. Citations
   must be constructed from retrieval metadata, never parsed from model output.
4. Resolve the citation to a page/sentence highlight in the source viewer. If
   `bbox` is unavailable, fall back to page plus sentence text rather than
   claiming an exact visual highlight.

**Current gap (source-verified 2026-10-26).** The 642-chunk v3 corpus stores
`doc_id`, `chunk_id`, and chunk-relative sentence fields, but all 642 rows record
`start_sentence_idx = end_sentence_idx = 0` and no page, character-offset, or
bbox field. `document_loader.py` joins cleaned page text with `"\n\n"` and
discards page boundaries. `check_answer_evidence` in `generator.py` is lexical
overlap and explicitly not semantic entailment. No API response field carries
sentence-level citations.

**Acceptance evidence.** On a labeled set of answer sentences: citation
precision (fraction of citations that actually support the claim), support rate,
and deep-link resolution accuracy on a sample of PDFs. A fabricated citation is
a critical failure. The test set must include multi-column papers, tables, and
pages with headers/footers.

### 17.4 F2 — Evidence Conflict Detection (The "Truth" Auditor)

**Product behaviour.** When retrieved documents disagree, the system reports
the disagreement with citations instead of averaging the claims into a single
answer. Example: "Paper A reports the method is efficient; Paper B reports a
failure risk at high temperature." Where the conflict bears on the answer, the
system abstains under §11.3; where it is peripheral, it reports a conflict
warning.

**Technical approach.** Run a Natural Language Inference (NLI) model over
retrieved chunk pairs and classify `ENTAILMENT`, `NEUTRAL`, or `CONTRADICTION`.
A deterministic threshold decides when a contradiction is actionable. The
detector must be an NLI model or a validated judge with measured precision and
recall; an LLM's self-reported "no conflict" is not sufficient evidence.
Pairwise comparison is O(k²), so cap the number of pairs (for example, the top
five chunks) and measure the cost against §6.

**Current gap.** No NLI implementation exists in `src/`. §11.3 is a
specification only. `check_answer_evidence` is a lexical guard, not
contradiction detection. There is no contradiction-labeled evaluation set, and
the 13-query retrieval set cannot support a precision/recall claim.

**Policy question to settle before implementation.** Not every contradiction
invalidates an answer. Define which conflicts force abstention, which produce a
warning, and which are out of scope (for example, different experimental
conditions). This policy must be tested, not inferred from a single example.

**Acceptance evidence.** Precision, recall, and F1 on a labeled contradiction
set; false-conflict rate on a neutral set; false-abstention impact; latency
within §6 or an explicit degradation path; human adjudication of a sample.

### 17.5 F3 — Calibrated Confidence Certificate (Not "Deterministic")

**Terminology correction.** "Deterministic confidence" is a misnomer. Isotonic
regression produces a *calibrated probability*, not a deterministic guarantee.
A score of 0.95 means that among similarly scored answers in the calibration
distribution, about 95% were correct. It does **not** mean that this particular
answer has a 1-in-20 chance of being wrong, and it is not a per-answer
frequency claim. Product copy must use the former wording only.

**Relationship to the existing roadmap.** This is not a new feature. Phase 3
(§7) already requires ≥200 `(raw_confidence, actual_correct)` pairs, isotonic
regression, a reliability diagram, ECE < 0.10 on held-out data, a calibrated
abstain threshold, and separate calibration by query type (§12.4). The current
engine is heuristic and uncalibrated (§5.1), which is already the highest
priority blocker in Appendix B.

**Certificate contents.** If Phase 3 passes, an answer may expose: calibration
sample size, calibration date, dataset/corpus fingerprints (§15), ECE,
per-query-type coverage, model and prompt versions, and the calibrated score.
Until then, label confidence as **heuristic and uncalibrated**. Do not display a
percentage, and do not claim "95% accuracy", "zero hallucination", or a
per-answer error rate.

**Acceptance evidence.** All Phase 3 exit criteria, plus reproducibility of the
certificate and a recalibration trigger when the corpus, model, or prompt
changes. The zero-confident-wrong constraint in §2.3 remains absolute.

### 17.6 F4 — Adversarial Noise Filtering

**Product behaviour.** The system rejects chunks that are similar to the query
but do not contain usable evidence: bibliography-only passages, title/author
lines, keyword-stuffed distractors, near-duplicate boilerplate, and chunks
containing instructions aimed at the model.

**Current foundation.** §3.3 already layers a post-RRF score gate, a
pre-reranker `is_low_quality_chunk` gate, a strict filter with a minimum-filter
fallback, and a generation-time score gate. These are heuristics with known
limitations, not proof of adversarial robustness. The `min_score_threshold`
correction in Appendix A is a concrete example of a gate that filters nothing
on the RRF scale.

**Technical approach.** Add an adversarial slice to the Phase 2 dataset before
tightening any rule: distractor injection, near-duplicate irrelevant chunks,
empty/bibliography-only chunks, and prompt injection in retrieved text. Measure
distractor rejection and clean-set false rejection separately. Only then extend
the filters. Strip or neutralize instructions in chunks and enforce the
answer-only-from-facts policy in §16.1.

**Acceptance evidence.** Adversarial-category EM/answer accuracy, distractor
rejection rate, false-rejection rate on clean queries, zero executed prompt
injections, and correct abstention when the only retrieved chunks are
distractors. Tightening a filter without these measurements is not an
improvement.

### 17.7 Packaging: "The Audit Dossier"

A proposed formal output mode, in Markdown first and PDF only if needed:

1. **Conclusion** — the answer, or an explicit abstention.
2. **Evidence map** — claim-by-claim mapping to source spans (F1).
3. **Conflict analysis** — contradictory findings, with both sides cited (F2).
4. **Integrity metrics** — calibrated confidence and coverage (F3).

This is a presentation/reporting layer, not a new retrieval pipeline and not a
multi-turn chatbot. It depends on F1–F3; without them it would be a longer
answer with decorative citations. It does not relax §9's non-goals. The word
"audit" is product language, not a legal, patent, or compliance certification;
the report must carry that disclaimer.

**Acceptance evidence.** Every claim in a generated dossier maps to a citation;
the conflict section appears only when a conflict was detected; metrics are
reproducible from stored records; human review confirms no unsupported claims.

### 17.8 Priority Recommendation

The recommendation is **F1 first, F2 second**, matching the research-user value
argument, with F4 as inexpensive evaluation work and F3 treated as the existing
Phase 3 blocker rather than a new feature.

| Order | Work | Why this order | Suggested placement |
|---|---|---|---|
| 1 | F1 — citation-to-sentence attribution | Highest visible user value; turns the existing §11.1/§11.2 rule into evidence navigation; creates the span substrate F2 and the dossier need | Provenance schema design during Phase 2; implementation after Phase 2 exit |
| 2 | F2 — conflict detection | Strongest "audit" differentiator, but only credible once claims are citation-mapped and a contradiction set exists | Phase 3/4, after F1 |
| 3 | F4 — adversarial filtering | Defensive and comparatively cheap; the existing gates provide a starting point, but the evaluation slice must come first | Phase 2 dataset work; rule changes after baseline |
| 4 | F3 — calibrated confidence | Necessary trust infrastructure, already the critical blocker; it is not a differentiator until ECE and sample-size gates pass | Existing Phase 3; do not re-scope |

No implementation work in this section starts before Phase 1 acceptance closes.
Designing datasets, provenance schemas, and label guidelines is planning work,
not an acceptance substitute.

### 17.9 Risks and Guardrails

- **Citation spoofing.** Citations must come from retrieval metadata, not from
  model-generated text. Retrieved documents can contain fake citation strings.
- **Overclaiming.** Lexical support is not entailment. "Zero hallucination" and
  "95% accuracy" are prohibited until the corresponding measured gates pass.
- **Re-indexing risk.** Adding page/offset metadata changes the corpus
  fingerprint. Version it and re-run evaluation under §14 and §15; do not
  overwrite the v3 corpus in place.
- **False conflicts and false abstention.** NLI false positives can increase
  abstention on answerable queries. Measure both error directions.
- **Latency.** Pairwise NLI can violate §6. Cap pairs and use the degradation
  strategy rather than silently exceeding the budget.
- **Scope displacement.** PDF highlighting and report export are ingestion,
  API, and frontend work. They must not displace Phase 1 acceptance or Phase 3
  calibration.

---

## Appendix A — Current Runtime Snapshot (Source-Verified 2026-10-11)

```json
{
  "retrieval_strategy": "hybrid_rrf",
  "api_top_k_policy_defaults": {"factual": 5, "general": 8, "reasoning": 10},
  "api_top_k_request_override": "optional, 1..20",
  "candidate_k_setting": 50,
  "typical_computed_candidate_k_at_top_k_10": 40,
  "rrf_k": 60,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "reranker_enabled_by_default": false,
  "reranker_input_cap": 20,
  "hyde_enabled": false,
  "multi_query_enabled": false,
  "llm_model": "openai/gpt-oss-20b",
  "confidence_reject_threshold": 0.35,
  "confidence_partial_threshold": 0.65,
  "api_retrieval_attempts": "one initial attempt plus at most one BM25-heavy retry",
  "context_max_chars": 6000
}
```

This snapshot describes defaults and code behavior, not a calibrated quality baseline. Runtime and evaluation changes should be measured against the same versioned dataset and recorded with their experiment evidence.

### Where each value lives

Re-verified against source on 2026-10-11. A value listed here is not
necessarily configurable, and two of them changed on that date.

| Key | Source | Env-overridable |
|---|---|---|
| `retrieval_strategy` | hardcoded path in `hybrid_retriever.py` | no |
| `api_top_k_policy_defaults` | `PolicyEngine.retrieval_strategy()` | no |
| `api_top_k_request_override` | `GenerateRequest.top_k` (`ge=1, le=20`) | n/a — request field |
| `candidate_k_setting` | `settings.candidate_k` | yes |
| `typical_computed_candidate_k_at_top_k_10` | derived by `_choose_candidate_k` | no |
| `rrf_k` | `settings.hybrid_rrf_k` | yes |
| `embedding_model` | `settings.embed_model` | yes (`RAG_EMBED_MODEL` / `EMBED_MODEL`) |
| `reranker_model` | `settings.reranker_model` | yes (`RAG_RERANK_MODEL` / `RERANKER_MODEL`) |
| `reranker_enabled_by_default` | `DEFAULT_USE_RERANKER` in `api/main.py` | no |
| `reranker_input_cap` | literal `candidates[:20]` in `hybrid_retriever.py` | no |
| `hyde_enabled` | `settings.use_hyde` | yes |
| `multi_query_enabled` | `settings.use_multi_query` | yes |
| `llm_model` | `settings.groq_model` | yes |
| `confidence_reject_threshold` | `settings.confidence_reject_threshold` | **yes** (was `no` until 2026-10-11) |
| `confidence_partial_threshold` | `settings.confidence_partial_threshold` | **yes** (was `no` until 2026-10-11) |
| `api_retrieval_attempts` | `MAX_RETRIEVAL_RETRIES + 1` in `api/main.py`; shared retry limit in `controller/agent_state.py` | no |
| `context_max_chars` | `settings.generation_max_chars` | yes |

### Changed 2026-10-11 — confidence thresholds are now configurable

Both cuts were literals in `ConfidenceEngine.__init__`. Phase 3 calibration
required editing source, which is why this table previously marked them **no**.
They now come from `Settings` with the same values, so the numbers above are
unchanged and only the configurability changed.

`tests/test_confidence_thresholds.py` re-implements the original inline branch
verbatim and compares it against the refactored version across six threshold
pairs and six query types: **zero behavioural difference**. The constructor
still accepts explicit arguments, which take precedence.

Two properties are preserved deliberately rather than fixed, because changing
them would alter behaviour:

- `reject` is floored at `0.0` for `reasoning` queries. If `confidence_reject_threshold`
  is set below `0.05`, reasoning queries cannot be rejected at all.
- `partial` is floored at `reject`.

### Changed 2026-10-11 — the retriever holds no per-request state

`_run_search` used to assign `self.use_multi_query` and `self.use_hyde` before
calling `_expand_and_hyde_parallel`, which read them back. `api/main.py` builds
one retriever at startup and serves every request through it with
`run_in_threadpool`, so concurrent requests could overwrite each other's
flags.

The flags are now local variables passed as arguments, and the constructor
parameters that were always overridden have been removed. `search()` still
accepts them. Impact was nil while both flags are False, which §9 requires;
§13.3 graceful degradation could not have been implemented with the old shape.

Also consolidated on 2026-10-11: the PARTIAL_TRUST presentation rule had four
implementations that had drifted, and `top_k` had three schemes plus two tests
that asserted against their own local copy of a retired heuristic. Both now
have a single source — `apply_partial_trust_policy` and `PolicyEngine`.

### Correction — `min_score_threshold` filters nothing

§3.4 names three retry actions, the third being "filtering relaxed
(`min_score`: `0.0` → `-0.1`)". **Measured, that action does nothing.**

`hybrid_retriever.py` applies `min_score_threshold` as a floor on
`retrieval_score`, which is the RRF score: `w * boost / (k + rank + 1)` with
`k = rrf_k = 60` and `w = 0.5 / 0.5`. That bounds it from above at

```
max = 0.5 * 1.4 / 61 + 0.5 * 1.4 / 61 = 0.02295   (best-case section boost)
```

Measured over 120 RRF scores from 3 canonical queries: min `0.005051`,
max `0.021054`. A floor of `0.0` removes **0** chunks. A floor of `-0.1`
removes **0** chunks.

The two scales are conflated in `generator.filter_chunks_by_score`, which
switches on `has_rerank`: with the cross-encoder it reads `rerank_score`
(roughly -11..+11) and uses `-3.0`, which does work. Without it, `_get_score`
falls through to `retrieval_score` and uses `settings.min_score_threshold`,
which is 0.0 on the RRF scale. One threshold variable compared against two
incompatible distributions.

Consequence for §3.4: the score-floor change is inert in both the Agent and API.
The retry contract records it for traceability, but only widening requested
`top_k` and shifting BM25 weight currently change retrieval behavior. Do not
describe the score-floor adjustment as improved recall or relaxed filtering;
see `results/metrics/k5_behaviour_changes.md`.

### Correction — §3.5 context control

§3.5 lists `reference_ratio` (default 0.3) and `context_length_budget` (default
1800 tokens). Neither exists in `src/api/config.py`.

- `max_ref_ratio` is produced per query by `PolicyEngine`, not read from config.
- The context budget is `settings.generation_max_chars = 3000` — a **character**
  limit, not a token limit. `generator.py` enforces it in `build_context_with_ids`.

Treat §3.5's figures as superseded by the table above.

### Correction — retry bound

§3.4 specifies a maximum of 2 iterations. That was not enforced on the
`RETRIEVE_AGAIN` path: the branch consulted only the step budget, so an unstable
retrieval signal produced 3 attempts at the default `max_steps=8` and scaled
linearly with `max_steps` when stagnation detection could not fire. Corrected in
`src/controller/agent.py`; see `tests/test_retry_contract.py`.

---

## Appendix B — Known Issues (Active Blockers)

Updated 2026-10-11. The previous version listed only open-ended quality gaps
and no mechanical blockers, which is why a RAGAS dependency break and a CI job
that could never pass both went unrecorded here for as long as they lasted.

### Resolved — closed 2026-10-10/11

| Issue | Was | Resolution |
|---|---|---|
| RAGAS unimportable | `ragas/llms/base.py` imports `langchain_community.chat_models.vertexai`, removed in langchain-community 0.4.2 | Pinned `langchain-community==0.4.1`. An earlier revision listed the package as "unused" on a 0-import scan; that reasoning does not hold for a transitive dependency. Guard: `tests/test_ragas_importable.py` |
| `datasets.Dataset` unusable on 3.14 | datasets 3.6.0 pins `dill<0.3.9`; dill 0.3.8 breaks `Pickler._batch_setitems` | `ragas_evaluator` builds an `EvaluationDataset`, the native v2 type. ragas only runs `convert_v1_to_v2_dataset` on an HF Dataset, so this is also the forward path |
| Every RAGAS metric returned NaN | `ragas/executor.py:15` calls `nest_asyncio.apply()` at module level; on 3.14 `asyncio.wait_for` then raises "Timeout should be used inside a task". nest_asyncio 1.6.0 is the last release and has no `unpatch()` | Neutralized in `src/_compat.py`. Reproduced without an API call |
| RAGAS reported 0.0000 on total failure | NaN filtered to 0.0; a run with 55/55 NaN was stored as `0.0` with `hallucination_rate 0.0%` | `nan_counts` per metric plus `RuntimeError` when a metric is entirely NaN. Historical artifact kept and explained in `results/metrics/k2_ragas_legacy_lineage.md` |
| CI `rag-eval` could never pass | pointed `eval_ci.py` at `data/adversarial/adversarial_queries.json`, which has no `answer`, `contexts` or `ground_truth` | Replaced by `ci.yml` (tests + contracts) and `quality-gate.yml` (manual) |
| Corpus could be overwritten silently | `chunker.py:469` wrote 582 legacy-format chunks over the 796-chunk live corpus at the same path | Redirected to `chunks_legacy_chunker.json`; `check_ci_contracts.py` verifies the corpus sha256 recorded in the dataset envelope (§15) |
| Label errors scored as retrieval failures | `FAILURE_EVAL_DATA` defined and never assigned; a gold id absent from the corpus counted as `retrieval_miss` and stayed in the denominator | `evaluate_query` takes `corpus_chunk_ids`; those rows are excluded and reported. One bad label could manufacture 33 recall points on a 3-query set |

### Open

| Issue | Severity | Blocks | Owner |
|---|---|---|---|
| `reference_policy` cannot be honoured | High | Citation and authorship questions | `allow_references=True` never takes effect: the section boost removes every reference at RRF fusion. Needs `rrf_fuse` to accept a per-query boost — a retrieval behaviour change, not a cleanup |
| Confidence engine not calibrated | **CRITICAL** | Phase 3, Abstain reliability | Thresholds are configurable now (`settings.confidence_reject_threshold` / `confidence_partial_threshold`) but still uncalibrated. `reasoning` queries cannot be rejected at all if the reject threshold is set below 0.05, because of the `max(0.0, ...)` floor |
| Query expansion produces noisy multi-query outputs | High | Disabled — do not re-enable | Post Phase 3 |
| Classifier accuracy not measured | Medium | Phase 4 adaptation | Phase 2 eval |
| No intra-list diversity metric implemented | Medium | Detecting duplicate retrieval | Phase 2 |
| No load test conducted | High | Production readiness | Phase 5 |
| Shared retriever mutates per-request state | Medium | §13 graceful degradation | `hybrid_retriever.py:457` writes `self.use_multi_query`. Harmless while both flags are False, but §13.3 needs per-request control |
| `min_score_threshold` filters nothing | Medium | §3.4 retry contract | Compared against RRF scores, measured in [0.005, 0.021]. A floor of 0.0 and the `-0.1` retry relaxation each remove zero chunks, so the §3.4 "relax filtering" action is a no-op |
| `scripts/` is 1.6× `src/` | Low | — | `scripts/MANIFEST.json` classifies all 37; pruning not done |
| 125 evidence artifacts, no index | Low | — | `results/metrics/` |
| Git identity is a placeholder | Low, but it blocks archaeology | — | 21 commits recorded as `Your Name <you@example.com>` |

---

*END OF DOCUMENT — Version 2.2*
