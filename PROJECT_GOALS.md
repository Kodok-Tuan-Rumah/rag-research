# RAG SYSTEM — ENGINEERING GOALS & PRODUCTION SPEC
> **Version:** 2.1
> **Repo:** `rag-research`  
> **Status:** Phase 1 acceptance passed (2026-10-15); Phase 2 evaluation-infrastructure work is ready to begin
> **Last Updated:** 2026-10-15

> **Current evidence note:** The K7.10–K7.13 retrieval audits use the 13-query `data/evaluation/retrieval_eval_dataset_v2.json`. This is distinct from the older 14-query `data/evaluation/retrieval_eval_dataset.json` summarized in the historical README baseline. The 13-query K7 set does not satisfy the Phase 2 target of at least 200 labeled samples, and its results do not establish production readiness or confidence calibration.

> **Phase 1 evidence (2026-10-15):** `results/metrics/phase1_smoke.json` records a full-generation run with 20 cases, complete human review, `runtime_pass: true`, `quality_pass: true`, and `phase1_pass: true`; the checker was rerun and returned the same result. This is acceptance evidence for the frozen set, not a production-readiness or confidence-calibration claim. A follow-up spot check was suggested for grounding judgments on `p1_answer_03` and `p1_answer_15_attention`; revise the annotations and re-run the checker if that review finds either judgment incorrect. See [`docs/evaluation_guide.md`](docs/evaluation_guide.md).

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
Attempt 1: Standard retrieval → generate
  │
  └── IF (confidence < threshold OR grounding_score < 0.5):
        Attempt 2 (ONE retry only):
          - candidate_k *= 1.5 (round up)
          - BM25 weight += 0.2
          - Filtering relaxed (min_score: 0.0 → -0.1)
          │
          └── IF still low confidence → ABSTAIN (do not retry again)
```

**Hard constraints on retry:**
- Max iterations: **2** (initial + 1 retry)
- Max additional LLM calls from retry: **1**
- Max latency addition from retry: **2.0s**
- If retry produces same confidence as attempt 1: **ABSTAIN immediately**

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
**Status: `PASSED` — acceptance report and checker passed 2026-10-15**

**Goal:** Every query produces either a valid answer or a clean abstain. No crashes, no silent failures, no infinite loops.

**Exit Criteria (all must pass):**
- [x] The fixed 20-case `data/evaluation/phase1_smoke_set_v1.json` completes via full generation with no unhandled exception. Generation must not be disabled.
- [x] The set contains 15 answerable cases and 5 `should_abstain` cases; all answerable gold chunks resolve to the frozen canonical corpus and the five abstain cases have no gold answer/chunk.
- [x] Runtime output for every case has status `ANSWERED` or `INSUFFICIENT_CONTEXT`, confidence finite and within `[0.0, 1.0]`, at most 2 generation attempts, complete §4 JSON log fields, and complete retrieval/generation latency fields.
- [x] The run verifies the live Qdrant collection/path, exact vector count (796 for the frozen corpus), and a non-empty dense probe. Dense/BM25 source and stage counts are recorded.
- [x] Runtime/pipeline/provider exceptions are either prevented or explicitly classified and logged; no unhandled exception or `None` failure state escapes to the API.
- [x] Human review confirms at least 18/20 correct final decisions: answerable cases must be answered correctly and grounded in retrieved evidence; abstain cases must return a reasoned abstention. No wrong or ungrounded answer may have the effective `GENERATE` confidence decision. The current report records 20/20 and zero confident-wrong answers.
- [x] `sanity_check_answer()` passes its five manual cases: short answer, verbatim copy, high overlap, normal grounded answer, and empty answer.

**Acceptance procedure:** Run `python scripts/phase1_smoke.py`. Before a new run replaces an existing report, the runner archives the prior report under `results/metrics/archive/`. Review answerable rows with `python scripts/phase1_smoke.py --review-report results/metrics/phase1_smoke.json`; this interactive review can resume partially annotated reports. Mark `answer_correct=yes` only when the answer addresses the question and includes the gold answer's essential facts (paraphrase is acceptable). Mark `grounded=yes` only when every material claim is supported by retrieved chunks; retrieving the gold chunk alone does not establish groundedness. For each answer marked incorrect or ungrounded, select at least one `failure_categories` value (`answer_incorrect`, `unsupported_claim`, `retrieval_miss`, `overly_specific`, `incomplete_answer`, or `other`) and add notes where useful. Then run `python scripts/phase1_smoke.py --check-report results/metrics/phase1_smoke.json`. The checker exits successfully only when the runtime, dataset, Qdrant, complete review, failure reasons, 18/20, and zero-confident-wrong gates all pass. Any already-known confident-wrong answer is reported even while other reviews are pending and must be fixed in a fresh acceptance run. A run with `SMOKE_STOP_BEFORE_GEN` is not Phase 1 acceptance evidence.

**Follow-up audit note:** The accepted human review marks all 15 answerable cases correct and grounded. A later advisory review raised questions about whether all claims in `p1_answer_03` and `p1_answer_15_attention` are supported strictly by chunks actually included in the LLM prompt. Re-open those cases with `python scripts/phase1_smoke.py --review-report results/metrics/phase1_smoke.json` if a stricter grounding audit is desired; update annotations only after reviewer verification, then rerun `--check-report`. This follow-up does not change the recorded checker result unless the review is changed.

---

### Phase 2 — Evaluation Infrastructure
**Status: `TODO`**

**Goal:** Automated evaluation loop that gives a reliable signal on system quality.

**Exit Criteria (all must pass):**
- [ ] Versioned retrieval evaluation dataset(s) contain ≥ 200 labeled samples across all 4 query categories (lexical, paraphrase, multihop, adversarial)
- [ ] Each sample has: `query`, `gold_answer`, `gold_chunk_id`, `should_abstain` label
- [ ] `run_eval.py` runs end-to-end without manual intervention
- [ ] `run_eval.py` outputs: EM, Recall@10, MRR@10, abstain accuracy, false abstain rate, failure type distribution
- [ ] EM normalization function unit-tested with ≥ 10 edge cases
- [ ] Eval results versioned and stored (JSON + timestamp)

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
  "context_max_chars": 3000
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

Consequence for §3.4: of the three retry actions, two are effective (widening
`k`, shifting BM25 weight) and one is inert **in both the Agent and the API**.
§3.4 should be corrected before the adapters are aligned to it; see
`results/metrics/k5_behaviour_changes.md`.

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

*END OF DOCUMENT — Version 2.0*
