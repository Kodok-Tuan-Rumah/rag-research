# RAG SYSTEM — ENGINEERING GOALS & PRODUCTION SPEC
> **Version:** 2.1
> **Repo:** `rag-research`  
> **Status:** Phase 1 Active — runtime decision-path alignment is implemented; Phase 1 exit criteria are not all verified
> **Last Updated:** 2026-10-08

> **Current evidence note:** The K7.10–K7.13 retrieval audits use the 13-query `data/evaluation/retrieval_eval_dataset_v2.json`. This is distinct from the older 14-query `data/evaluation/retrieval_eval_dataset.json` summarized in the historical README baseline. The 13-query K7 set does not satisfy the Phase 2 target of at least 200 labeled samples, and its results do not establish production readiness or confidence calibration.

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
| Exact Match (EM) | ≥ 70% | Token-level F1 after normalization (lowercase, strip punct, remove articles). NOT raw string match. | `TODO` |
| Semantic F1 | ≥ 0.75 | BERTScore F1 against ground truth, averaged over eval set | `TODO` |
| Faithfulness | ≥ 0.88 | RAGAS faithfulness — NLI entailment between answer and retrieved chunks | `DONE` |
| Hallucination Rate | 0% | Definition: answer contains claim not entailed by any retrieved chunk. Measured via NLI (≥0.5 entailment threshold). Evaluated on full eval set per release. | `WIP` |
| Answer Relevancy | ≥ 0.88 | RAGAS answer relevancy metric | `DONE` |

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
**Status: `ACTIVE`**

**Goal:** Every query produces either a valid answer or a clean abstain. No crashes, no silent failures, no infinite loops.

**Exit Criteria (all must pass):**
- [ ] `run_single_query.py --mode full` completes without exception on 20 random queries
- [ ] Retry loop hard-limited to max 2 iterations — verified in code
- [ ] Confidence score always in range [0.0, 1.0] — assertion added
- [ ] Every query produces structured JSON log with all required fields
- [ ] No `None` or unhandled exception propagates to API layer
- [ ] `sanity_check_answer()` correctly rejects reference-dump answers on 5 manual test cases

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



## Appendix A — Current Runtime Snapshot (Source-Verified 2026-10-08)

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

---

## Appendix B — Known Issues (Active Blockers)

| Issue | Severity | Blocks | Owner |
|---|---|---|---|
| Confidence engine not calibrated | **CRITICAL** | Phase 3, Abstain reliability | Phase 2 eval |
| Query expansion produces noisy multi-query outputs | High | Disabled — do not re-enable | Post Phase 3 |
| Classifier accuracy not measured | Medium | Phase 4 adaptation | Phase 2 eval |
| No intra-list diversity metric implemented | Medium | Detecting duplicate retrieval | Phase 2 |
| No load test conducted | High | Production readiness | Phase 5 |

---

*END OF DOCUMENT — Version 2.0*