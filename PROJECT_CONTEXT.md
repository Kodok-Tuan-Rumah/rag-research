# PROJECT: RAG Research System

> **Project direction:** failure-aware, single-turn research QA. Prioritize evidence quality, grounded answers, explicit abstention, and reproducible evaluation over model or pipeline complexity.
> **Runtime snapshot:** 2026-10-23. This file describes source behavior at the snapshot date; `PROJECT_GOALS.md` defines targets and phase gates. Current Phase 1 evidence is `results/metrics/phase1_smoke_v3.json` and is not inferred from this architecture summary.

## GOAL

Build a reliable research-domain RAG QA system that:

* retrieves evidence relevant to the query
* grounds claims in retrieved context and abstains when context is insufficient
* reports retrieval and post-generation confidence as distinct signals
* supports repeatable, provenance-rich evaluation

This is a project objective, not a claim that production readiness or calibrated confidence has already been achieved.

## CURRENT RUNTIME ARCHITECTURE

```text
Query
→ heuristic query policy
→ dense Qdrant + BM25
→ weighted RRF (k=60)
→ score/noise prefilter
→ optional cross-encoder reranking
→ strict chunk filter (minimum-filter fallback if strict returns none)
→ retrieval confidence and query-evidence decision
→ grounded Groq generation with an answer/context guard, or reasoned abstention
```

RRF-only is the default. The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` is loaded lazily on the first explicitly enabled rerank request; health checks do not load it. The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Runtime defaults and validation are defined in `src/api/config.py`, `src/api/main.py`, and `src/retrieval/hybrid_retriever.py`.

`PolicyEngine` classifies each query as factual, reasoning, or general. Both API and Agent now use the policy's top-k and generation settings by default; an explicit API `top_k` request value overrides only the policy top-k. This classifier is a keyword heuristic and does not emit a confidence score.

## CONFIDENCE AND DECISIONS

`ConfidenceEngine` currently computes heuristic confidence from normalized retrieval scores, score gap, entropy, source agreement, and—after generation—answer/context overlap. Its `REJECT`, `PARTIAL_TRUST`, and `GENERATE` outcomes are not statistically calibrated.

* `REJECT`: abstain without calling the LLM.
* `PARTIAL_TRUST`: generate using at most the top three chunks and mark the answer as low-confidence.
* `GENERATE`: generate with the selected chunks.
* Retrieval confidence/decision and post-generation confidence/decision are separate fields in the API response.

The decision path includes heuristic query-to-evidence relevance checks for broad or weakly supported questions. Generation also checks whether answer claims have lexical support in the selected context; insufficient evidence can turn a proposed answer into an abstention with a reason. These safeguards are heuristic regression controls, not semantic verification or a guarantee against hallucination. These rules describe the current API and Agent contract; they do not imply that confidence predicts factual correctness at a calibrated rate.

## EVALUATION SNAPSHOT

The K7.10–K7.13 experiments used `data/evaluation/retrieval_eval_dataset_v2.json` (13 queries), not the older 14-query dataset used for the README's historical baseline. K7.10 found 12/13 gold chunks at RRF rank 1; q06 was at rank 24 and outside reranker@5. All K7 v2 difficulty and answer-type labels were `UNKNOWN`.

K7.11's local CPU retrieval measurements (three interleaved repetitions, `max_length=326`) reported median retrieval totals of 53.31 ms for RRF-only, 574.72 ms for reranker@5, and 3,095.61 ms for reranker@20. These are retrieval benchmark measurements on one machine—not end-to-end API/LLM latency, a production SLA, or a general claim about reranker value. See the K7.10–K7.13 artifacts under `results/metrics/`.

## PHASE 1 ACCEPTANCE STATUS

The current working-tree `results/metrics/phase1_smoke_v3.json` records a full-generation 20-case run with `runtime_pass: true`. Four rows are provider errors after the Groq token quota was exhausted, and human review is pending (`human_review_complete: false`). Therefore `quality_pass: false` and `phase1_pass: false`; this run gives no final quality score. Rerun after quota reset and review the resulting report. This result does not establish production readiness or calibrated confidence. See [`PROJECT_GOALS.md`](PROJECT_GOALS.md#phase-1--pipeline-stabilization).

The current acceptance candidate (`data/evaluation/phase1_smoke_set_v3.json`) is a frozen
generation and abstention contract. It is distinct from retrieval benchmark
datasets; see [`data/evaluation/README.md`](data/evaluation/README.md).

## PROJECT CONSTRAINTS

* The engineering target is end-to-end P95 latency ≤3.5 seconds on the no-retry path; the original `<2s` aspiration is superseded by the budget in `PROJECT_GOALS.md`.
* Keep current model complexity bounded; require controlled evaluation before promoting optional retrieval features.
* Prefer validated data, corpus, and retrieval-quality improvements over adding model complexity.

## PROPOSED DIRECTION — FORENSIC AI (NOT IMPLEMENTED)

A product-direction proposal for sentence-level citation, evidence-conflict
detection, calibrated confidence certificates, and adversarial noise filtering
is recorded in [`PROJECT_GOALS.md`](PROJECT_GOALS.md#17-proposed-product-differentiators--forensic-ai).
None of these capabilities are implemented, and this section is not a runtime
description. Confidence remains heuristic and uncalibrated. The goals document
recommends citation-to-sentence attribution first and conflict detection second,
and treats calibrated confidence as the existing Phase 3 blocker rather than a
new feature.
