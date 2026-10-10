# K7.13 — Retrieval Decision Record

- **Date:** 2026-10-08
- **Decision status:** Provisional; based on the current 13-query canonical set and one paired latency replication.
- **Decision:** Retain **RRF-only as the production default**. Keep reranking opt-in for controlled evaluation; do not promote reranker@5 or @20 as the default on the current evidence.
- **Scope:** This is an evaluation decision, not a production code/configuration change.

## Evidence reviewed

| Evidence | Artifact | Key result |
|---|---|---|
| K7.10 coverage audit | [`k710_coverage_audit_20261008T063515Z.md`](k710_coverage_audit_20261008T063515Z.md) | 12/13 gold chunks are already at RRF rank 1. The only rank>1 gold is q06 at rank 24, outside the reranker@5 input. Difficulty and answer-type labels are `UNKNOWN` for all 13 queries. |
| K7.11 paired latency replication | [`k711_latency_replication_20261008T064223Z.md`](k711_latency_replication_20261008T064223Z.md) | 3 interleaved repetitions per mode and query: 39 measurements/mode. Median total latency was 53.31 ms RRF-only, 574.72 ms reranker@5, and 3,095.61 ms reranker@20. |
| K7.12 q11/q20 forensics | [`k712_q11_q20_forensics_20261008T065401Z.md`](k712_q11_q20_forensics_20261008T065401Z.md) | Both live CPU reranker@5 traces reproduced the saved K7.9 candidate/input/output order. q11 and q20 each moved the gold from rank 1 to rank 2. |

K7.9 reference provenance used by the audits:

- Dataset file SHA-256: `bc96e95372eeb2abebe917c15cb1dbb289c2bb3fe63cbcb1ab4f68c64008c42e`
- Documents SHA-256: `9048aca17d69289a7174edc3d938ea2d79f03896d3900deba76f9fb6c7f0dc8a`
- Chunks SHA-256: `4375a26668366674473a8917a3d0bc4ce7a2e53ace913457b80ff09be3481ba6`
- K7.9 recorded a dirty working tree. The current K7.10–K7.12 audits validate dataset and corpus hashes, but cannot establish that every other working-tree source file at the time of K7.9 matched the current tree.

## Retrieval quality and coverage

The K7.9 comparison and K7.11 quality sanity results agree:

| Mode | Recall@10 / Hit@10 | MRR@10 |
|---|---:|---:|
| RRF-only | 0.9231 | 0.9231 |
| Reranker@5 | 0.9231 | 0.8462 |
| Reranker@20 | 0.9231 | 0.8462 |

The 13-query set provides limited evidence about reranking's ability to *improve* ranking. Twelve gold chunks already occupy RRF rank 1, so there is no upward rank movement available for those queries. The remaining gold, q06, is at rank 24 and not among the five candidates passed to reranker@5; reranking that shortlist cannot recover it. Consequently, there are zero observed queries where the gold is both below rank 1 and available to reranker@5 to improve.

The measured reranker@5 ranking changes were regressions for q11 and q20:

| Query | Gold | Gold RRF rank → reranked rank | Competitor promoted above gold | Cross-encoder score: gold vs competitor |
|---|---|---:|---|---:|
| q11 | `tier1_rag_lewis2020_rs_0026` | 1 → 2 | `tier1_rag_lewis2020_rs_0027` (RRF rank 2 → reranked rank 1) | 8.02268 vs 8.61904 |
| q20 | `tier3_llama2_rs_0152` | 1 → 2 | `tier3_llama2_rs_0086` (RRF rank 2 → reranked rank 1) | 4.14714 vs 4.31879 |

Both queries remain within Hit@10, so the observed quality cost is lower reciprocal rank rather than a lost Hit@10. These examples show what happened for these candidates; they do not establish why the cross-encoder preferred the competitors in general.

## Latency and operational cost

The K7.11 run used CPU, the same 13 queries and corpus, the same model names, `max_length=326`, candidate pool 40, and three interleaved repetitions per query/mode. It excluded one warm-up query and recorded 39 measurements per mode.

| Mode | Median total | P95 total | Median reranker stage |
|---|---:|---:|---:|
| RRF-only | 53.31 ms | 113.33 ms | 0 ms |
| Reranker@5 | 574.72 ms | 1,095.07 ms | 533.63 ms |
| Reranker@20 | 3,095.61 ms | 3,549.29 ms | 3,050.18 ms |

Paired median total latency deltas were +518.77 ms for reranker@5 versus RRF-only and +2,482.40 ms for reranker@20 versus reranker@5. The repeated CPU measurements varied (total standard deviation: 197.236 ms for @5 and 456.003 ms for @20), but reranker execution remained the dominant added stage. These figures are specific to this machine, model/runtime, and workload; they are not a hardware-independent service-level estimate.

## Decision and rationale

Keep RRF-only as the default for now. On this fixed set, reranking did not improve Recall@10 or Hit@10, reduced MRR@10 by 0.0769, and added substantial CPU latency. The coverage audit also shows that the set has almost no rank-improvement headroom, so the result is insufficient to conclude that reranking has no value on harder queries. The appropriate interpretation is **“no demonstrated benefit on this small, rank-1-heavy set, with a measured latency cost”**, not “rerankers are universally ineffective.”

Reranker@5 may remain available as an explicit opt-in for experiments or use cases with a separate latency/quality requirement. The evidence does not justify changing production defaults or retrieval thresholds.

## Uncertainty and limits

- Only 13 canonical queries were evaluated; 12/13 have gold at RRF rank 1.
- q06 is the only below-rank-1 gold, and its rank 24 is outside the reranker@5 candidate shortlist.
- All `difficulty` and `answer_type` labels in the v2 set are `UNKNOWN`; results cannot be stratified by those dimensions.
- q11 and q20 are useful forensic cases but only two examples; they cannot characterize cross-encoder relevance behavior broadly.
- K7.11 is one interleaved replication session with three repetitions, one machine, CPU execution, and a warm-up exclusion. It characterizes observed local variability, not fleet-wide or GPU latency.
- The dataset envelope's declared content hash differs from the dataset file-byte hash. Both values are retained in the artifacts; this discrepancy does not change the validated query/gold alignment, but hash-generation provenance remains unresolved.
- The canonical audits establish exact dataset/corpus identity and reproduce q11/q20 retrieval order, but K7.9 recorded `working_tree: dirty`; exact historical source-tree identity is therefore not fully established.

## Criteria for revisiting this decision

Reassess only after a broader, validated evaluation set provides enough cases in which gold is retrieved below rank 1 and is present inside the reranker candidate cutoff. The next comparison should:

1. Include more diverse and difficult queries, with reviewed difficulty/type labels and verified gold mappings.
2. Separate candidate-retrieval misses from ranking errors; report gold rank before reranking and whether it entered the reranker input.
3. Compare paired quality and latency on the same corpus/model/configuration, with repeated interleaved runs and relevant deployment hardware.
4. Record per-candidate RRF and cross-encoder scores for regressions and improvements, and report confidence intervals or other uncertainty appropriate to the sample size.
5. Define the acceptable quality gain and latency budget before examining the next results.

## Non-changes

This decision record does not modify the canonical dataset, gold labels, RRF implementation, candidate cutoff, confidence thresholds, retrieval defaults, or API behavior. No benchmark/evaluation was run to produce this document; it synthesizes the saved K7.10–K7.12 artifacts.
