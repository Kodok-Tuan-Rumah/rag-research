# I4 — Architectural Decision Support

## Component Assessment on Current 14-Query Canonical Sample

### BM25

| metric | value | evidence limitation |
|---|---:|---|
| Recall@10 | 1.0000 | N=14, no CI significance |
| MRR@10 | 0.8833 | N=14 |
| P50 latency | ~7 ms | directly measured |
| P95 latency | ~10 ms | directly measured |

**Classification: HIGH_VALUE_ON_CURRENT_SAMPLE**

### Dense

| metric | value | evidence limitation |
|---|---:|---|
| Recall@10 | 0.7857 | N=14 |
| MRR@10 | 0.7551 | N=14 |
| P50 latency | ~40 ms | directly measured |
| P95 latency | ~50 ms | directly measured |

**Classification: UNCERTAIN**

**Reason:** The dataset contains 0 dense-only gold cases. Dense retrieval's value for hybrid coverage cannot be assessed from this sample. Dense may add value on a larger dataset with dense-only gold cases, but this is untestable here.

### RRF (Dense + BM25 Fusion)

| metric | value | evidence limitation |
|---|---:|---|
| Recall@10 | 0.8571 | N=14, McNemar p=0.50 |
| MRR@10 | 0.8571 | N=14 |
| P50 latency | ~48 ms | directly measured |
| P95 latency | ~63 ms | directly measured |

**Classification: LOW_VALUE_ON_CURRENT_SAMPLE**

**Reason:** RRF Recall@10 (0.8571) < BM25 Recall@10 (1.0000). RRF loses 2 BM25-only gold cases (q06, q15) and adds 0 dense-only gold coverage. The only benefit is rank improvement for q04 (5→1, both already in top-10).

### Reranker

| metric | value | evidence limitation |
|---|---:|---|
| Recall@10 | 0.8571 | same as RRF (no crossing change) |
| MRR@10 | 0.7627 | N=14 |
| P50 latency | ~3171 ms | directly measured |
| P95 latency | ~4065 ms | directly measured |

**Classification: LOW_VALUE_ON_CURRENT_SAMPLE**

**Reason:** Reranker provides no Recall@10 improvement, degrades MRR@10 by -0.0982, and adds ~3.1 s p50 latency.

## Summary Table

| Component | Recall@10 | MRR@10 | P50 | P95 | Classification |
|---|---:|---:|---:|---:|---|
| BM25 | 1.0000 | 0.8833 | ~7 ms | ~10 ms | HIGH_VALUE_ON_CURRENT_SAMPLE |
| Dense | 0.7857 | 0.7551 | ~40 ms | ~50 ms | UNCERTAIN |
| RRF | 0.8571 | 0.8571 | ~48 ms | ~63 ms | LOW_VALUE_ON_CURRENT_SAMPLE |
| Reranker | 0.8571 | 0.7627 | ~3171 ms | ~4065 ms | LOW_VALUE_ON_CURRENT_SAMPLE |

## Scope Language

**All architectural value assessments are restricted to the current 14-query canonical sample.**

These classifications do NOT constitute "REMOVE" decisions. They provide diagnostic evidence for a later controlled ablation on a larger dataset.
