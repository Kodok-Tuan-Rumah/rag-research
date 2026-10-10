# D7 Candidate Selection Fix Report

## Baseline (pre-fix)

Frozen in `results/metrics/d7_baseline_freeze.json`.

| query | RRF rank | rerank input index | rerank rank | final rank |
|---|---:|---:|---:|---:|
| q06 | 24 | 2 | 1 | 1 |
| q15 | 11 | 2 | 16 | — |
| q20 | 1 | 12 | 7 | 7 |

Canonical metrics pre-fix:

```text
Dense Recall@10      = 0.7857
BM25 Recall@10       = 1.0000
RRF Recall@10        = 0.8571
RRF+Reranker@10      = 0.9286
Final@10             = 0.9286
MRR@10               = 0.8316
```

## Correctness Fix

File: `src/retrieval/hybrid_retriever.py`
Function: `MasterHybridRetriever._run_search`
Lines changed: 442–446

Before:

```python
rerank_candidates = sorted(
    candidates,
    key=lambda x: x.get("retrieval_score", 0.0) if isinstance(x, dict) else getattr(x, "retrieval_score", 0.0),
    reverse=True
)[:20]
```

After:

```python
# Correctness fix: candidates are already in RRF order after filtering.
# Do NOT re-sort by retrieval_score: dense RetrievalResult objects lack
# that attribute and would fall back to 0.0, destroying RRF order.
rerank_candidates = candidates[:20]
```

Reason: `candidates` is the RRF-ordered list after low-quality filtering. The secondary sort used `retrieval_score`, which exists on BM25 dicts but not on dense `RetrievalResult` objects, causing dense candidates to fall back to 0.0 and destroying RRF order before reranking.

## Regression Test

File: `tests/test_hybrid_retriever.py`
Test: `test_rerank_candidates_preserve_rrf_order` (appended after line 63)

Assertion:

```python
rrf_order = [FakeDense("dense_a"), bm25_item, FakeDense("dense_c")]
...
result = retriever.search("test query", top_k=3)
ids = [r["chunk_id"] for r in result]
assert ids == ["dense_a", "bm25_chunk", "dense_c"]
```

The test fails under the old implementation (BM25 dict would sort first due to `retrieval_score=0.01` > dense fallback `0.0`) and passes under the corrected implementation.

## Candidate Order Before vs After

| query | RRF rank | rerank input BEFORE | rerank input AFTER | rerank rank BEFORE | rerank rank AFTER | final rank BEFORE | final rank AFTER |
|---|---:|---:|---:|---:|---:|---:|---:|
| q06 | 24 | 2 | **None** (not in top-20) | 1 | — | 1 | — |
| q15 | 11 | 2 | **11** | 16 | 19 | — | — |
| q20 | 1 | 12 | **1** | 7 | 8 | 7 | — |

Interpretation:

- q06: previously entered reranker at index 2 due to the sort-key bug; after the fix it is correctly excluded from the reranker input because its RRF rank (24) exceeds the rerank depth (20).
- q15: previously entered at index 2; after the fix it enters at its true RRF position 11 and is reranked to 19.
- q20: previously entered at index 12 despite RRF rank 1; after the fix it enters at index 1 and is reranked to 8.

Reranker input order now preserves RRF order.

## Retrieval Metrics Before vs After

| Stage | Recall@10 BEFORE | Recall@10 AFTER | MRR@10 BEFORE | MRR@10 AFTER |
|---|---:|---:|---:|---:|
| dense_only | 0.7857 | 0.7857 | 0.7551 | 0.7551 |
| bm25_only | 1.0000 | 1.0000 | 0.8833 | 0.8833 |
| rrf_fused | 0.8571 | 0.8571 | 0.8666 | 0.8666 |
| rrf_reranked | 0.9286 | **0.8571** | 0.8361 | **0.7627** |
| final_filtered | 0.9286 | **0.8571** | 0.8316 | **0.7589** |

No parameter changes. The metric movement is a direct consequence of the correctness fix: candidates that were incorrectly promoted by the sort-key bug (e.g., q06) are no longer reranked, and candidates that were incorrectly demoted (e.g., q20) now enter the reranker at their true RRF position.

## Interpretation

### Correctness effect

- Reranker input order now equals RRF order after filtering.
- Dense `RetrievalResult` objects are no longer demoted by a missing `retrieval_score` attribute.
- The candidate-selection path is internally consistent with the RRF stage.

### Metric effect

- `rrf_reranked` and `final_filtered` recall decreased because q06 (previously rescued by the bug) no longer enters the reranker.
- q20 now enters the reranker at position 1 but is still reranked to 8, so it remains in the final top-10.
- q15 enters at position 11 and is reranked to 19, so it is not in the final top-10.

### Reranker effect

- The reranker itself was not modified. Its behavior on the corrected candidate set is the true baseline.
- The previous "rescue" of q06 was an artifact of the candidate-selection bug, not a reranker capability.

## Remaining Unknown

- Whether the reranker's demotion of q15 (RRF 11 → rerank 19) and q20 (RRF 1 → rerank 8) reflects true cross-encoder judgment requires manual review of the q15/q20 competitor texts.
- The q15 manual judgment (D4) labeled the gold as only weakly relevant (0.25), which is consistent with the reranker demoting it, but this is not a formal proof of correctness.
