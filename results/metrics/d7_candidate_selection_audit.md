# D7_1 — RECONSTRUCT ACTUAL CANDIDATE SELECTION PATH

## Stage-by-stage order in `src/retrieval/hybrid_retriever.py`

| stage | how it is built | source order | key used | row form |
|---|---|---|---|---|
| dense raw | `vector_store.search` -> `_dense_search` | store/index order | cosine score | `RetrievalResult` dataclass object |
| BM25 raw | `self._bm25.get_scores` + `np.argsort(scores)[::-1]` | BM25 normalized score order | BM25 score | `dict` with keys `chunk_id/text/score/doc_id/metadata` |
| RFF fused | `rrf_fuse(all_dense_hits, all_bm25_hits, candidate_k=40, dense_weight=0.5, bm25_weight=0.5, source_labels=...)` | first occurrence wins for duplicates; final list sorted by `fused_scores[cid] desc` | `score` if dense object, `retrieval_score` if dict | mixed `RetrievalResult` and `dict` |
| qualified | `[c for c in fused_results if not _cf_is_low_quality_chunk(c)]` | same order as fused RFF | low-quality heuristic | same |
| rerank input | sorted filtered candidates[:20] | **not RFF order** | `retrieval_score` for dict items, getattr(RetrievalResult, `retrieval_score`, 0.0) for dense object | mixed, but dense items carry fallback 0.0 |
| reranker output | `_rerank` calls `score_pairs`, then sorts by `rerank_score` desc | cross-encoder score order | cross-encoder scores | same item types |

###Terminal order per real query

1. dense raw result ordering: determined by qdrant index.
2. BM25 raw result ordering: determined by BM25Okapi.
3. RFF fused ordering: `rrf_fuse` sorts `fused_scores`
4. qualified ordering: unchanged fused order after dropping low-quality.
5. rerank input ordering: **mixed wrong key**.
6. reranked ordering: sorted by `rerank_score`.

---

# D7_2 — OFFICIAL MEANING OF `rerank_input_index`

`rerank_input_index` viết trong artifact cũ là 1-based index **After candidate selection sort**, i.e. the order passed to `_rerank`.

It is **not**:

- RRF fused list position
- qualified position after low-quality
- RFF top20 slot
- final sort

In current trace:
```"rerank_input_count": 20```
`rerank_input_index` belongs to that same list? Let's verify by using `rerank_input_ids` in trace: q06 appears at index 2 of `rerank_input_ids` output after reran with the key used above. This corresponds not to RFF ranking.

---

# D7_3 — CANDIDATE SORT KEY AUDIT

## OBSERVATION

In `_run_search`:

```python
rerank_candidates = sorted(
    candidates,
    key=lambda x: x.get("retrieval_score", 0.0) if isinstance(x, dict) else getattr(x, "retrieval_score", 0.0),
    reverse=True
)[:20]
```

- For dict hits (BM25 items), the key is `x["retrieval_score"]`.
- For `RetrievalResult` dense hits, the code uses `getattr(x, "retrieval_score", 0.0)`, but this attribute does **not** exist on `RetrievalResult`.

In `src/retrieval/qdrant_store.py`:
```python
@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id:   str
    text:     str
    score:    float
    metadata: dict
```
No `retrieval_score` field.

In fusion output (`rrf_fuse`): dense items only assign `item.score = rrf_score`; BM25 dict items assign `item["retrieval_score"] = rrf_score`.

Therefore:

- dict items with `retrieval_score` get their real fused score as sort key.
- dense `RetrievalResult` objects get real fused score in `.score`, but candidate sort reads `.retrieval_score`, missing, i.e. `0.0`.

## IMPACT

This affects all queries in the same mechanical way:

- All dense `RetrievalResult` hits fall back to key 0.0.
- Only BM25 dict items can place in first slots in the first positive section.
- Since `reverse=True`, any candidate with positive `retrieval_score` outranks any real dense object mapped to key 0.0.
- Therefore the reranker candidate pool is effectively biased by BM25-derived items that have a proper `retrieval_score`.

## PROOF: q06 / q15 / q20

Exact dynamic evidence:

| query | gold object type | has `retrieval_score` attr? | candidate sort key | rrf rank | candidate idx after filter |
|---|---|---|---:|---:|---:|
| q06 | dict (BM25-origin) | yes | 0.008197 | 24 | 2 |
| q15 | dict (BM25-origin) | yes | 0.010606 | 11 | 2 |
| q20 | RetrievalResult (dense-origin) | MISSING | 0.0 | 1 | 12 |

This means q20 had the best *fused* rank 1 but got sort key 0, pushing it below many BM25 hits.
q06/q15 were pushed into reranker input even though their fused-ranked positions were lower, because their key is a small positive BM25 retrieval_score that beats 0.0 dense keys.

## UNKNOWN

We cannot know whether the original intent was to sort dense by `r.score`. But the FACT remains:
```"getattr(dense_result, 'retrieval_score', 0.0)" cannot work because field absent.```
This is not a hypothesized path error; it is code inspection of actual line in branch.

---

# D7_4 — q06 RECONSTRUCT

| stage | rank | chunk_id | score/source |
|---|---:|---|---|
| BM25 raw | 1 | tier1_advanced_rag_survey_rs_0028 | BM25 score 20.846 |
| dense raw | none | not present | MISSING |
| RRF | 24 | tier1_advanced_rag_survey_rs_0028 | retrieval_score 0.008197, dense_contrib None, bm25_contrib 0.008197 |
| qualified | 24 | tier1_advanced_scn… rs_0028 | qualified list after low-quality uses original RFF order but rank |
| rerank input | 2 | tier1_advanced_rag_survey_rs_0028 | retrieval_score 0.008197, dense score fallback 0.0 |
| reranked | 1 | tier1_advanced_rag_survey_rs_0028 | reranker_score 6.276 |

Evidence: soft-evidence but all steps mechanically reconstructible.

- BM25 rank 1 alone cannot explain RFF rank 24. It contributes only half-weight because dense misses.
- q06’s RFF 24 is explained by no dense contribution and all dense-supported chunks scoring above it.
- Candidate filter preserves top list except low-quality; `rerank_candidates` however use the wrong key for dense objects, butq06 is a dict and still enters top-20 because its retrieval_score>all dense fallbacks? Yes: many dense objects fallback 0.0, buttheir real fused score maybe high, they should have been candidates but are treated as 0.0 and cut.
-q06 appears reranked because in candidate sort it is sort key 0.008197, which is higher than 0.0 dense objects.

Conclusion for q06:
**A is partially true: q06’s raw BM25 rank 1 did not preserve in RFF. Its rerank input position is an artifact of the candidate sort key, notfiltering. Filtering did not remove candidates before it.**

---

# D7_5 — q15 RECONSTRUCT

| stage | rank | chunk_id | score/source |
|---|---:|---|---|
| BM25 | 6 | tier1_rag_lewis2020_rs_0046 | BM25 score 13.155 |
| dense | none | not present | MISSING |
| RFF | 11 | tier1_rag_lewis2020_rs_0046 | retrieval_score 0.010606, dense_contrib None |
| qualified | 11 | tier1_rag_lewis2020_rs_0046 | qualified rank in raw fused order |
| rerank input | 2 | tier1_rag_lewis2020_rs_0046 | retrieval_score 0.010606, dense fallback 0.0 |
| reranked | 16 | tier1_rag_lewis2020_rs_0046 | reranker_score -8.4747 |

Same signature:
- gold is a bm25 dict positive candidate
- gold lacks dense contribution; RFF rank is low
- candidate sort key for dense objects is 0.0
- however q06 has a middling cross-encoder score and落在16.

---

# D7_6 — q20 RECONSTRUCT

| stage | rank | chunk_id | score/source |
|---|---:|---|---|
| dense raw | 2 | tier3_llama2_rs_0152 | dense score (likely Mutated after RFF) 0.016261 |
| BM25 raw | 1 | tier3_llama2_rs_0152 | true BM25 score, rel ker 36.5697 |
| RFF fused | 1 | tier3_llama2_rs_0152 | fused score 0.016261, object is dense RetrievalResult |
| qualified | 1 | tier3_llama2_rs_0152 | yes (filtered list keeps) |
| rerank input | 12 | tier3_llama2_rs_0152 | key 0.0 due to missing `.retrieval_score` |
| reranked | 7 | tier3_llama2_rs_0152 | reranker_score -5.579 |

Here BM25 rank 1 and dense rank 2 should have driven RFF 1. It did. But candidate order dropped it because its object type is dense `RetrievalResult`, which sets `.score` but not `.retrieval_score`. Sorting key reads `retrieval_score` missing ->0.0. It ranks behind positive BM25 keys. This explains deterministic mismatch: **the fuse list is used for rank tracking, but the candidate selection later disables dense-only scores from driving rerank selection.**

---

# D7_7 — CONSISTENCY STATUS

- `rerank_input_index` in old artifacts was computed as `enumerate` over `rerank_input_ids` from trace. This is **not** RFF rank.
- The trace wasinternallyconsistent once code paths are separated.
-No evidence yet for low-quality filter shifting RFF rank into rerank index slot. Drift yes, but driven by diverse retrieval scoring attributes, not low-qualityfilter.

FACTS:
- Currently for denseRetrievalResult, candidate sort key is 0.0.
- This biases candidate selection away from actual RFFordering.
- q06 BM25rank1 but RFFrank24 because dense no-show and mixed scoresdominate. Candidate index2 not because filter demoted RFF top candidates? Actually low-quality filter removing many? Did you see filtering actually changed order? In our computation for q06, `removed_before_gold=0` after low-quality. So no filter drift.
- q20 actually shows where OBJECT TYPE matters: dense-result is RetrievalResult, sort key 0, rerank input index 12 despite R20FF rank 1.

UNKNOWN/HYPOTHESIS:
- Whether the candidate sort key bug was intentional should be classified as a **production bug** until clarified. Its externally visible impact is confirmed.

---

# D7_8 — D3 consistency answers

1. Official meaning of rerank_input_index: index within `rerank_candidates` list after candidate filter + candidate sort.
2. Actual candidate-selection ordering: BM25 dicts sorted by `retrieval_score`; dense `RetrievalResult` objects sorted by 0.0 due to missing attribute.
3. Actual filtering before candidate sort: `is_low_quality_chunk` removal but no reordering. However because retrieval_score fallback is broken, the subsequent candidate sort reshuffles.
4. q06 reconstruction: see table above.
5. q15 reconstruction: see table above.
6. q20 reconstruction: see table above.
7. Dense/BM25 sort-key audit: exact line in source is evidence. unfus_reused? in current implementation, dense `RetrievalResult` is unkeyed in candidate sort.
8. Whether current trace is internallyconsistent: partially. `rrf_ids` trace says rank N based on list order, but `rerank_input_ids` is after a different sort, so naive relation "rrf_ids[10]=rerank_input_ids[11]" does not hold. Trace is internals paths consistent, but not consistent under simplistic assumption.
9. q06 rank24 explanation: now explained by weighting+candidate key bug, not tie.
10. Is current candidate selection production bug / instrumentation bug / UNKNOWN:

- **Candidate selection has a clear object-type bug in current implementation.**
- Not necessarily only an instrumentation false alias? The runtime behavior is flawed for dense items. This is a FACT. No model/training is changing.

---

# SIDE NOTE: canonical RRF audit is still valid as weighted + section-boosted RFF

- This implementation is not standard unweighted RRF.
- Calling it "canonical RRF" is risky.
- Use `weighted + section-boosted RFF` in all reports.
