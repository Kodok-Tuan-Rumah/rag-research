# q15 Rank Forensic

```json
{
  "query_id": "q15",
  "gold_chunk_id": "tier1_rag_lewis2020_rs_0046",
  "dense_rank": null,
  "bm25_rank": 6,
  "rrf_rank": 11,
  "reranker_rank": 19,
  "final_status": "out",
  "rank_delta_rrf": null,
  "rank_delta_reranker": 8,
  "failure_stage": "TOP_K_CUTOFF",
  "dense_score": null,
  "bm25_score": 13.155501042255782,
  "rrf_score": 0.010606,
  "reranker_score": -8.474681854248047,
  "final_rank": null,
  "latency_trace": {
    "dense": 36.84,
    "bm25": 5.85,
    "fusion": 1.77,
    "reranker": 2921.85,
    "fuse_and_rerank": 2923.62,
    "total": 2979.48
  }
}
```
