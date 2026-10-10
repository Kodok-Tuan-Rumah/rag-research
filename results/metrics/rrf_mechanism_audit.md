# RRF Mechanism Audit

- RRF formula: `sum(w_i * boost_i(DOC) / (k + rank_i + 1))`
- k: 60
- weights: dense 0.5, bm25 0.5
- boost from section/metadata: methods 1.4, experiments/result 1.3, abstract 1.1, intro 1.05, conclusion 1.0, refs 0.3, other 1.0
- Candidate pool: all dense hits + all BM25 hits before candidate_k truncation
- Duplicate handling: keyed by chunk_id; scores are summed
- Parent/subchunk: not deduplicated at fusion stage
- Ties: slotted by score then stable chunk_map insertion order
- Ordering: RFF score desc, filtered by low quality, then top candidate_k and rerank
- Observing: BM25strong wins if dense misses item; BM25 is weighted like dense in current config
