# J0 — Evaluation Gate

## Gate Checks

| Check | Status | Evidence |
|---|---|---|
| MRR@10 definition | PASS | RR@10 = 1/rank if 1<=rank<=10, else 0; verified from per-query values |
| Aggregate reproducibility | PASS | mean(per_query_rr) == aggregate MRR@10 for both Config A and B |
| Gold integrity | PASS | All 14 items have gold_chunk_id, gold_doc_id, question, provenance |
| Confidence contract | PASS | Confidence engine predicts retrieval correctness; raw_confidence in [0,1] |
| Current system identity | PASS | Python 3.14.4, 796 chunks, MiniLM embedder, ms-marco reranker, 14 queries |

## Final

**J0_GATE = PASS**

All metric contracts are consistent. Aggregate MRR@10 is reproducible from per-query RR@10 values. Gold provenance is intact. Confidence engine contract is defined. Current system identity is documented from live artifacts.
