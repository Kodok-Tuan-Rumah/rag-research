# H5 — Section Boost Forensics

## RRF Regression Queries

| query_id | gold_section | section_boost | dense_contrib | bm25_contrib | total_rrf_score | implicated? |
|---|---|---:|---:|---:|---:|---|
| q06 | ii. overview of rag | 1.0 | None | 0.008197 | 0.008197 | No (no boost) |
| q15 | methods | 1.4 | None | 0.010606 | 0.010606 | Partial (boost helps but insufficient) |

## Analysis

**q06:** Section "ii. overview of rag" receives boost 1.0 (no boost). The regression is NOT caused by section boost — it is caused by BM25-only weight effect.

**q15:** Section "methods" receives boost 1.4. The boost increases the BM25 contribution from 0.007576 (without boost) to 0.010606 (with boost). However, this is still insufficient to overcome dense competition. Section boost is NOT the primary cause — it partially mitigates but does not prevent the regression.

## Conclusion

**OBSERVATION:** Section boost is not the primary cause of RRF regressions in this sample. The dominant mechanism is BM25-only weight effect when dense misses the gold chunk.
