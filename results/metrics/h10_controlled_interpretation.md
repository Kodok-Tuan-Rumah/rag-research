# H10 — Controlled Offline Interpretation

## Question

"What is the dominant reason RRF Recall@10 is below BM25 Recall@10 in this sample?"

## Evidence

| mechanism | evidence | supported? |
|---|---|---|
| A. Dense misses do not matter | q06 and q15 are BM25-only cases where dense misses directly cause RRF demotion | **YES** |
| B. BM25-only cases are demoted by fusion | q06: BM25 1 → RRF 24; q15: BM25 6 → RRF 11 | **YES** |
| C. Dense candidates introduce competing RRF scores | q06: 23 dense+BM25 mixed chunks outscore BM25-only gold | **YES** |
| D. Section boosts alter gold ordering | q06: boost 1.0 (no effect); q15: boost 1.4 (helps but insufficient) | **PARTIAL** |
| E. Filtering affects final hit status | `removed_before_gold=0` for both regressions | **NO** |
| F. Combination of mechanisms | B + C combined explain both regressions | **YES** |

## Dominant Reason

**B + C: BM25-only cases are demoted by fusion because dense candidates introduce competing RRF scores.**

When dense misses a BM25-only gold chunk:
1. The gold chunk receives only half-weight BM25 contribution.
2. Many dense+BM25 mixed chunks receive combined contributions from both sources.
3. These mixed chunks outscore the BM25-only gold chunk.
4. The gold chunk is demoted below top-10.

**Note:** This interpretation is based on the current 14-query canonical sample only. No generalization beyond this dataset is made.
