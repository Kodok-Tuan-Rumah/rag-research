# J1 — Dataset Expansion Feasibility

## Source Inventory

| Source | Rows | Valid Gold | Notes |
|---|---:|---:|---|
| failure_analysis.json | 20 | 10 (v3 validated) | 6 unmapped (evaluation_data_error) |
| adversarial_queries.json | 6 | 0 | No gold chunk IDs |
| ragas_phase2_postfix*.json | 45 | 0 | Question-only, no gold chunk IDs |
| ablation_sprint2.json | 30 | 0 | Stale gold IDs (30/30 missing from corpus) |

## Evidence-First Matching

| match_type | count | queries |
|---|---:|---|
| EXACT_ANSWER_SPAN | 2 | q01, q03 |
| TOKEN_RECALL_VALIDATED | 8 | q02,q04,q09,q10,q11,q13,q14,q16,q19,q20 |
| evaluation_row_only | 74 | unmapped failure_analysis + ragas + adversarial |

## Non-Overlap Audit

| group | count | queries |
|---|---:|---|
| CURRENT_CANONICAL_14 | 14 | q01-q04,q06,q09-q11,q13-q16,q19,q20 |
| NEW_VALIDATED_CANDIDATES | 0 | — |

The 10 validated v3 rows are a subset of the 14 canonical queries. No genuinely new queries can be added from existing artifacts.

## Category Coverage

| category | count | queries |
|---|---:|---|
| factual | 7 | q01,q03,q04,q09,q10,q13,q14 |
| reasoning | 7 | q02,q06,q11,q15,q16,q19,q20 |
| general | 0 | — |
| adversarial | 0 (separate) | adv01-adv06 |

## Dense/BM25 Coverage

| category | count | queries |
|---|---:|---|
| dense-only | 0 | — |
| BM25-only | 2 | q06, q15 |
| both | 12 | q01-q04,q09-q11,q13-q14,q16,q19,q20 |
| neither | 0 | — |

## Realistic Expansion Size

**0 new queries** can be added from existing artifacts without fabrication.

## Conclusion

1. **Valid query tambahan yang benar-benar tersedia:** 0
2. **Genuinely new:** 0 (all validated rows overlap with canonical 14)
3. **N=14 dapat diperbesar secara defensible?** TIDAK dari artifact yang ada
4. **Dense-only cases dapat ditemukan secara evidence-first?** TIDAK (0 cases)
5. **Corpus/evidence source saat ini menjadi bottleneck?** YA — tidak ada sumber gold baru yang terverifikasi
