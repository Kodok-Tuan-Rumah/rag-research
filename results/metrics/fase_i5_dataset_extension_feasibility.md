# I5 — Dataset Extension Feasibility

## Available Sources

| Source | Rows | Valid Gold | Notes |
|---|---:|---:|---|
| `data/debug/failure_analysis.json` | 20 | 10 (v3 validated) | 6 unmapped (evaluation_data_error) |
| `data/adversarial/adversarial_queries.json` | 6 | 0 | No gold chunk IDs |
| `results/metrics/ragas_phase2_postfix*.json` | 45 | 0 | Question-only, no gold chunk IDs |
| `results/metrics/ablation_sprint2.json` | 30 | 0 | Stale gold IDs (30/30 missing from corpus) |

## Validated Rows

| Source | Valid | Duplicates Removed | Evaluation Row Only |
|---|---:|---:|---:|
| failure_analysis.json | 10 | 0 | 10 |
| adversarial_queries.json | 0 | 0 | 6 |
| ragas_phase2_postfix*.json | 0 | 0 | 45 |
| ablation_sprint2.json | 0 | 0 | 30 |

## Realistic Non-Overlapping Expansion Size

**Current canonical: 14 queries**
**Validated expansion available: 0 new queries**

The 10 validated v3 rows are a subset of the 14 canonical queries (they overlap). No additional high-integrity gold queries can be added from existing artifacts without:
1. Fabricating gold labels (PROHIBITED)
2. Using stale IDs (PROHIBITED)
3. Using question-only rows without gold evidence (PROHIBITED)

## Conclusion

**Is N=14 likely to remain the effective evaluation size unless new evidence sources are introduced?**

**YES.**

Unless new gold-labeled data is created (e.g., manual annotation, new QA generation with verified gold chunk IDs), the effective evaluation size remains 14 queries.

**Note:** This is a data availability constraint, not a technical limitation. New evidence sources could expand the dataset, but that requires explicit authorization and new data creation.
