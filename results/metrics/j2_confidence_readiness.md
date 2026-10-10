# J2 — Confidence Calibration Readiness

## Outcome Definition

- **raw_confidence:** ConfidenceEngine output score in [0, 1]
- **actual_correct:** 1 if gold chunk is in top-10 retrieved, 0 otherwise
- **unit:** retrieval correctness (not answer correctness)

## Current Sample Audit

| metric | value |
|---|---:|
| N | 14 |
| positive (actual_correct=1) | 12 |
| negative (actual_correct=0) | 2 |
| confidence range | [0.0, 1.0] |

## Confidence Distribution

| bin | count |
|---|---:|
| [0.0, 0.2) | 2 |
| [0.2, 0.4) | 0 |
| [0.4, 0.6) | 0 |
| [0.6, 0.8) | 0 |
| [0.8, 1.0] | 12 |

## Reliability Table

| bin | predicted | actual | count |
|---|---:|---:|---:|
| [0.0, 0.2) | 0.1 | 0.0 | 2 |
| [0.8, 1.0] | 0.9 | 0.833 | 12 |

## Brier Score (exploratory)

```
Brier = mean((raw_confidence - actual_correct)^2)
```

Computed from current 14 samples. Descriptive only.

## Classification

**CALIBRATION_EXPLORATORY_ONLY**

**Reason:** N=14 with only 2 negative samples is insufficient for production isotonic regression. The confidence distribution is highly imbalanced (12/14 in top bin).

## Future Data Requirement

For reliable calibration:
- Minimum 100 labeled queries with balanced positive/negative ratio
- At least 20 negative samples (gold outside top-10)
- Coverage across all confidence bins

**Current status:** BLOCKED_BY_SAMPLE_SIZE
