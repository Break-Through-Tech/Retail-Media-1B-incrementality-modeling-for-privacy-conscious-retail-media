# Data Validation Summary

Created: July 31, 2026

## Readiness Decision

**The prepared files are ready for educational uplift modeling with the limitations documented
below.**

The full source passed schema, missing-value, finite-value, binary-value, and cross-field checks.
The one-million-record sample closely preserves feature means and key treatment and outcome rates
from the source.

## Prepared Files

| Split | Rows | Treatment | Control | Visits | Conversions | Exposures |
|---|---:|---:|---:|---:|---:|---:|
| Training | 700,000 | 594,967 | 105,033 | 32,857 | 1,987 | 21,336 |
| Validation | 150,000 | 127,493 | 22,507 | 7,041 | 426 | 4,659 |
| Test | 150,000 | 127,493 | 22,507 | 7,041 | 426 | 4,598 |
| **Total** | **1,000,000** | **849,953** | **150,047** | **46,939** | **2,839** | **30,593** |

## Full-Source Quality Checks

- Source records checked: 13,979,592
- Missing values: 0
- Infinite feature values: 0
- Invalid values in binary columns: 0
- Control records marked as exposed: 0
- Conversion records without a visit: 0
- First control record: source row 5,571,800

## Representativeness Checks

- Largest absolute difference in an overall binary rate: 0.007768 percentage points
- Largest treatment-conditional rate difference: 0.050691 percentage points
- Largest absolute standardized feature-mean difference: 0.000975
- Source-row overlap across splits: 0
- Exact anonymous row-pattern overlap across splits: 0
- Prepared files reproduced with byte-identical SHA-256 checksums

The sample contains 8,426 repeated anonymous row patterns beyond the first occurrence. The source
does not provide a unique identifier, so these records may represent different experiment units.
They were retained, but every identical pattern was kept within one split to prevent the same
pattern from appearing in both training and evaluation data.

## Important Limitations

1. The meanings of `f0` through `f11` are intentionally hidden. Results cannot be translated into
   named customer characteristics or business personas.
2. The dataset has no identity keys. It cannot be used to build or demonstrate identity matching.
3. `exposure` occurs after treatment assignment and should not be used as a normal feature in the
   primary treatment-assignment analysis.
4. Conversion is rare. Visit should remain the primary outcome.
5. The SolePeak and MarketBridge data clean room workflow is a teaching analogy. Criteo does not
   state that the source data was produced by a retailer, retail media network, or data clean room.
6. Each validation and test split contains 871 control visits. Small differences between model
   results may therefore reflect uncertainty rather than a reliable performance difference.
7. The public files omit the source row numbers used internally for sampling verification.

## Required Student Checks

- Verify shapes, columns, and treatment and outcome rates after loading each file
- Exclude `exposure` from the primary feature matrix
- Do not filter the primary analysis to records where `exposure = 1`
- Fit preprocessing only on training data
- Use validation data for modeling decisions
- Preserve the test data for final evaluation
- Compare uplift models with random ranking and conventional response prediction
- Report treatment and control counts, rates, uplift, and uncertainty for every audience group
- Treat close model results as inconclusive when the evidence does not support a reliable winner
