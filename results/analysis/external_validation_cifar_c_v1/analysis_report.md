# CIFAR-C External Validation Analysis

Generated: `2026-09-21T09:58:22.953061+00:00`

## Integrity

The combined ledger covers 2700 of 2700 registered jobs with no missing or duplicate matrix cells. Source immutability and temporal ordering were checked before statistics.

## Method summaries

| Method | H_abs mean | H_abs 95% CI | H_upd mean | Target accuracy |
|---|---:|---:|---:|---:|
| source | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.565222 |
| tent | 0.067583 | [0.064588, 0.071842] | 0.067583 | 0.535114 |
| eata | 0.649115 | [0.639611, 0.655947] | 0.649115 | 0.187878 |
| sar | 0.036729 | [0.033396, 0.040460] | 0.036729 | 0.550940 |
| cotta | 0.003145 | [0.002592, 0.003698] | 0.003145 | 0.562429 |
| rotta | 0.000117 | [-0.000018, 0.000313] | 0.000117 | 0.565103 |

## Interpretation boundary

The frozen external protocol uses `bn_affine_frozen_stats`, so configuration_gap is zero in all audited shards and H_abs equals H_upd. This external campaign therefore tests update-induced return loss under a fixed normalization policy; it does not replace the formal_v2 normalization decomposition.

EATA's target accuracy is substantially lower than the other methods in this independent reimplementation. That observation is retained as a stability signal and is not promoted to a universal claim about EATA without an implementation-sensitivity audit against the pinned upstream reference.

All inferential comparisons use source-training seed as the independent replication unit, exact two-sided sign-flip tests, 95% seed-cluster bootstrap intervals, and Holm correction within each registered family.
