# Trajectory Ablation GPU v1 Analysis

Generated: `2026-09-20T11:57:47.194895+00:00`

All 1,080 formal shards passed hash, provenance, temporal-order, mutation-contract, metric-range, and source-immutability checks.
Inference uses five source-training seeds as the independent replication units. Experimental rows are not treated as independent samples.

## Method summaries

| Method | Mean Habs | 95% CI | Mean Hupd | Mean target accuracy |
|---|---:|---:|---:|---:|
| SOURCE | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.6587 |
| TENT | 0.311492 | [0.294957, 0.325729] | 0.311492 | 0.4620 |
| EATA | 0.732532 | [0.724703, 0.740856] | 0.732532 | 0.1679 |
| SAR | 0.034163 | [0.029520, 0.038692] | 0.034163 | 0.6279 |
| COTTA | 0.010062 | [0.008533, 0.011774] | 0.010062 | 0.6515 |
| ROTTA | 0.000023 | [-0.000028, 0.000077] | 0.000023 | 0.6587 |

## Planned Habs contrasts

Positive values mean the alternative condition has greater return-to-source loss.

| Method | Contrast | Mean difference | 95% CI | Exact p | Holm p |
|---|---|---:|---:|---:|---:|
| TENT | additional_shift | 0.087487 | [0.083909, 0.091853] | 0.0625 | 1.0000 |
| TENT | order_reversal | -0.019218 | [-0.028213, -0.014056] | 0.0625 | 1.0000 |
| TENT | recurrence | 0.121200 | [0.117669, 0.124464] | 0.0625 | 1.0000 |
| TENT | pass_1_to_2 | 0.055480 | [0.047447, 0.063513] | 0.0625 | 1.0000 |
| TENT | pass_1_to_4 | 0.184347 | [0.169612, 0.199082] | 0.0625 | 1.0000 |
| TENT | lr_mid_to_low | -0.219752 | [-0.255140, -0.182347] | 0.0625 | 1.0000 |
| TENT | lr_mid_to_high | 0.493573 | [0.467435, 0.519712] | 0.0625 | 1.0000 |
| EATA | additional_shift | 0.062811 | [0.057296, 0.068987] | 0.0625 | 1.0000 |
| EATA | order_reversal | -0.049247 | [-0.061273, -0.033989] | 0.0625 | 1.0000 |
| EATA | recurrence | 0.062331 | [0.058218, 0.067827] | 0.0625 | 1.0000 |
| EATA | pass_1_to_2 | 0.074605 | [0.057262, 0.088767] | 0.0625 | 1.0000 |
| EATA | pass_1_to_4 | 0.078587 | [0.062375, 0.091582] | 0.0625 | 1.0000 |
| EATA | lr_mid_to_low | -0.088092 | [-0.098518, -0.074893] | 0.0625 | 1.0000 |
| EATA | lr_mid_to_high | 0.002888 | [0.000752, 0.005748] | 0.0625 | 1.0000 |
| SAR | additional_shift | 0.011089 | [-0.014284, 0.034742] | 0.5000 | 1.0000 |
| SAR | order_reversal | -0.002976 | [-0.031391, 0.025049] | 0.9375 | 1.0000 |
| SAR | recurrence | -0.002620 | [-0.022780, 0.008964] | 1.0000 | 1.0000 |
| SAR | pass_1_to_2 | -0.002795 | [-0.010827, 0.004528] | 0.6875 | 1.0000 |
| SAR | pass_1_to_4 | 0.010930 | [-0.002933, 0.023970] | 0.3125 | 1.0000 |
| SAR | lr_mid_to_low | -0.016342 | [-0.017847, -0.014608] | 0.0625 | 1.0000 |
| SAR | lr_mid_to_high | 0.069462 | [0.054447, 0.083078] | 0.0625 | 1.0000 |
| COTTA | additional_shift | 0.000009 | [-0.002933, 0.002951] | 1.0000 | 1.0000 |
| COTTA | order_reversal | 0.004493 | [0.002409, 0.007029] | 0.0625 | 1.0000 |
| COTTA | recurrence | 0.008518 | [0.006240, 0.011660] | 0.0625 | 1.0000 |
| COTTA | pass_1_to_2 | 0.002690 | [0.001820, 0.003523] | 0.0625 | 1.0000 |
| COTTA | pass_1_to_4 | 0.010105 | [0.007118, 0.013012] | 0.0625 | 1.0000 |
| COTTA | lr_mid_to_low | -0.004693 | [-0.005778, -0.003693] | 0.0625 | 1.0000 |
| COTTA | lr_mid_to_high | 0.019473 | [0.016765, 0.022873] | 0.0625 | 1.0000 |
| ROTTA | additional_shift | 0.000011 | [0.000002, 0.000022] | 0.2500 | 1.0000 |
| ROTTA | order_reversal | -0.000002 | [-0.000022, 0.000018] | 1.0000 | 1.0000 |
| ROTTA | recurrence | 0.000098 | [-0.000002, 0.000209] | 0.2500 | 1.0000 |
| ROTTA | pass_1_to_2 | -0.000015 | [-0.000032, 0.000007] | 0.3125 | 1.0000 |
| ROTTA | pass_1_to_4 | 0.000103 | [0.000018, 0.000192] | 0.1250 | 1.0000 |
| ROTTA | lr_mid_to_low | 0.000020 | [0.000000, 0.000057] | 0.5000 | 1.0000 |
| ROTTA | lr_mid_to_high | 0.000073 | [-0.000012, 0.000157] | 0.2500 | 1.0000 |

## Claim boundary

the factorial campaign estimates how trajectory order, recurrence, pass count, and learning rate change return behavior under the registered CIFAR-10 protocol

generalization to standard CIFAR-C corruptions, CIFAR-100-C, natural shifts, or a universal mechanism

with five independent source-training seeds, exact two-sided sign-flip tests cannot attain p < 0.05; effect sizes and uncertainty intervals carry the interpretation
