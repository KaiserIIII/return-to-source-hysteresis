# Normalization Ablation v1 Analysis

Generated: `2026-09-20T11:57:46.515375+00:00`

The 120 raw shards passed hash, provenance, temporal, mutation, metric-range, and source-immutability audits.
Inference uses source-training seed as the independent replication unit; rows are descriptive only.

## Policy comparability

| Policy | Source accuracy | Drop vs source_eval | Primary comparable |
|---|---:|---:|:---:|
| bn_affine_frozen_stats | 0.8638 | 0.0000 | yes |
| bn_affine_running_stats | 0.8638 | 0.0000 | yes |
| groupnorm_checkpoint | 0.1002 | 0.7637 | no |
| layernorm_checkpoint | 0.1029 | 0.7609 | no |
| source_eval | 0.8638 | 0.0000 | yes |
| train_eval_policy_variant | 0.8638 | 0.0000 | yes |

## TENT paired contrasts vs source_eval

| Policy | Metric | Mean difference | 95% CI | dz | exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| bn_affine_frozen_stats | absolute_hysteresis | 0.00611 | [0.00388, 0.00761] | 2.461 | 0.06250 | 0.56250 |
| bn_affine_frozen_stats | update_induced_hysteresis | 0.00611 | [0.00388, 0.00761] | 2.461 | 0.06250 | 0.56250 |
| bn_affine_frozen_stats | target_accuracy | -0.01339 | [-0.01973, -0.00557] | -1.477 | 0.12500 | 0.56250 |
| bn_affine_running_stats | absolute_hysteresis | 0.06110 | [0.05631, 0.06540] | 10.263 | 0.06250 | 0.56250 |
| bn_affine_running_stats | update_induced_hysteresis | 0.06110 | [0.05660, 0.06543] | 10.263 | 0.06250 | 0.56250 |
| bn_affine_running_stats | target_accuracy | 0.12200 | [0.11694, 0.12726] | 17.863 | 0.06250 | 0.56250 |
| train_eval_policy_variant | absolute_hysteresis | 0.00611 | [0.00388, 0.00761] | 2.461 | 0.06250 | 0.56250 |
| train_eval_policy_variant | update_induced_hysteresis | 0.00611 | [0.00388, 0.00761] | 2.461 | 0.06250 | 0.56250 |
| train_eval_policy_variant | target_accuracy | -0.01339 | [-0.01973, -0.00534] | -1.477 | 0.12500 | 0.56250 |
| groupnorm_checkpoint | absolute_hysteresis | 0.00018 | [0.00000, 0.00042] | 0.671 | 0.50000 | — |
| groupnorm_checkpoint | update_induced_hysteresis | 0.00018 | [0.00000, 0.00042] | 0.671 | 0.50000 | — |
| groupnorm_checkpoint | target_accuracy | -0.53554 | [-0.54022, -0.53198] | -97.441 | 0.06250 | — |
| layernorm_checkpoint | absolute_hysteresis | 0.00292 | [0.00000, 0.00876] | 0.447 | 1.00000 | — |
| layernorm_checkpoint | update_induced_hysteresis | 0.00292 | [0.00000, 0.00876] | 0.447 | 1.00000 | — |
| layernorm_checkpoint | target_accuracy | -0.53554 | [-0.54022, -0.53198] | -97.441 | 0.06250 | — |

## Claim boundary

normalization policy materially changes observed return-to-source hysteresis in this CIFAR-10 noise protocol

a universal causal decomposition into configuration gap and recurrent updates; this campaign sets configuration_gap to zero within each policy checkpoint

GroupNorm and LayerNorm checkpoints collapse source-domain accuracy and are exploratory architecture controls, not fair primary comparisons
