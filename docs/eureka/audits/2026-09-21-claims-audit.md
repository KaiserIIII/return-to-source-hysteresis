# Claims Audit Report
**Date:** 2026-09-21
**Manuscript:** `paper/main.tex` (sha256 `9e0ff0d798109fd08cf28c01c12ff9ed7ddbc5733db4a109b18c8b2054b16099`)
**Traceability table:** `docs/eureka/audits/2026-09-21-claim-traceability.csv`
**Auditor:** fresh deterministic local checker; no external service
**Overall status:** PASS

The checker extracted 468 numeric tokens. Every token is classified as evidence, protocol, bibliographic, or metadata and bound to a concrete local file. Load-bearing result tokens are compared at displayed precision against the selected analysis artifact. `CHECK` means that a displayed value is a registered rounded value requiring a source-table spot-check; it is not an unbound claim.

## Token accounting

| Class | Count | Bound source |
|---|---:|---|
| evidence | 322 | YES |
| protocol | 116 | YES |
| bibliographic | 30 | YES |
| metadata | 0 | YES |

## Evidence artifacts

| Package | Path | Exists | SHA-256 |
|---|---|---:|---|
| formal_v2 | `results/analysis/expansion_v2_final/analysis.json` | YES | `2dc8eef90c5ee548892cc9ccef028c47d615f6dc8d6439079c754befa34036d7` |
| modern_baselines | `results/analysis/modern_baselines_v1/analysis.json` | YES | `f4d35370b82f586c8dabe7f11cbe3ac60cdc7158597a21f91ff2516b862139f1` |
| normalization | `results/analysis/normalization_ablation_v1/analysis.json` | YES | `8dbb9b4f0264b3a80677257dbdd8faae041f6b6722d19e2b2e00bc1e5a5aeb31` |
| trajectory | `results/analysis/trajectory_ablation_gpu_v1/analysis.json` | YES | `7ba92f879a7309633be333ba1b2e3368ed6c7855a537d7be381d319723fcb27b` |
| external | `results/analysis/external_validation_cifar_c_v1/analysis.json` | YES | `21e2299309e31ca601adc799397701565b613fa4bfe87cc15412594fe6e5772c` |
| formal_manifest | `configs/expansion_campaign_v2.json` | YES | `875cb278c0d61781c06116cc64ad828a03a50c7a72022ceefb1d587bf556cc29` |
| publication_manifest | `configs/publication_expansion_campaigns.json` | YES | `85e5f8ea01a3246df81352eaa85f94a5ad71fc4eb0c05a66b13aeb4af111b41e` |
| trajectory_manifest | `configs/publication_trajectory_gpu_campaign.json` | YES | `c087475522cc61dad640e83b29c2bd848b599f22a5a73bcb008064f2bd91e92d` |
| dataset_provenance | `configs/cifar_dataset_provenance_v2.json` | YES | `4a909492144b016d792323f19cd239a48b01c5ffc9289e47becb3b87b39f0c9b` |

## Figure regeneration evidence

| Figure | Generator | Output | Artifact manifest |
|---|---|---:|---:|
| expansion_return_loss | `analyze_expansion_results.py` | YES | YES |
| expansion_target_return_pareto | `analyze_expansion_results.py` | YES | YES |
| expansion_shift_method_heatmap | `analyze_expansion_results.py` | YES | YES |
| expansion_mechanism_screen | `analyze_expansion_results.py` | YES | YES |
| external_cifar10_c_method_summary | `external_validation_analysis.py` | YES | YES |
| external_cifar100_c_method_summary | `external_validation_analysis.py` | YES | YES |
| external_cifar_c_shift_method_heatmap | `external_validation_analysis.py` | YES | YES |

The analysis and visualization commands were rerun in this audit session. Artifact manifests bind generated files to the analysis-script hash and input artifacts.

## Completeness and negative-result accounting

| Package | Formal artifact | Negative/failure retained |
|---|---:|---:|
| formal_v2 | YES | YES, including the prior invalid campaign |
| modern_baselines_v1 | YES | YES, including EATA implementation sensitivity |
| normalization_ablation_v1 | YES | YES, including GroupNorm/LayerNorm controls |
| trajectory_ablation_gpu_v1 | YES | YES, including null/order effects |
| external validation | YES | YES, including the failed v1 and 926-job recovery |
| mechanism screen | YES | YES, predictor claim formally abandoned |

## Mechanical checks

| Check | Hits |
|---|---:|
| em_dash | 0 |
| exclamation | 0 |
| hype_verbs | 0 |
| throat_clearers | 0 |

## Conditions and interpretation

Unbound tokens: **0**. Rounded evidence tokens requiring a human spot-check: **158**. Missing artifacts: **0**.

The claims audit is a necessary evidence check, not an acceptance prediction. Publication sufficiency remains a separate gate because venue-specific rules, author declarations, and the editor's scientific judgment cannot be established by a local script.
