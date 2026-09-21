# Closest-work exact-region audit

**Date:** 2026-09-20  
**Scope:** works that could preempt the paper's recurrent return-to-source endpoint, path/order analysis, or configuration/update decomposition.  
**Local evidence rule:** a relation is `EXACT_REGION_VERIFIED` only when a retrieved full-text file contains the relevant method or evaluation passage. Metadata-only records remain background evidence.

| Work | Retrieval | Exact region inspected | Relation to this paper | Preemption |
|---|---|---|---|---|
| Continual Test-Time Domain Adaptation / CoTTA (CVPR 2022) | `FULLTEXT_RETRIEVED` | `literature/continual_2203.13591.txt:65-76` states continual target streams and source preservation; `:239-243` and `:275-297` describe stochastic restoration | Continual adaptation and source forgetting, with restoration. It does not define a clean-return endpoint before return-stage updating or decompose configuration and update terms. | Not found |
| EATA (ICML 2022) | `FULLTEXT_RETRIEVED` | `literature/efficient_2204.02610.txt:95-142` and `:256-281` describe reliable/non-redundant filtering and Fisher regularization | Anti-forgetting TTA method. It does not study recurrent A-B-A/A-B-C-A return measurement or the paper's decomposition. | Not found |
| Back to the Source (2022) | `FULLTEXT_RETRIEVED` | `literature/source_2207.03442.txt:14-33`, `:85-105`, and `:379-392` describe input diffusion, fixed model weights, and order sensitivity | Source-directed input adaptation and order sensitivity. The model remains fixed; the paper does not measure return hysteresis caused by adaptive normalization or recurrent parameter updates. | Not found |
| Cyclic Test-Time Adaptation (2023) | `FULLTEXT_RETRIEVED` | `literature/cyclic_2308.06554.txt:37-48`, `:182-202`, and `:424-432` define task-specific cycles for human mesh reconstruction | Cyclic adaptation in a different task with two interacting networks and generated supervision. It does not evaluate a source-like return endpoint in image classification. | Not found |
| Order-Aware TTA (2026) | `FULLTEXT_RETRIEVED` | `literature/order_aware_2601.21012.txt:18-51`, `:97-116`, `:225-254`, and `:373-407` use temporal transition priors and order-aware prediction | Uses sequence order as a prediction prior across natural temporal datasets. It does not define recurrent domain return, configuration gap, or update-induced hysteresis. | Not found |
| NOTE, SAR, RoTTA, MEMO | `METADATA_ONLY` for exact-region audit in this workspace; citations and pinned code registries are present | `paper/references.bib` and upstream code READMEs identify the methods; full-text relation was not used for the novelty conclusion | Baseline and context works. They are not treated as exact closest-work evidence. | No preemption claim made |

## Positioning decision

The retrieved full text supports a bounded distinction: this paper measures a clean source-like return before return-stage adaptation, separates configuration and update components, and tests the endpoint across recurrent trajectories and fixed-statistics baselines. The audit finds no retrieved work that covers the complete combination. The novelty claim remains limited to this estimand and protocol; it does not claim that no uncollected work exists.

`RB-06 = CLOSED_WITH_BOUNDED_POSITIONING`.
