# Independent Review Board

Date: 2026-09-21  
Manuscript: `paper/main.tex`  
Evidence snapshot: claims audit `2026-09-21-claims-audit.md`, formal_v2 and four expansion analysis artifacts

This is a local, evidence-bound review by separate review lenses. It is not an acceptance prediction and it does not replace a venue editor or a specialist review of the raw implementation.

## Methods reviewer

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| HIGH | Methods 4.7; Results 5.7 | The modern-baseline comparison uses a shared fixed-statistics protocol and independent reimplementations rather than each paper's complete recipe. | `results/analysis/modern_baselines_v1/analysis.json`; `docs/eata_implementation_audit_2026-09-20.md` | Keep the protocol-difference table and call the results independent reimplementation evidence throughout. | CLOSED_WITH_SCOPE_RESTRICTION | A reviewer can still request method-specific recipe replications. |
| MEDIUM | Methods 4.5; Appendix B | The corrected formal runner retains a frozen metadata-label discrepancy. | `results/analysis/expansion_v2_final/analysis.json` (`PASS_WITH_METADATA_ALIAS`) | Expose the alias and state that raw shards were not rewritten. | CLOSED | The label discrepancy may require a reviewer explanation. |

## Statistics reviewer

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| HIGH | Methods 4.6; Limitations | Five source-training seeds limit exact sign-flip resolution and precision for small contrasts. | `docs/statistical_precision_plan_2026-09-20.md`; all expansion `analysis.json` files | Treat seed as the independent replication unit, report paired effects and intervals, and avoid equivalence or winner claims. | CLOSED_WITH_PRECISION_BOUNDARY | Additional seeds could change small contrasts. |
| MEDIUM | Results 5.9 | The trajectory campaign contains many nuisance-factor cells and is confirmatory only for the registered CIFAR-10 frozen-statistics boundary. | `results/analysis/trajectory_ablation_gpu_v1/analysis.json` (`claim_boundary`) | Keep the registered contrast family and state the generalization boundary. | CLOSED | Natural-shift generalization remains untested. |

## TTA domain reviewer

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| CRITICAL | Results 5.6; Discussion | The prospective mechanism predictor fails on the held seed and must not be presented as an explanation. | `docs/mechanism_question_closure_2026-09-20.md`; Figure 4 | Retain the negative result and formally abandon the predictor claim. | CLOSED_BY_ABANDONMENT | A future mechanism study would need fresh seeds and a new preregistered campaign. |
| HIGH | Results 5.7 and 5.10 | EATA's extreme instability may be implementation-sensitive and cannot support a universal ranking. | `docs/eata_implementation_audit_2026-09-20.md`; external analysis artifact | Keep the result as an implementation-sensitive signal and report the shared-protocol boundary. | CLOSED_WITH_SCOPE_RESTRICTION | Paper-specific hyperparameters may yield different values. |
| MEDIUM | Limitations | The evidence is limited to CIFAR-family corruption and one ResNet family. | `paper/main.tex:463-468` | State the deployment and architecture boundary explicitly. | CLOSED_FOR_CURRENT_CLAIM | Broader claims remain unsupported. |

## Reproducibility and integrity reviewer

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| HIGH | Data/code availability; Appendix C | The public archive excludes raw data, checkpoints, per-job shards, and logs, so an external reader cannot rerun every formal job from the repository alone. | `README.md`; `configs/cifar_dataset_provenance_v2.json`; public repository | Keep the exclusion explicit, publish code, manifests, derived rows, figures, locks, and dataset hashes, and retain the complete private archive. | CLOSED_WITH_ACCESS_BOUNDARY | Controlled data/checkpoint access may still be requested by a venue. |
| MEDIUM | Reproducibility 8 | Some analysis JSON files contain host-specific provenance paths in the private archive. | claims audit and public scrub check | Remove host paths from the public artifact while preserving hashes and relative artifact identities. | CLOSED_IN_PUBLIC_RELEASE | Private provenance remains local-only. |

## Novelty and closest-work reviewer

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| HIGH | Related work 2; novelty matrix | The exact-region search establishes bounded distinction against retrieved full text, not universal absence of prior work. | `docs/closest_work_exact_region_audit_2026-09-20.md`; `docs/novelty_matrix.md` | Use bounded positioning language and avoid absolute first/only claims. | CLOSED_WITH_BOUNDED_POSITIONING | A newly published or unindexed paper could narrow novelty. |

## Adversarial editor

| Severity | Anchor | Finding | Evidence anchor | Smallest defensible fix | Status | Residual risk |
|---|---|---|---|---|---|---|
| HIGH | Submission package | The scientific package is internally coherent, but venue-specific JCR status, article type, live fee fields, and author-controlled declarations are not verifiable from the local evidence. | `docs/venue_preflight_nca.md`; `docs/publication_editor_gate_2026-09-21.md` | Verify the live venue profile and complete the author-controlled submission checklist before upload. | OPEN | Final editor decision remains external. |
| MEDIUM | Full manuscript | The paper is a bounded measurement and characterization study rather than a new adaptation algorithm. | Abstract, Discussion, Conclusion | Keep the contribution framed as an estimand, decomposition, and audited protocol boundary; do not oversell method superiority. | CLOSED | Venue fit may still be judged weak by a particular editor. |

## Board decision

Scientific MUST_HAVE findings are closed by explicit scope restrictions, retained negative results, and evidence anchors. The submission gate remains `PUBLICATION_SUFFICIENCY = FAIL` until the live venue profile and author-controlled package declarations are completed. `READY_TO_SUBMIT = NO`.

