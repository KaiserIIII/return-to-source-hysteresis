# Return-to-Source Hysteresis in Test-Time Adaptation

[简体中文](README_zh.md)

A research artifact examining how a model behaves on its first clean, source-like return after an unlabeled test-time adaptation path. The study separates normalization and configuration effects from the incremental contribution of parameter updates, with method-stability analysis and CIFAR-C external validation.

## Research scope

- Measure return-to-source behavior across adaptation paths and methods.
- Separate configuration effects from update-induced changes.
- Compare baseline behavior using pinned upstream provenance.
- Preserve negative results and failed campaigns.

The prospective mechanism screen did not support the tested mechanism. Conclusions are bounded by the recorded models, datasets, protocols, and configurations.

## Repository map

| Location | Contents |
| --- | --- |
| [configs/](configs/) | Frozen campaigns, dataset provenance, and method contracts |
| [results/analysis/](results/analysis/) | Derived results, row tables, and artifact manifests |
| [paper/](paper/) | Manuscript, bibliography, and figures |
| [docs/](docs/) | Implementation, integrity, claims, and review audits |
| [tests/](tests/) | Analysis and runtime regression checks |

The root contains experiment runners, supervisors, and analysis scripts. Raw datasets and model checkpoints must be obtained separately from the sources recorded in the manifests.

## Environment and tests

The recorded analyses used Python 3.13. In PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt -r requirements-analysis-lock.txt -r requirements-dev-lock.txt
.\.venv\Scripts\python -m pytest -q
```

For the recorded CUDA 12.8 stack, replace `requirements-lock.txt` with `requirements-cuda-lock.txt`.

Tests do not start formal campaigns. Full experiments additionally require the referenced datasets, source checkpoints, compute resources, and campaign-admission records.

## Provenance

Modern baseline implementations were independently reimplemented and behavior-audited against pinned upstream references. Source commits, file hashes, and licenses are recorded in [modern_baseline_upstreams.json](configs/modern_baseline_upstreams.json).

The manuscript is a research draft. Claim-traceability checks and submission-readiness checks are recorded separately in [docs/](docs/).

## Citation and license

Use [CITATION.cff](CITATION.cff). Original code and technical documentation use [MIT](LICENSE). Manuscript, figures, tables, and derived scientific results follow [SCIENTIFIC_ARTIFACT_NOTICE.md](SCIENTIFIC_ARTIFACT_NOTICE.md); third-party terms are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
