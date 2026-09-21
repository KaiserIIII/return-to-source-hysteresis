# Third-party notices

This repository records provenance for external baseline references and datasets. It does not redistribute their source code.

## Baseline references

See `configs/modern_baseline_upstreams.json` for the exact upstream repository, commit, file hashes, and declared license for EATA, SAR, CoTTA, and RoTTA. Each project remains governed by its own license and copyright notices.

## Datasets

Dataset URLs, identifiers, checksums, and declared licenses are recorded in `configs/cifar_dataset_provenance_v2.json`, `configs/cifar_c_dataset_provenance_v1.json`, and `configs/external_cifar_c_sources.json`. Users must follow each dataset provider's current terms.

## Vendored materials

No private Research OS skills, private provider contents, virtual environments, checkpoints, raw datasets, or unreviewed third-party source trees are included in this release.

## CI actions

The public CI workflow pins `actions/checkout` v4.2.2 at commit `11bd71901bbe5b1630ceea73d27597364c9af683` and `actions/setup-python` v5.6.0 at commit `a26af69be951a213d495a4c3e4e4022e16d87065`. No action is referenced by a floating tag.
