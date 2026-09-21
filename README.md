# Return-to-Source Hysteresis in Test-Time Adaptation

This repository contains the public, reproducibility-oriented artifact for:

**Return-to-Source Hysteresis in Test-Time Adaptation: Configuration Effects, Method Stability, and CIFAR-C External Validation**

The study measures the first clean source-like return after an unlabeled adaptation path. It separates normalization/configuration effects from the incremental update-induced component and reports a formally negative prospective mechanism screen.

## Public release boundary

This repository intentionally excludes raw datasets, model checkpoints, per-job raw outputs, local run logs, virtual environments, private skills, author contact details, author-only submission files, and machine-specific paths. Download the public CIFAR and CIFAR-C datasets from their cited sources and verify them against the manifests in `configs/`.

The published PDF and TeX source remain anonymous review copies. The repository itself is an authored public artifact: author attribution appears only in `CITATION.cff`, `LICENSE`, and the GitHub account metadata. No email address, ORCID, affiliation, cover letter, title page, or submission credential is included.

## Evidence included

- Frozen campaign manifests and dataset provenance records.
- Analysis summaries, derived row tables, and artifact manifests for the formal and expansion campaigns.
- Figure files used by the manuscript.
- Reproducibility and integrity audit notes, including negative and failed-campaign boundaries.
- Source code and tests for the local analysis and experiment runners.

The analysis summaries are evidence records, not a claim that every campaign can run without downloading the referenced public datasets and installing the locked environment.

## Reproduction

Python 3.13 was used for the recorded analyses. Create a clean environment and install one runtime lock plus the analysis and test locks:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt -r requirements-analysis-lock.txt -r requirements-dev-lock.txt
.\.venv\Scripts\python -m pytest -q
```

Use `requirements-cuda-lock.txt` in place of `requirements-lock.txt` for the recorded CUDA 12.8 stack. The formal campaigns require the public CIFAR data, source checkpoints, substantial compute, and the manifests under `configs/`; they are not started by the test suite. `run_expansion_v2_task.ps1` refuses to resume a campaign until its persisted formal-admission record passes.

## Baseline provenance

Modern baseline implementations were independently reimplemented and behavior-audited against pinned upstream references. The repositories, exact commits, file hashes, and licenses are recorded in `configs/modern_baseline_upstreams.json`. Their source code is not redistributed here.

## Scientific status

The local publication gate was `PUBLICATION_SUFFICIENCY = FAIL` at release preparation time because the claims audit was conditional and venue-specific submission checks remained open. The repository preserves these limits and does not claim acceptance, a JCR quartile, or a universal leaderboard.

## License

Original source code and technical documentation are released under the MIT License. The manuscript, figures, tables, and derived scientific result artifacts remain copyright (c) 2026 Yue Yu, all rights reserved pending publication. See `SCIENTIFIC_ARTIFACT_NOTICE.md`; third-party references and datasets remain under their upstream terms in `THIRD_PARTY_NOTICES.md`.

## Citation

See `CITATION.cff` and `paper/references.bib`.
