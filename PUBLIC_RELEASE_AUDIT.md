# Public release audit

Date: 2026-09-21

## Excluded from the public repository

- Raw CIFAR and CIFAR-C data.
- Model checkpoints and per-job raw result shards.
- Virtual environments, package caches, and generated logs.
- Private Research OS skills and local provider paths.
- Author email, ORCID, affiliation, cover letter, title page, and submission credentials. Author attribution is intentionally limited to `CITATION.cff`, `LICENSE`, and GitHub account metadata.
- Formal run-state files containing machine-specific paths or process identifiers.

## Included

- Source code, tests, locked dependency lists, frozen public manifests, analysis summaries, derived row tables, figures, and anonymous manuscript materials.
- Upstream baseline repository URLs, exact commits, file hashes, and licenses.
- Dataset provenance and checksum records without redistributing the data.

## Claim boundary

The release preserves the `CONDITIONAL` claims-audit status and the negative mechanism result. It does not claim publication readiness, acceptance probability, JCR quartile, or a universal method ranking.

## Verification commands

```text
pytest -q
python -m compileall -q .
```

The full formal campaigns require separately downloaded public datasets and are not launched by these checks.
