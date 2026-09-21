# EATA implementation audit

**Date:** 2026-09-20  
**Scope:** `modern_baselines_v1` and the shared independent reimplementation in `publication_expansion_experiment.py`  
**Pinned behavior reference:** `configs/modern_baseline_upstreams.json`, EATA commit `f739b3668cc7617e9b9f1979c1a358497a3472c3` (MIT)

## Audit question

The audit asks whether the local EATA branch implements the same semantic components as the pinned reference and whether the reported result can be interpreted as an exact upstream reproduction. The answer is **semantic components match; hyperparameters and optimizer policy differ; the result is an independent reimplementation under a matched CIFAR protocol**.

## Component comparison

| Component | Pinned reference | Local implementation | Assessment |
|---|---|---|---|
| Adapted parameters | Batch-normalization affine parameters | `_adaptable()` selects affine parameters from batch/group/layer normalization modules | Matches the BN case; the local selector is broader for other policies |
| Reliable-sample filter | Entropy threshold `e_margin` | Per-batch entropy threshold `max(0.35, 0.85 log C)` | Semantic match; threshold is protocol-specific and differs numerically |
| Non-redundancy filter | Cosine-similarity threshold `d_margin=0.05` | Cosine similarity magnitude threshold `abs(similarity) < 0.999` | Same filter role; threshold and comparison rule are different |
| Entropy weighting | `1 / exp(H - e_margin)` | Uniform mean over selected entropy values | Local branch omits the upstream reliability reweighting |
| Fisher estimate | Pseudo-label cross-entropy on a configured source subset, with upstream sample-size control | Pseudo-label cross-entropy on all episode source examples | Same diagonal-Fisher idea; sample policy differs |
| Fisher regularizer | `fisher_alpha=2000` by default | `50.0` in the local branch | Numerically different; this is a registered protocol adaptation |
| Optimizer | SGD, upstream example uses `lr=0.00025`, momentum `0.9` | Adam for EATA with the campaign learning-rate field | Numerically different; the result cannot be called a reference-recipe reproduction |
| Checkpointing | Copies model and optimizer state for episodic reset | Uses a copied source model per episode; records model and optimizer digests | Equivalent for the non-episodic recurrent protocol |
| Update count | One optimizer step when selected samples exist | One optimizer step per processed batch, including empty-selection zero-loss steps | The local counter is explicit and recorded in every shard |
| Label use | No target labels in adaptation | Labels score predictions after the adaptation loss and never enter the loss | Matches the no-label contract |

## Evidence and interpretation

The implementation audit finds no hidden label path, future-outcome path, or unrecorded optimizer state. The differences above are load-bearing implementation choices, so the EATA rows remain valid evidence about the registered local algorithm and protocol. They do not establish a universal statement about every EATA implementation, and they do not reproduce the original ImageNet-C recipe.

The manuscript therefore keeps the EATA result as an **implementation-sensitivity signal**. It removes any wording that could be read as a universal EATA failure or as a direct reproduction of the upstream leaderboard. A new campaign is unnecessary for the current claim because the paper does not use EATA to support a universal ranking; a future exact-recipe replication would receive a new campaign identity.

## Source anchors

- Pinned reference: `literature/upstream_fixed/eata__eata.py:13-126` and `literature/upstream_fixed/eata__main.py:65-146`.
- Local Fisher and EATA branch: `publication_expansion_experiment.py:215-239` and `publication_expansion_experiment.py:410-444`.
- Formal result provenance and implementation status: `results/analysis/modern_baselines_v1/analysis.json` (`INDEPENDENT_REIMPLEMENTATION`, pinned registry hash, 60 observed jobs).

## Gate decision

`RB-02 = CLOSED_WITH_SCOPE_RESTRICTION`.

The result remains in the manuscript, but only with the independent-reimplementation and implementation-sensitivity qualification. The raw shards and their provenance are unchanged.
