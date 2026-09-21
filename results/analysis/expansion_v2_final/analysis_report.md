# Corrected Expansion Analysis

Generated: `2026-09-09T04:22:03.791115+00:00`

The analysis passed structural, hash, dataset, checkpoint, protocol, timing, and source-invariant checks before computing statistics.

## Method summaries

| Method | Total return loss (95% CI) | Update-induced loss (95% CI) | Target accuracy (95% CI) |
|---|---:|---:|---:|
| anchor | 0.0061 [0.0053, 0.0071] | -0.0003 [-0.0006, -0.0000] | 0.5651 [0.5631, 0.5672] |
| ema_restore | 0.0062 [0.0054, 0.0071] | -0.0002 [-0.0004, -0.0000] | 0.5651 [0.5630, 0.5671] |
| periodic_reset | 0.0064 [0.0057, 0.0072] | -0.0000 [-0.0001, 0.0000] | 0.5646 [0.5626, 0.5666] |
| source | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.4796 [0.4780, 0.4809] |
| tent | 0.0061 [0.0053, 0.0071] | -0.0003 [-0.0006, -0.0000] | 0.5651 [0.5631, 0.5672] |

## Mechanism boundary

Status: `NOT_SUPPORTED`. No mechanism claim is permitted from one held seed; execute fresh held seeds 5-9.

## Reporting policy

All null and negative results remain in `analysis.json`. Row-wise p-values in the held-seed screen are descriptive because repeated conditions within a seed are dependent. Formal contrasts use seed-level pairing and Holm correction.
