# Statistical precision and replication rationale

**Date:** 2026-09-20  
**Primary replication unit:** source-training seed  
**Independent seeds per registered comparison:** five (`0` through `4`)

## Exact-test resolution

Each paired contrast uses the five seed-level differences and enumerates all `2^5 = 32` sign assignments. A two-sided exact sign-flip test therefore has a smallest attainable unadjusted p-value of `2/32 = 0.0625` under the conventional doubled-tail definition. Holm correction cannot create a p-value below `0.05` when the unadjusted resolution cannot reach that threshold.

This resolution is a design boundary, not a failed analysis. The study does not use a corrected p-value below `0.05` as a prerequisite for the main descriptive endpoint. It reports the seed-level effect, a seed-cluster percentile interval, the exact p-value, and the registered multiplicity procedure together. Claims about small paired differences remain descriptive.

## What five seeds can support

- The primary estimand is a seed-aggregated effect under a frozen protocol. The seed is the independent replication unit; rows within a seed do not increase the inferential sample size.
- Large, protocol-specific contrasts such as the EATA instability signal are visible in the effect estimate and interval, but they still do not justify a universal ranking.
- Small contrasts among TENT, SAR, CoTTA, RoTTA, and the reversible controls remain precision-limited. The manuscript does not describe them as confirmatory superiority findings.
- Bootstrap intervals use the five seed means to display uncertainty. They are descriptive intervals and do not replace additional independent replications.

## Power and scope decision

The paper does not perform a post-hoc power calculation or treat a non-significant exact test as evidence of equivalence. The precision boundary is stated in Methods and Threats, and the claim set is restricted to effects whose direction and scale are visible under the registered protocol. A future confirmatory study that needs sub-0.05 two-sided sign-flip resolution must register at least six independent seeds and justify the target effect before data collection.

`RB-04 = CLOSED_WITH_PRECISION_BOUNDARY` for this manuscript. No raw result is changed and no additional campaign is required for the bounded characterization claim.
