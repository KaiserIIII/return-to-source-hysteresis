from __future__ import annotations

import external_validation_analysis as analysis


def _rows() -> list[dict]:
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "dataset": "cifar10_c",
                    "method": "source",
                    "seed": seed,
                    "severity": 1,
                    "corruption": "gaussian_noise",
                    "habs": 0.0,
                },
                {
                    "dataset": "cifar10_c",
                    "method": "tent",
                    "seed": seed,
                    "severity": 1,
                    "corruption": "gaussian_noise",
                    "habs": 0.01 + seed * 0.001,
                },
            ]
        )
    return rows


def test_exact_sign_flip_uses_all_two_to_the_n_assignments() -> None:
    result = analysis.exact_sign_flip([1.0, 1.0, 1.0, 1.0, 1.0])
    assert result["permutations"] == 32
    assert result["p_value_two_sided"] == 0.0625


def test_paired_contrast_aggregates_within_seed_before_inference() -> None:
    result = analysis.paired_contrast(
        _rows(), value="habs", reference="tent", control="source"
    )
    assert result["n_seeds"] == 5
    assert result["mean_difference"] == 0.012
    assert result["seed_differences"] == {
        "0": 0.01,
        "1": 0.011,
        "2": 0.012,
        "3": 0.013,
        "4": 0.014,
    }


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
