from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_publication_trajectory",
        ROOT / "analyze_publication_trajectory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationTrajectoryAnalysisTests(unittest.TestCase):
    def test_factor_contrast_averages_nuisance_cells_within_seed(self):
        analysis = load_module()
        rows = []
        for seed in (0, 1):
            for lr in (0.001, 0.01):
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "method": "tent",
                            "trajectory": "A-B-A",
                            "pass_count": 1,
                            "learning_rate": lr,
                            "absolute_hysteresis": float(seed + lr),
                        },
                        {
                            "seed": seed,
                            "method": "tent",
                            "trajectory": "A-B-C-A",
                            "pass_count": 1,
                            "learning_rate": lr,
                            "absolute_hysteresis": float(seed + lr + 0.2),
                        },
                    ]
                )
        result = analysis.factor_contrast(
            rows,
            method="tent",
            metric="absolute_hysteresis",
            factor="trajectory",
            reference="A-B-A",
            alternative="A-B-C-A",
            draws=200,
            seed=7,
        )
        self.assertEqual(result["n_seeds"], 2)
        self.assertEqual(result["difference_by_seed"], {"0": 0.2, "1": 0.2})
        self.assertAlmostEqual(result["mean_difference"], 0.2)
        self.assertEqual(result["permutations"], 4)

    def test_design_coverage_rejects_missing_factorial_cell(self):
        analysis = load_module()
        rows = [
            {
                "seed": 0,
                "method": "tent",
                "trajectory": "A-B-A",
                "pass_count": 1,
                "learning_rate": 0.001,
            }
        ]
        result = analysis.design_coverage(
            rows,
            methods=["tent"],
            trajectories=["A-B-A", "A-B-C-A"],
            pass_counts=[1],
            learning_rates=[0.001],
            seeds=[0],
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["expected_cells"], 2)
        self.assertEqual(result["observed_cells"], 1)
        self.assertEqual(len(result["missing_cells"]), 1)

    def test_holm_adjustment_is_monotone_in_rank_order(self):
        analysis = load_module()
        self.assertEqual(analysis.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_loader_rejects_incomplete_campaign(self):
        analysis = load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "status.json"
            results = root / "results"
            results.mkdir()
            status.write_text(
                json.dumps({"status": "RUNNING", "completed": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "campaign is not complete"):
                analysis.load_verified_rows(status, results)


if __name__ == "__main__":
    unittest.main()
