import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_expansion_results", ROOT / "analyze_expansion_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpansionAnalysisTests(unittest.TestCase):
    def test_known_runner_label_alias_is_explicitly_audited(self):
        analysis = load_module()
        audit = analysis.audit_evidence_labels(
            {"FORMAL_EXPANSION"}, "FORMAL_EXPANSION_CORRECTED"
        )
        self.assertTrue(audit["alias_applied"])
        self.assertEqual(audit["status"], "CONDITIONAL_METADATA_ALIAS")
        self.assertEqual(audit["observed_labels"], ["FORMAL_EXPANSION"])

    def test_unknown_or_mixed_evidence_labels_are_rejected(self):
        analysis = load_module()
        with self.assertRaisesRegex(ValueError, "evidence label mismatch"):
            analysis.audit_evidence_labels({"FORMAL_EXPANSION", "UNREGISTERED"}, "FORMAL_EXPANSION_CORRECTED")

    def test_figure_export_uses_a_headless_backend(self):
        analysis = load_module()
        import matplotlib

        matplotlib.use("TkAgg", force=True)
        methods = ["source", "tent", "anchor", "ema_restore", "periodic_reset"]
        rows = [
            {
                "method": method,
                "seed": seed,
                "shift_family": "noise",
                "absolute_hysteresis": 0.01 * (seed + 1),
                "update_induced_hysteresis": 0.005 * (seed + 1),
                "target_accuracy": 0.5 + 0.01 * seed,
            }
            for method in methods
            for seed in (0, 1)
        ]
        mechanism = {"held_observed": [0.1, 0.2], "held_predicted_full": [0.11, 0.19]}
        with tempfile.TemporaryDirectory() as td:
            figures = analysis.make_figures(rows, {"mechanism_screen": mechanism}, Path(td), force=False)
        self.assertEqual(len(figures), 4)
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        analysis = load_module()
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_exact_sign_flip_uses_seed_level_differences(self):
        analysis = load_module()
        result = analysis.exact_sign_flip([1, 2, 3, 4, 5])
        self.assertEqual(result["n_seeds"], 5)
        self.assertEqual(result["permutations"], 32)
        self.assertAlmostEqual(result["p_value_two_sided"], 0.0625)

    def test_cluster_bootstrap_is_reproducible(self):
        analysis = load_module()
        seed_values = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
        first = analysis.cluster_bootstrap_ci(seed_values, draws=500, rng_seed=41)
        second = analysis.cluster_bootstrap_ci(seed_values, draws=500, rng_seed=41)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.3)
        self.assertEqual(first["n_seeds"], 5)

    def test_loader_refuses_an_incomplete_campaign(self):
        analysis = load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "status.json"
            manifest = root / "manifest.json"
            results = root / "results"
            results.mkdir()
            status.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "campaign is not complete"):
                analysis.load_verified_results(status, manifest, results, project_root=root)


if __name__ == "__main__":
    unittest.main()
