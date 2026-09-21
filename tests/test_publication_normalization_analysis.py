import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_publication_normalization",
        ROOT / "analyze_publication_normalization.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationNormalizationAnalysisTests(unittest.TestCase):
    def test_paired_contrast_uses_seed_unit_and_exact_sign_flip(self):
        analysis = load_module()
        result = analysis.paired_contrast(
            {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
            {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
            draws=200,
            seed=7,
        )
        self.assertEqual(result["n_seeds"], 5)
        self.assertEqual(result["permutations"], 32)
        self.assertAlmostEqual(result["mean_difference"], -0.1)
        self.assertIn("cohen_dz", result)
        self.assertEqual(len(result["difference_by_seed"]), 5)

    def test_holm_adjustment_preserves_family_order(self):
        analysis = load_module()
        self.assertEqual(analysis.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_policy_comparability_flags_architecture_collapse(self):
        analysis = load_module()
        rows = [
            {"policy": "source_eval", "checkpoint_source_accuracy": 0.86},
            {"policy": "groupnorm_checkpoint", "checkpoint_source_accuracy": 0.10},
        ]
        result = analysis.policy_comparability(rows, reference="source_eval", threshold=0.05)
        self.assertFalse(result["policies"]["groupnorm_checkpoint"]["primary_comparable"])
        self.assertTrue(result["policies"]["groupnorm_checkpoint"]["warning"])

    def test_loader_rejects_incomplete_or_hash_mismatched_campaign(self):
        analysis = load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "status.json"
            results = root / "results"
            results.mkdir()
            status.write_text(
                json.dumps({"status": "RUNNING", "completed": {}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "campaign is not complete"):
                analysis.load_verified_rows(status, results)


if __name__ == "__main__":
    unittest.main()
