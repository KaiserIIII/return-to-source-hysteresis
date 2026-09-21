import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_admission():
    spec = importlib.util.spec_from_file_location(
        "formal_campaign_admission", ROOT / "formal_campaign_admission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FormalCampaignAdmissionTests(unittest.TestCase):
    def test_checkpoint_inventory_is_frozen_from_the_full_manifest_matrix(self):
        admission = load_admission()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints" / "expansion" / "formal"
            checkpoint_root.mkdir(parents=True)
            manifest = {"datasets": ["alpha", "beta"], "seeds": [0, 1]}
            expected = {}
            for dataset in manifest["datasets"]:
                for seed in manifest["seeds"]:
                    path = checkpoint_root / f"{dataset}_s{seed:02d}.pt"
                    path.write_bytes(f"{dataset}-{seed}".encode("ascii"))
                    expected[f"{dataset}:s{seed:02d}"] = admission.sha256(path)

            inventory = admission.freeze_checkpoint_inventory(root, manifest)
            (checkpoint_root / "beta_s01.pt").unlink()
            missing = admission.freeze_checkpoint_inventory(root, manifest)

        self.assertEqual(inventory["status"], "PASS")
        self.assertEqual(inventory["checkpoint_sha256s"], expected)
        self.assertEqual(missing["status"], "FAIL")
        self.assertTrue(any("beta_s01.pt" in item for item in missing["findings"]), missing)

    def test_semantic_canary_covers_every_method_trajectory_and_two_seeds(self):
        admission = load_admission()
        contracts = admission.load_json(ROOT / "configs" / "tta_method_state_contracts.json")
        result = admission.run_semantic_canary(
            ROOT / "expansion_experiment.py",
            contracts,
            seeds=(901, 902),
            sample_count=8,
        )

        self.assertEqual(result["status"], "PASS", result["findings"])
        self.assertEqual(result["case_count"], 20)
        self.assertEqual(
            {case["method"] for case in result["cases"]},
            {"source", "tent", "anchor", "ema_restore", "periodic_reset"},
        )
        self.assertEqual({case["trajectory"] for case in result["cases"]}, {"A-B-A", "A-B-C-A"})
        self.assertEqual({case["seed"] for case in result["cases"]}, {901, 902})
        source_cases = [case for case in result["cases"] if case["method"] == "source"]
        self.assertTrue(source_cases)
        for case in source_cases:
            self.assertEqual(case["state_dict_before_sha256"], case["state_dict_after_sha256"])
            self.assertEqual(case["bn_buffers_before_sha256"], case["bn_buffers_after_sha256"])
            self.assertEqual(case["optimizer_step_count"], 0)
            self.assertEqual(case["trainable_parameter_delta"], 0.0)
            self.assertFalse(case["entered_mutating_train_mode"])

    def test_future_information_leakage_fails_temporal_gate(self):
        admission = load_admission()
        shard = {
            "episode": {
                "prospective_mechanism": {"measurement_step": 12},
                "return_outcome": {"measurement_step": 12},
                "target_stream_evaluation": {"timing": "pre_update_online_predictions"},
            }
        }
        findings = admission.temporal_findings(shard, "leaked.json")
        self.assertTrue(any("measurement_time < outcome_time" in item for item in findings), findings)

    def test_scale_out_refuses_missing_semantic_or_provenance_evidence(self):
        admission = load_admission()
        incomplete = {
            "semantic_canary": {"status": "NOT_RUN"},
            "temporal_availability": {"status": "PASS"},
            "provenance_binding": {"status": "PASS"},
            "early_sanity_audit": {"status": "PASS"},
        }
        self.assertEqual(admission.scale_out_decision(incomplete)["status"], "FORMAL_ADMISSION_FAIL")

        complete = {
            "semantic_canary": {"status": "PASS"},
            "temporal_availability": {"status": "PASS"},
            "provenance_binding": {"status": "PASS"},
            "early_sanity_audit": {"status": "PASS"},
        }
        self.assertEqual(admission.scale_out_decision(complete)["status"], "FORMAL_ADMISSION_PASS")

    def test_v1_failure_is_a_permanent_integrity_regression_fixture(self):
        admission = load_admission()
        fixture = admission.load_json(ROOT / "configs" / "formal_v1_integrity_regression.json")
        self.assertEqual(fixture["disposition"], "INVALID_FOR_PRIMARY_EVIDENCE")
        self.assertEqual(fixture["observed_jobs"], 1000)
        self.assertEqual(
            set(fixture["reason_codes"]),
            {
                "SOURCE_BN_STATE_CONTAMINATION",
                "DRIFT_TEMPORAL_MISLABELING",
                "INCOMPLETE_RUNNER_DATASET_PROVENANCE",
            },
        )
        self.assertEqual(admission.validate_v1_fixture(fixture), [])

    def test_provenance_binding_detects_a_stale_result_hash(self):
        admission = load_admission()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "job.json"
            shard.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            status = {
                "completed": {"job": "sha256:" + "0" * 64},
                "runner_sha256": "sha256:" + "1" * 64,
                "supervisor_sha256": "sha256:" + "2" * 64,
                "manifest_sha256": "sha256:" + "3" * 64,
                "dataset_provenance_sha256": "sha256:" + "4" * 64,
            }
            result = admission.audit_shard_bindings(
                status,
                root,
                expected={
                    "runner_sha256": "sha256:" + "1" * 64,
                    "supervisor_sha256": "sha256:" + "2" * 64,
                    "manifest_sha256": "sha256:" + "3" * 64,
                    "dataset_provenance_sha256": "sha256:" + "4" * 64,
                },
                manifest={"batch_size": 128, "adaptation_passes": 1, "adaptation_learning_rate": 0.001},
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("result hash mismatch" in item for item in result["findings"]), result)

    def test_durable_launcher_rechecks_admission_before_resuming(self):
        launcher = (ROOT / "run_expansion_v2_task.ps1").read_text(encoding="utf-8-sig")
        admission_call = launcher.index("formal_campaign_admission.py")
        supervisor_call = launcher.index("run_expansion_supervisor.py")
        self.assertLess(admission_call, supervisor_call)
        self.assertIn("check-admission", launcher)
        self.assertIn("formal_v2_admission.json", launcher)


if __name__ == "__main__":
    unittest.main()
