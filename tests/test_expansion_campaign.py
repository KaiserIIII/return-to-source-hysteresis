import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load("expansion_experiment")
supervisor = load("run_expansion_supervisor")


class ExpansionCampaignTests(unittest.TestCase):
    def test_cuda_determinism_workspace_is_configured_before_formal_runs(self):
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")

    def test_protocol_expands_dataset_model_baseline_mechanism_and_ablation_scope(self):
        value = json.loads((ROOT / "configs" / "expansion_campaign_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(value["evidence_label"], "FORMAL_EXPANSION")
        self.assertEqual(set(value["datasets"]), {"cifar10", "cifar100"})
        self.assertEqual(value["model"]["architecture"], "cifar_resnet18")
        self.assertGreaterEqual(len(value["seeds"]), 5)
        self.assertTrue({"source", "tent", "anchor", "ema_restore", "periodic_reset"}.issubset(value["methods"]))
        self.assertTrue({"learning_rate", "adaptation_passes", "trajectory_length"}.issubset(value["ablations"]))
        self.assertTrue(value["prospective_mechanism"]["measured_before_return_outcome"])
        self.assertFalse(value["labels_allowed_during_adaptation"])

    def test_return_outcome_is_measured_after_prospective_predictors(self):
        torch.manual_seed(7)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(12, 3, 8, 8)
        y = torch.arange(12) % 3
        result = experiment.run_recurrent_episode(
            model,
            x,
            y,
            method="tent",
            trajectory="A-B-A",
            shift_family="noise",
            severity=1,
            seed=7,
            batch_size=4,
            adaptation_passes=1,
        )
        self.assertLess(
            result["prospective_mechanism"]["measurement_step"],
            result["return_outcome"]["measurement_step"],
        )
        self.assertIn("gradient_norm_before_update", result["prospective_mechanism"])
        self.assertIn("prediction_entropy_before_update", result["prospective_mechanism"])
        self.assertFalse(result["labels_used_during_adaptation"])
        self.assertIn("accuracy_before_return_adaptation", result["return_outcome"])
        self.assertIn("target_stream_evaluation", result)
        self.assertGreaterEqual(result["target_stream_evaluation"]["accuracy"], 0.0)
        self.assertLessEqual(result["target_stream_evaluation"]["accuracy"], 1.0)
        self.assertEqual(result["target_stream_evaluation"]["examples"], 12)

    def test_abca_uses_a_fixed_distinct_c_domain(self):
        torch.manual_seed(29)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(12, 3, 8, 8)
        y = torch.arange(12) % 3

        result = experiment.run_recurrent_episode(
            model,
            x,
            y,
            method="source",
            trajectory="A-B-C-A",
            shift_family="contrast",
            severity=1,
            seed=29,
            batch_size=4,
            adaptation_passes=1,
        )

        self.assertEqual(result["shift_sequence"], ["contrast", "channel_permutation"])
        self.assertEqual(result["target_stream_evaluation"]["examples"], 24)

    def test_run_job_records_protocol_parameters_and_dataset_provenance(self):
        torch.manual_seed(31)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(12, 3, 8, 8)
        y = torch.arange(12) % 3
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = Path(td) / "source.pt"
            torch.save(
                {"classes": 3, "model_state": model.state_dict()},
                checkpoint_path,
            )
            args = argparse.Namespace(
                seed=31,
                source_checkpoint=checkpoint_path,
                device="cpu",
                dataset="cifar10",
                method="source",
                trajectory="A-B-A",
                shift_family="noise",
                severity=1,
                batch_size=4,
                adaptation_passes=2,
                learning_rate=0.005,
                max_test_samples=0,
                smoke=False,
                job_id="metadata-test",
                dataset_provenance_sha256="sha256:" + "a" * 64,
            )
            with mock.patch.object(experiment, "cifar_resnet18", return_value=experiment.TinyConvNet(classes=3)), mock.patch.object(
                experiment, "load_cifar", return_value=(x, y)
            ):
                result = experiment.run_job(args)

        self.assertEqual(
            result["protocol_parameters"],
            {"batch_size": 4, "adaptation_passes": 2, "adaptation_learning_rate": 0.005},
        )
        self.assertEqual(result["dataset_provenance_sha256"], "sha256:" + "a" * 64)

    def test_source_baseline_does_not_update_model_state_on_the_stream(self):
        torch.manual_seed(17)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(24, 3, 8, 8)
        y = torch.arange(24) % 3
        snapshots = []
        original_metrics = experiment._metrics

        def recording_metrics(current_model, *args, **kwargs):
            snapshots.append(
                {name: value.detach().clone() for name, value in current_model.state_dict().items()}
            )
            return original_metrics(current_model, *args, **kwargs)

        with mock.patch.object(experiment, "_metrics", side_effect=recording_metrics):
            result = experiment.run_recurrent_episode(
                model,
                x,
                y,
                method="source",
                trajectory="A-B-C-A",
                shift_family="brightness",
                severity=2,
                seed=17,
                batch_size=4,
                adaptation_passes=2,
            )

        self.assertEqual(len(snapshots), 2)
        for name in snapshots[0]:
            torch.testing.assert_close(snapshots[1][name], snapshots[0][name], rtol=0, atol=0)
        self.assertEqual(result["return_outcome"]["absolute_hysteresis_before_return_adaptation"], 0.0)

    def test_parameter_drift_is_captured_before_return_adaptation(self):
        torch.manual_seed(23)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(24, 3, 8, 8)
        y = torch.arange(24) % 3
        snapshots = []
        original_metrics = experiment._metrics

        def recording_metrics(current_model, *args, **kwargs):
            snapshots.append(
                {
                    name: value.detach().clone()
                    for name, value in current_model.named_parameters()
                }
            )
            return original_metrics(current_model, *args, **kwargs)

        with mock.patch.object(experiment, "_metrics", side_effect=recording_metrics):
            result = experiment.run_recurrent_episode(
                model,
                x,
                y,
                method="tent",
                trajectory="A-B-A",
                shift_family="noise",
                severity=2,
                seed=23,
                batch_size=4,
                adaptation_passes=1,
                learning_rate=0.01,
            )

        self.assertEqual(len(snapshots), 4)
        adaptable_names = ["features.1.weight", "features.1.bias"]
        expected = torch.cat(
            [(snapshots[2][name] - snapshots[1][name]).flatten() for name in adaptable_names]
        ).norm()
        self.assertAlmostEqual(
            result["prospective_mechanism"]["parameter_drift"],
            float(expected),
            places=7,
        )

    def test_zero_learning_rate_separates_bn_configuration_from_update_hysteresis(self):
        torch.manual_seed(37)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(24, 3, 8, 8)
        y = torch.arange(24) % 3

        result = experiment.run_recurrent_episode(
            model,
            x,
            y,
            method="tent",
            trajectory="A-B-C-A",
            shift_family="blur",
            severity=2,
            seed=37,
            batch_size=4,
            adaptation_passes=2,
            learning_rate=0.0,
        )

        self.assertIn("checkpoint_source_evaluation", result)
        self.assertIn("configured_source_evaluation", result)
        self.assertAlmostEqual(
            result["return_outcome"]["update_induced_hysteresis_before_return_adaptation"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["return_outcome"]["absolute_hysteresis_before_return_adaptation"],
            result["configuration_gap"],
            places=12,
        )

    def test_periodic_reset_executes_after_eight_adaptation_steps(self):
        torch.manual_seed(11)
        model = experiment.TinyConvNet(classes=3)
        x = torch.rand(32, 3, 8, 8)
        y = torch.arange(32) % 3
        result = experiment.run_recurrent_episode(
            model,
            x,
            y,
            method="periodic_reset",
            trajectory="A-B-A",
            shift_family="noise",
            severity=1,
            seed=11,
            batch_size=4,
            adaptation_passes=1,
        )
        self.assertEqual(result["adaptation_steps_before_return"], 8)
        self.assertGreater(result["prospective_mechanism"]["update_norm"], 0.0)

    def test_resume_skips_hash_valid_outputs_and_preserves_failed_jobs(self):
        manifest = {
            "datasets": ["cifar10"],
            "seeds": [0],
            "methods": ["source", "tent"],
            "trajectories": ["A-B-A"],
            "shift_families": ["noise"],
            "severities": [1],
        }
        jobs = supervisor.jobs(manifest)
        self.assertEqual(len(jobs), 2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result_path = root / f"{jobs[0]['job_id']}.json"
            result_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
            prior = {
                "completed": {jobs[0]["job_id"]: supervisor.sha256(result_path)},
                "failed": [{"job_id": "old_failure", "attempts": 2, "error": "kept"}],
            }
            plan = supervisor.resume_plan(jobs, prior, root)
        self.assertEqual([job["job_id"] for job in plan["pending"]], [jobs[1]["job_id"]])
        self.assertEqual(plan["preserved_failures"][0]["job_id"], "old_failure")

    def test_resume_refuses_to_mix_results_from_different_provenance(self):
        manifest = {
            "datasets": ["cifar10"],
            "seeds": [0],
            "methods": ["source"],
            "trajectories": ["A-B-A"],
            "shift_families": ["noise"],
            "severities": [1],
        }
        job = supervisor.jobs(manifest)[0]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result_path = root / f"{job['job_id']}.json"
            result_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
            prior = {
                "completed": {job["job_id"]: supervisor.sha256(result_path)},
                "runner_sha256": "sha256:" + "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "provenance mismatch.*runner_sha256"):
                supervisor.resume_plan(
                    [job],
                    prior,
                    root,
                    expected_provenance={"runner_sha256": "sha256:" + "1" * 64},
                )

    def test_status_writer_serializes_paths_in_failure_records(self):
        with tempfile.TemporaryDirectory() as td:
            status = Path(td) / "status.json"
            checkpoint = Path(td) / "checkpoint.pt"
            supervisor.write(status, {"failed": [{"checkpoint": checkpoint}]})
            value = json.loads(status.read_text(encoding="utf-8"))
        self.assertEqual(value["failed"][0]["checkpoint"], str(checkpoint))

    def test_corrected_campaign_uses_an_isolated_result_set(self):
        self.assertEqual(
            supervisor.campaign_result_set({"result_set": "formal_v2"}, smoke=False),
            "formal_v2",
        )
        self.assertEqual(
            supervisor.campaign_result_set({"smoke_result_set": "smoke_v2"}, smoke=True),
            "smoke_v2",
        )
        with self.assertRaisesRegex(ValueError, "unsafe result_set"):
            supervisor.campaign_result_set({"result_set": "../formal"}, smoke=False)

    def test_resume_separates_historical_failures_from_current_run(self):
        manifest = {
            "datasets": ["cifar10"],
            "seeds": [0],
            "methods": ["source"],
            "trajectories": ["A-B-A"],
            "shift_families": ["noise"],
            "severities": [1],
        }
        job = supervisor.jobs(manifest)[0]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = root / "configs" / "campaign.json"
            result_path = root / "results" / "expansion" / "formal" / f"{job['job_id']}.json"
            status_path = root / "run_state" / "status.json"
            (root / "expansion_experiment.py").write_text("# fixture runner\n", encoding="utf-8")
            manifest_path.parent.mkdir(parents=True)
            result_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
            status_path.parent.mkdir(parents=True)
            provenance = {
                "manifest_sha256": supervisor.sha256(manifest_path),
                "runner_sha256": supervisor.sha256(root / "expansion_experiment.py"),
                "supervisor_sha256": supervisor.sha256(Path(supervisor.__file__)),
                "dataset_provenance_sha256": None,
            }
            status_path.write_text(
                json.dumps(
                    {
                        "completed": {job["job_id"]: supervisor.sha256(result_path)},
                        "failed": [{"job_id": "old_failure", "attempts": 3}],
                        **provenance,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "run_expansion_supervisor.py",
                "--manifest",
                str(manifest_path),
                "--status",
                str(status_path),
                "--resume",
                "--device",
                "cpu",
            ]
            with mock.patch.object(supervisor, "ROOT", root), mock.patch.object(supervisor.sys, "argv", argv):
                exit_code = supervisor.main()
            state = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(state["status"], "COMPLETED")
        self.assertEqual(state["failed"], [])
        self.assertEqual(state["preserved_failures"][0]["job_id"], "old_failure")
        self.assertRegex(state["runner_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(state["supervisor_sha256"], r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
