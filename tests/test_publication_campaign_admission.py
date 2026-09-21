from __future__ import annotations

import json
from pathlib import Path


def _admission():
    import publication_campaign_admission as admission

    return admission


def _campaign():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "publication_expansion_campaigns.json").read_text(
            encoding="utf-8"
        )
    )
    return config["modern_baselines_v1"]


def test_canary_coverage_requires_every_method_two_trajectories_and_two_seeds() -> None:
    admission = _admission()
    observed = {
        ("source", "A-B-A", 0),
        ("tent", "A-B-A", 0),
        ("eata", "A-B-A", 0),
        ("sar", "A-B-A", 0),
        ("cotta", "A-B-A", 0),
        ("rotta", "A-B-A", 0),
    }
    findings = admission.canary_coverage_findings(observed, _campaign())
    assert findings
    assert any("seed" in finding for finding in findings)
    assert any("trajectory" in finding for finding in findings)


def test_canary_coverage_passes_only_with_complete_required_matrix() -> None:
    admission = _admission()
    campaign = _campaign()
    observed = {
        (method, trajectory, seed)
        for method in campaign["methods"]
        for trajectory in campaign["trajectories"]
        for seed in campaign["seeds"][:2]
    }
    assert admission.canary_coverage_findings(observed, campaign) == []


def test_normalization_canary_coverage_requires_all_policies() -> None:
    admission = _admission()
    campaign = json.loads((Path(__file__).resolve().parents[1] / "configs" / "publication_expansion_campaigns.json").read_text(encoding="utf-8"))["normalization_ablation_v1"]
    observed = {
        (method, trajectory, seed, policy)
        for method in campaign["methods"]
        for trajectory in campaign["trajectories"][:2]
        for seed in campaign["seeds"][:2]
        for policy in campaign["policies"]
    }
    assert admission.canary_coverage_findings(observed, campaign) == []


def test_admission_ledger_exposes_six_gates_and_fails_incomplete_canary(tmp_path: Path) -> None:
    admission = _admission()
    campaign = _campaign()
    status_path = tmp_path / "status.json"
    result_root = tmp_path / "results"
    status_path.write_text(
        json.dumps(
            {
                "campaign_id": "modern_baselines_v1",
                "result_set": "canary",
                "status": "COMPLETED",
                "completed": {},
                "failed": [],
                "total_jobs": 0,
            }
        ),
        encoding="utf-8",
    )
    ledger = admission.build_admission_ledger(
        status_path,
        result_root,
        campaign,
        expected_provenance={
            "manifest_sha256": "sha256:manifest",
            "runner_sha256": "sha256:runner",
            "supervisor_sha256": "sha256:supervisor",
            "dataset_provenance_sha256": "sha256:dataset",
        },
    )
    assert set(admission.REQUIRED_GATES) == set(ledger["gates"])
    assert ledger["status"] == "FORMAL_ADMISSION_FAIL"
    assert ledger["gates"]["semantic_canary"]["status"] == "FAIL"


def test_provenance_binding_rejects_shard_without_frozen_identities(tmp_path: Path) -> None:
    admission = _admission()
    shard_path = tmp_path / "job.json"
    shard_path.write_text(
        json.dumps(
            {
                "job_id": "job",
                "method": "source",
                "seed": 0,
                "trajectory": "A-B-A",
                "episode": {
                    "events": {"predictor_step": 1, "outcome_step": 2},
                    "state_transition": {},
                    "return_outcome": {"accuracy_before_return_adaptation": 0.5},
                    "target_stream_evaluation": {"accuracy": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    result = admission.inspect_shard(
        shard_path,
        expected_provenance={
            "runner_sha256": "sha256:runner",
            "manifest_sha256": "sha256:manifest",
            "supervisor_sha256": "sha256:supervisor",
            "dataset_provenance_sha256": "sha256:dataset",
        },
    )
    assert result["status"] == "FAIL"
    assert any("provenance" in finding for finding in result["findings"])


def test_admission_rejects_modern_method_without_semantic_or_upstream_binding(tmp_path: Path) -> None:
    admission = _admission()
    shard_path = tmp_path / "eata.json"
    shard_path.write_text(
        json.dumps(
            {
                "job_id": "eata",
                "method": "eata",
                "seed": 0,
                "trajectory": "A-B-A",
                "episode": {
                    "events": {"predictor_step": 1, "outcome_step": 2},
                    "state_transition": {"unexpected_mutations": [], "optimizer_step_count": 1},
                    "return_outcome": {"accuracy_before_return_adaptation": 0.5, "accuracy_after_return_adaptation": 0.5},
                    "target_stream_evaluation": {"accuracy": 0.5},
                    "labels_used_during_adaptation": False,
                },
            }
        ),
        encoding="utf-8",
    )
    result = admission.inspect_shard(shard_path)
    assert result["status"] == "FAIL"
    assert any("algorithm semantics" in finding for finding in result["findings"])
    assert any("upstream" in finding for finding in result["findings"])


def test_admission_rejects_source_optimizer_steps(tmp_path: Path) -> None:
    admission = _admission()
    shard_path = tmp_path / "source.json"
    shard_path.write_text(
        json.dumps(
            {
                "method": "source",
                "episode": {
                    "events": {"predictor_step": 1, "outcome_step": 2},
                    "state_transition": {"unexpected_mutations": [], "optimizer_step_count": 1},
                    "return_outcome": {"accuracy_before_return_adaptation": 0.5, "accuracy_after_return_adaptation": 0.5},
                    "target_stream_evaluation": {"accuracy": 0.5},
                    "labels_used_during_adaptation": False,
                },
            }
        ),
        encoding="utf-8",
    )
    result = admission.inspect_shard(shard_path)
    assert result["status"] == "FAIL"
    assert any("source optimizer" in finding for finding in result["findings"])
