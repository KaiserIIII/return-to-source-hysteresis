from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "publication_expansion_campaigns.json"


def _supervisor():
    import publication_expansion_supervisor as supervisor

    return supervisor


def _campaign(name: str) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return config[name]


def test_job_generation_covers_methods_trajectories_and_never_formal_v2() -> None:
    supervisor = _supervisor()
    jobs = supervisor.enumerate_jobs("modern_baselines_v1", _campaign("modern_baselines_v1"), smoke=False)
    assert jobs
    assert {job["method"] for job in jobs} >= {"source", "tent", "eata", "sar", "cotta", "rotta"}
    assert {job["trajectory"] for job in jobs} == {"A-B-A", "A-B-C-A"}
    assert all(job["result_set"] != "formal_v2" for job in jobs)
    assert all("formal_v2" not in str(job["output"]) for job in jobs)


def test_smoke_plan_is_small_but_scientifically_representative() -> None:
    supervisor = _supervisor()
    jobs = supervisor.enumerate_jobs("modern_baselines_v1", _campaign("modern_baselines_v1"), smoke=True)
    assert len(jobs) >= 12
    assert {job["method"] for job in jobs} >= {"source", "tent", "eata", "sar", "cotta", "rotta"}
    assert {job["trajectory"] for job in jobs} == {"A-B-A", "A-B-C-A"}
    assert len({job["seed"] for job in jobs}) >= 2


def test_normalization_canary_covers_all_registered_policies() -> None:
    supervisor = _supervisor()
    campaign = _campaign("normalization_ablation_v1")
    jobs = supervisor.enumerate_jobs("normalization_ablation_v1", campaign, canary=True)
    assert {job["normalization_policy"] for job in jobs} == set(campaign["policies"])
    assert all(job["policy_checkpoint_required"] for job in jobs)


def test_scale_out_requires_all_canary_gates() -> None:
    supervisor = _supervisor()
    passing = {gate: {"status": "PASS"} for gate in supervisor.REQUIRED_GATES}
    assert supervisor.scale_out_decision(passing)["scale_out_authorized"] is True
    failing = dict(passing)
    failing["semantic_canary"] = {"status": "FAIL"}
    assert supervisor.scale_out_decision(failing)["scale_out_authorized"] is False
    assert "semantic_canary" in supervisor.scale_out_decision(failing)["failed_gates"]


def test_resume_rejects_provenance_mismatch_and_preserves_failures(tmp_path: Path) -> None:
    supervisor = _supervisor()
    jobs = [{"job_id": "x", "output": str(tmp_path / "x.json")}]
    prior = {"manifest_sha256": "sha256:old", "completed": {}, "failed": [{"job_id": "old"}]}
    with pytest.raises(ValueError, match="provenance mismatch"):
        supervisor.resume_plan(jobs, prior, tmp_path, expected_provenance={"manifest_sha256": "sha256:new"})
    plan = supervisor.resume_plan(jobs, {"completed": {}, "failed": [{"job_id": "old"}]}, tmp_path)
    assert plan["preserved_failures"] == [{"job_id": "old"}]


def test_resume_allows_first_launch_with_empty_prior_and_frozen_provenance(tmp_path: Path) -> None:
    supervisor = _supervisor()
    jobs = [{"job_id": "first", "output": str(tmp_path / "first.json")}]
    expected = {
        "manifest_sha256": "sha256:manifest",
        "runner_sha256": "sha256:runner",
        "supervisor_sha256": "sha256:supervisor",
        "dataset_provenance_sha256": "sha256:dataset",
    }
    plan = supervisor.resume_plan(jobs, {}, tmp_path, expected_provenance=expected)
    assert [job["job_id"] for job in plan["pending"]] == ["first"]


def test_admission_rejects_temporal_leak_and_unexpected_mutation(tmp_path: Path) -> None:
    admission = __import__("publication_campaign_admission")
    shard = {
        "method": "tent",
        "episode": {
            "events": {"predictor_step": 4, "outcome_step": 3},
            "state_transition": {"unexpected_mutations": ["bad.buffer"]},
            "return_outcome": {"accuracy_before_return_adaptation": 0.5},
            "target_stream_evaluation": {"accuracy": 0.4},
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(shard), encoding="utf-8")
    result = admission.inspect_shard(path)
    assert result["status"] == "FAIL"
    assert any("temporal" in finding for finding in result["findings"])
    assert any("unexpected" in finding for finding in result["findings"])


def test_formal_admission_requires_matching_completed_canary_ledger(tmp_path: Path) -> None:
    supervisor = _supervisor()
    expected = {
        "manifest_sha256": "sha256:manifest",
        "runner_sha256": "sha256:runner",
        "supervisor_sha256": "sha256:supervisor",
        "dataset_provenance_sha256": "sha256:dataset",
    }
    ledger = {
        "campaign_id": "modern_baselines_v1",
        "status": "FORMAL_ADMISSION_PASS",
        "provenance": dict(expected),
        "gates": {gate: {"status": "PASS"} for gate in supervisor.REQUIRED_GATES},
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert supervisor.validate_canary_admission(path, "modern_baselines_v1", expected)["scale_out_authorized"] is True
    ledger["provenance"]["runner_sha256"] = "sha256:changed"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert supervisor.validate_canary_admission(path, "modern_baselines_v1", expected)["scale_out_authorized"] is False


def test_finalize_canary_persists_completed_state_before_building_admission(tmp_path: Path) -> None:
    supervisor = _supervisor()
    status_path = tmp_path / "status.json"
    result_root = tmp_path / "results"
    state = {
        "campaign_id": "modern_baselines_v1",
        "result_set": "canary",
        "status": "RUNNING",
        "completed": {},
        "failed": [],
    }
    status_path.write_text(json.dumps(state), encoding="utf-8")
    campaign = {"methods": ["source"], "trajectories": ["A-B-A"], "seeds": [0, 1]}
    provenance = {
        "manifest_sha256": "sha256:manifest",
        "runner_sha256": "sha256:runner",
        "supervisor_sha256": "sha256:supervisor",
        "dataset_provenance_sha256": "sha256:dataset",
    }
    admission_module = __import__("publication_campaign_admission")
    original_builder = admission_module.build_admission_ledger
    observed_status: dict[str, str] = {}

    def checking_builder(*args, **kwargs):
        observed_status["status"] = json.loads(status_path.read_text(encoding="utf-8"))["status"]
        return {
            "status": "FORMAL_ADMISSION_FAIL",
            "campaign_id": "modern_baselines_v1",
            "gates": {gate: {"status": "FAIL"} for gate in supervisor.REQUIRED_GATES},
        }

    admission_module.build_admission_ledger = checking_builder
    try:
        finalized, _ = supervisor.finalize_canary_admission(
            status_path, result_root, state, campaign, provenance
        )
    finally:
        admission_module.build_admission_ledger = original_builder
    assert observed_status["status"] == "COMPLETED"
    assert finalized["status"] == "CANARY_ADMISSION_FAIL"
