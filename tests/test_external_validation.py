from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "external_validation_cifar_c_v1.json"


def test_extract_severity_block_aligns_standard_and_repeated_labels() -> None:
    import external_validation_experiment as runner

    images = np.arange(50000, dtype=np.int64)
    labels = np.arange(10000, dtype=np.int64)
    selected_images, selected_labels, sample_range = runner.extract_severity_block(
        images, labels, severity=3, block_size=10000
    )
    assert selected_images[[0, -1]].tolist() == [20000, 29999]
    assert selected_labels[[0, -1]].tolist() == [0, 9999]
    assert sample_range == [20000, 30000]

    repeated_labels = np.tile(labels, 5)
    _, selected_labels, _ = runner.extract_severity_block(
        images, repeated_labels, severity=5, block_size=10000
    )
    assert selected_labels[[0, -1]].tolist() == [0, 9999]


def test_external_episode_uses_real_stage_tensors_and_restores_engine() -> None:
    import publication_expansion_experiment as engine
    import external_validation_experiment as runner

    torch.manual_seed(9)
    model = engine.TinyConvNet(classes=3)
    clean = torch.rand(12, 3, 8, 8)
    stage_b = torch.zeros_like(clean)
    stage_c = torch.ones_like(clean)
    labels = torch.randint(0, 3, (12,))
    original = engine._stages
    result = runner.run_external_episode(
        model,
        clean,
        labels,
        stage_b=stage_b,
        stage_c=stage_c,
        method="source",
        trajectory="A-B-C-A",
        seed=0,
        batch_size=4,
        adaptation_passes=1,
        learning_rate=1e-3,
    )
    assert engine._stages is original
    assert result["shift_sequence"] == ["B", "C"]
    assert result["target_stream_evaluation"]["examples"] == 24
    assert result["state_transition"]["unexpected_mutations"] == []


def test_formal_matrix_is_complete_unique_and_isolated() -> None:
    import external_validation_supervisor as supervisor

    campaign = json.loads(CONFIG.read_text(encoding="utf-8"))
    jobs = supervisor.enumerate_formal_jobs(campaign)
    assert len(jobs) == 2700
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert {job["dataset"] for job in jobs} == {"cifar10_c", "cifar100_c"}
    assert {job["severity"] for job in jobs} == {1, 3, 5}
    assert {job["method"] for job in jobs} == {
        "source", "tent", "eata", "sar", "cotta", "rotta"
    }
    assert {job["trajectory"] for job in jobs} == {"A-B-A"}
    assert all("formal_v2" not in job["job_id"] for job in jobs)
    assert all("formal_v2" not in job["output"] for job in jobs)
    assert {job["method"] for job in jobs[:12]} == set(campaign["methods"])
    assert {job["seed"] for job in jobs[:12]} == {0, 1}


def test_semantic_canary_has_required_scientific_coverage() -> None:
    import external_validation_supervisor as supervisor

    campaign = json.loads(CONFIG.read_text(encoding="utf-8"))
    jobs = supervisor.enumerate_canary_jobs(campaign)
    assert len(jobs) == 48
    assert {job["dataset"] for job in jobs} == {"cifar10_c", "cifar100_c"}
    assert {job["method"] for job in jobs} == set(campaign["methods"])
    assert {job["seed"] for job in jobs} == {0, 1}
    assert {job["trajectory"] for job in jobs} == {"A-B-A", "A-B-C-A"}
    assert {job["severity"] for job in jobs} == {1, 5}
    categories = {
        campaign["corruption_categories"][job["corruption"]] for job in jobs
    }
    assert categories == {"noise", "blur", "weather", "digital"}
    assert all(job["paired_corruption"] != job["corruption"] for job in jobs)
    for dataset in campaign["datasets"]:
        for method in campaign["methods"]:
            cells = {
                (job["seed"], job["trajectory"])
                for job in jobs
                if job["dataset"] == dataset and job["method"] == method
            }
            assert cells == {
                (0, "A-B-A"),
                (0, "A-B-C-A"),
                (1, "A-B-A"),
                (1, "A-B-C-A"),
            }


def test_external_coverage_gate_rejects_synthetic_or_thin_canary() -> None:
    import external_validation_supervisor as supervisor

    campaign = json.loads(CONFIG.read_text(encoding="utf-8"))
    shards = [
        {
            "dataset": "cifar10_c",
            "method": "source",
            "seed": 0,
            "trajectory": "A-B-A",
            "severity": 1,
            "corruption": "gaussian_noise",
            "dataset_identity": {"protocol": "synthetic"},
        }
    ]
    findings = supervisor.external_coverage_findings(shards, campaign)
    assert any("real standard_cifar_c" in finding for finding in findings)
    assert any("dataset" in finding for finding in findings)
    assert any("corruption category" in finding for finding in findings)
    assert any("method-seed-trajectory" in finding for finding in findings)


def test_resume_rejects_frozen_identity_change(tmp_path: Path) -> None:
    import external_validation_supervisor as supervisor

    jobs = [{"job_id": "a"}]
    prior = {"runner_sha256": "sha256:old", "completed": {}}
    with pytest.raises(ValueError, match="provenance mismatch"):
        supervisor.resume_plan(
            jobs,
            prior,
            tmp_path,
            expected_provenance={"runner_sha256": "sha256:new"},
        )


def test_blocked_external_data_is_prelaunch_but_integrity_failure_is_terminal(tmp_path: Path) -> None:
    import external_validation_supervisor as supervisor

    jobs = [{"job_id": "a"}]
    plan = supervisor.resume_plan(
        jobs,
        {"status": "BLOCKED_EXTERNAL_DATA", "missing": ["data"]},
        tmp_path,
        expected_provenance={"runner_sha256": "sha256:new"},
    )
    assert plan["pending"] == jobs
    with pytest.raises(ValueError, match="integrity failure"):
        supervisor.resume_plan(
            jobs,
            {"status": "EARLY_SANITY_FAIL", "completed": {}},
            tmp_path,
            expected_provenance={"runner_sha256": "sha256:new"},
        )


def test_checkpoint_inventory_freezes_every_dataset_seed(tmp_path: Path) -> None:
    import external_validation_supervisor as supervisor

    campaign = {"datasets": ["cifar10_c", "cifar100_c"], "seeds": [0, 1]}
    checkpoint_root = tmp_path / "checkpoints" / "expansion" / "formal"
    checkpoint_root.mkdir(parents=True)
    for dataset in ("cifar10", "cifar100"):
        for seed in (0, 1):
            (checkpoint_root / f"{dataset}_s{seed:02d}.pt").write_bytes(
                f"{dataset}:{seed}".encode("ascii")
            )
    inventory = supervisor.freeze_checkpoint_inventory(campaign, root=tmp_path)
    assert len(inventory["files"]) == 4
    assert inventory["sha256"].startswith("sha256:")
    original = inventory["sha256"]
    (checkpoint_root / "cifar10_s00.pt").write_bytes(b"changed")
    assert supervisor.freeze_checkpoint_inventory(campaign, root=tmp_path)["sha256"] != original


def test_expected_digest_rejects_mid_campaign_file_change(tmp_path: Path) -> None:
    import external_validation_experiment as runner

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"frozen")
    expected = "sha256:" + hashlib.sha256(b"frozen").hexdigest()
    assert runner.verify_expected_digest(path, expected, "source checkpoint") == expected
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source checkpoint hash changed"):
        runner.verify_expected_digest(path, expected, "source checkpoint")
