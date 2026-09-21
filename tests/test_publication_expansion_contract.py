from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "publication_expansion_campaigns.json"
UPSTREAM = ROOT / "configs" / "modern_baseline_upstreams.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_publication_expansion_config_exists_and_is_versioned() -> None:
    value = load_config()
    assert value["schema_version"] == "1.0.0"
    assert value["formal_v2_immutable"] is True
    assert value["formal_v2_result_set"] == "formal_v2"
    assert set(value["campaigns"]) == {
        "modern_baselines_v1",
        "normalization_ablation_v1",
        "trajectory_ablation_v1",
        "external_validation_v1",
        "mechanism_v2",
    }


def test_modern_baselines_are_independent_and_complete() -> None:
    value = load_config()
    methods = set(value["modern_baselines_v1"]["methods"])
    assert {"source", "tent", "eata", "sar", "cotta", "rotta"} <= methods
    assert value["modern_baselines_v1"]["result_set"] != "formal_v2"
    assert value["modern_baselines_v1"]["formal_v2_overwrite"] is False
    assert value["modern_baselines_v1"]["independent_replication_unit"] == "source-training-seed"


def test_modern_baseline_upstreams_are_pinned_and_hash_audited() -> None:
    registry = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    assert registry["installation_performed"] is False
    assert set(registry["providers"]) == {"eata", "sar", "cotta", "rotta"}
    for provider in registry["providers"].values():
        assert len(provider["commit"]) == 40
        assert provider["license"] in {"MIT", "BSD-3-Clause"}
        assert provider["files"]
        assert all(value.startswith("sha256:") and len(value) == 71 for value in provider["files"].values())


def test_ablation_and_mechanism_factors_are_registered() -> None:
    value = load_config()
    norm = value["normalization_ablation_v1"]
    assert {"source_eval", "bn_affine_frozen_stats", "bn_affine_running_stats"} <= set(norm["policies"])
    trajectories = set(value["trajectory_ablation_v1"]["trajectories"])
    assert {"A-B-A", "A-B-C-A", "A-B-A-B-A", "A-C-B-A"} <= trajectories
    assert value["trajectory_ablation_v1"]["pass_counts"] == [1, 2, 4]
    assert value["trajectory_ablation_v1"]["learning_rates"] == [0.0001, 0.001, 0.01]
    mechanism = value["mechanism_v2"]
    assert mechanism["fresh_seeds"] == [5, 6, 7, 8, 9]
    assert mechanism["strict_pre_outcome"] is True
    assert mechanism["failure_policy"] in {"NEGATIVE", "ABANDONED"}


def test_all_campaigns_require_scientific_admission() -> None:
    value = load_config()
    for name in value["campaigns"]:
        campaign = value[name]
        assert campaign["smoke_result_set"] != campaign["result_set"]
        assert campaign["result_set"] != "formal_v2"
        assert campaign["formal_v2_overwrite"] is False
        assert campaign["required_gates"] == [
            "source_immutability",
            "method_state_transition",
            "temporal_availability",
            "provenance_binding",
            "semantic_canary",
            "early_sanity_audit",
        ]
