from __future__ import annotations

import copy

import torch


def _runner():
    import publication_expansion_experiment as runner

    return runner


def _episode_model():
    runner = _runner()
    torch.manual_seed(7)
    return runner.TinyConvNet(classes=3)


def _data():
    torch.manual_seed(8)
    x = torch.rand(12, 3, 8, 8)
    y = torch.randint(0, 3, (12,))
    return x, y


def test_runner_exposes_all_required_methods_and_trajectories() -> None:
    runner = _runner()
    assert {"source", "tent", "eata", "sar", "cotta", "rotta"} <= set(runner.METHODS)
    assert {"A-B-A", "A-B-C-A", "A-B-A-B-A", "A-C-B-A"} <= set(runner.TRAJECTORIES)


def test_source_episode_is_immutable_and_temporally_ordered() -> None:
    runner = _runner()
    model = _episode_model()
    before = runner.state_dict_digest(model)
    x, y = _data()
    result = runner.run_recurrent_episode(
        model,
        x,
        y,
        method="source",
        trajectory="A-B-C-A",
        shift_family="noise",
        severity=1,
        seed=5,
        batch_size=4,
        adaptation_passes=1,
        learning_rate=1e-3,
    )
    assert runner.state_dict_digest(model) == before
    assert result["state_transition"]["unexpected_mutations"] == []
    assert result["events"]["predictor_step"] < result["events"]["outcome_step"]
    assert result["labels_used_during_adaptation"] is False


def test_modern_methods_produce_declared_state_ledgers() -> None:
    runner = _runner()
    x, y = _data()
    for method in ("tent", "eata", "sar", "cotta", "rotta"):
        model = _episode_model()
        result = runner.run_recurrent_episode(
            model,
            x,
            y,
            method=method,
            trajectory="A-B-A",
            shift_family="brightness",
            severity=1,
            seed=11,
            batch_size=4,
            adaptation_passes=1,
            learning_rate=1e-3,
        )
        assert result["method"] == method
        assert result["state_transition"]["declared_mutations"]
        assert result["state_transition"]["unexpected_mutations"] == []
        assert result["events"]["predictor_step"] < result["events"]["outcome_step"]


def test_modern_methods_expose_component_state_ledgers() -> None:
    runner = _runner()
    x, y = _data()
    for method in ("eata", "sar", "cotta", "rotta"):
        result = runner.run_recurrent_episode(
            _episode_model(), x, y, method=method, trajectory="A-B-A",
            shift_family="noise", severity=1, seed=15, batch_size=4,
            adaptation_passes=1, learning_rate=1e-3,
        )
        components = result["state_transition"]["components"]
        assert components["student_model"]["after_digest"].startswith("sha256:")
        assert components["optimizer"]["step_count"] == result["algorithm_semantics"]["optimizer_step_count"]
        assert components["anchor"]["mutated"] is False
        if method in {"cotta", "rotta"}:
            assert components["teacher"]["before_digest"] != components["teacher"]["after_digest"]
        if method == "rotta":
            assert components["memory"]["occupancy"] > 0


def test_normalization_policy_is_explicit_and_source_state_is_preserved() -> None:
    runner = _runner()
    for policy in ("source_eval", "bn_affine_frozen_stats", "bn_affine_running_stats"):
        model = _episode_model()
        before = runner.state_dict_digest(model)
        x, y = _data()
        result = runner.run_recurrent_episode(
            model,
            x,
            y,
            method="tent",
            trajectory="A-B-A",
            shift_family="noise",
            severity=1,
            seed=13,
            batch_size=4,
            adaptation_passes=1,
            learning_rate=1e-3,
            normalization_policy=policy,
        )
        assert result["normalization_policy"] == policy
        assert runner.state_dict_digest(model) == before


def test_groupnorm_and_layernorm_policies_use_real_normalization_modules() -> None:
    runner = _runner()
    x, y = _data()
    for policy in ("groupnorm_checkpoint", "layernorm_checkpoint"):
        model = _episode_model()
        result = runner.run_recurrent_episode(
            model, x, y, method="source", trajectory="A-B-A",
            shift_family="noise", severity=1, seed=14, batch_size=4,
            adaptation_passes=1, learning_rate=1e-3,
            normalization_policy=policy,
        )
        assert result["normalization_modules"]["policy"] == policy
        assert result["normalization_modules"]["module_types"]
        assert all(policy.split("_")[0].capitalize() in value for value in result["normalization_modules"]["module_types"])


def test_eata_records_fisher_and_two_stage_sample_selection() -> None:
    runner = _runner()
    x, y = _data()
    result = runner.run_recurrent_episode(
        _episode_model(), x, y, method="eata", trajectory="A-B-A",
        shift_family="noise", severity=1, seed=21, batch_size=4,
        adaptation_passes=1, learning_rate=1e-3,
    )
    audit = result["algorithm_semantics"]
    assert audit["implementation_status"] == "INDEPENDENT_REIMPLEMENTATION"
    assert audit["fisher_samples"] > 0
    assert audit["fisher_digest"].startswith("sha256:")
    assert audit["reliable_samples"] >= audit["nonredundant_samples"]
    assert audit["eata_probability_ema_digest"].startswith("sha256:")


def test_sar_records_balanced_two_step_sam_and_recovery_state() -> None:
    runner = _runner()
    x, y = _data()
    result = runner.run_recurrent_episode(
        _episode_model(), x, y, method="sar", trajectory="A-B-A",
        shift_family="brightness", severity=1, seed=22, batch_size=4,
        adaptation_passes=1, learning_rate=1e-3,
    )
    audit = result["algorithm_semantics"]
    assert audit["sam_first_steps"] == audit["sam_second_steps"]
    assert audit["optimizer_step_count"] == audit["sam_second_steps"]
    assert audit["sam_perturbation_rollbacks"] == audit["sam_second_steps"]
    assert "entropy_ema" in audit


def test_cotta_records_teacher_augmentation_and_stochastic_restore() -> None:
    runner = _runner()
    x, y = _data()
    result = runner.run_recurrent_episode(
        _episode_model(), x, y, method="cotta", trajectory="A-B-A",
        shift_family="noise", severity=1, seed=23, batch_size=4,
        adaptation_passes=1, learning_rate=1e-3,
    )
    audit = result["algorithm_semantics"]
    assert audit["optimizer_type"] == "Adam"
    assert audit["teacher_updates"] == audit["optimizer_step_count"]
    assert audit["teacher_after_digest"] != audit["teacher_before_digest"]
    assert audit["augmentation_predictions"] >= 32
    assert audit["stochastic_restore_trials"] > 0


def test_rotta_records_balanced_memory_ages_teacher_and_robust_bn() -> None:
    runner = _runner()
    x, y = _data()
    result = runner.run_recurrent_episode(
        _episode_model(), x, y, method="rotta", trajectory="A-B-C-A",
        shift_family="noise", severity=1, seed=24, batch_size=4,
        adaptation_passes=1, learning_rate=1e-3,
    )
    audit = result["algorithm_semantics"]
    assert audit["memory_type"] == "class_balanced_timeliness_uncertainty"
    assert audit["memory_occupancy"] <= audit["memory_capacity"]
    assert sum(audit["memory_class_histogram"]) == audit["memory_occupancy"]
    assert audit["memory_max_age"] > 0
    assert audit["robust_bn_layers"] > 0
    assert audit["teacher_updates"] == audit["optimizer_step_count"]


def test_rotta_memory_replacement_handles_tensor_payloads() -> None:
    runner = _runner()
    x, y = _data()
    result = runner.run_recurrent_episode(
        _episode_model(), x.repeat(4, 1, 1, 1), y.repeat(4), method="rotta",
        trajectory="A-B-C-A", shift_family="noise", severity=1, seed=25,
        batch_size=4, adaptation_passes=2, learning_rate=1e-3,
    )
    audit = result["algorithm_semantics"]
    assert audit["memory_occupancy"] <= audit["memory_capacity"]
