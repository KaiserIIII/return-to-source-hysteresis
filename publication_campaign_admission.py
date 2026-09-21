from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GATES = (
    "source_immutability",
    "method_state_transition",
    "temporal_availability",
    "provenance_binding",
    "semantic_canary",
    "early_sanity_audit",
)
PROVENANCE_FIELDS = (
    "runner_sha256",
    "supervisor_sha256",
    "manifest_sha256",
    "dataset_provenance_sha256",
    "method_registry_sha256",
    "method_contracts_sha256",
)
MODERN_METHODS = {"eata", "sar", "cotta", "rotta"}


def _sha256_value(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _method_semantic_findings(shard: dict[str, Any], episode: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    method = str(shard.get("method", episode.get("method", "")))
    transition = episode.get("state_transition", {})
    optimizer_steps = transition.get("optimizer_step_count")
    if method == "source" and optimizer_steps != 0:
        findings.append(f"source optimizer step count must be 0, got {optimizer_steps!r}")
    if method not in MODERN_METHODS:
        return findings

    semantics = episode.get("algorithm_semantics")
    if not isinstance(semantics, dict):
        findings.append("algorithm semantics missing for modern method")
        semantics = {}
    elif semantics.get("implementation_status") != "INDEPENDENT_REIMPLEMENTATION":
        findings.append("algorithm semantics do not declare independent reimplementation")

    upstream = shard.get("method_upstream")
    provider = upstream.get("provider") if isinstance(upstream, dict) else None
    if not isinstance(provider, dict):
        findings.append("upstream binding missing for modern method")
    else:
        commit = provider.get("commit")
        files = provider.get("files")
        if not isinstance(commit, str) or len(commit) != 40:
            findings.append("upstream commit is not an immutable 40-character revision")
        if not isinstance(files, dict) or not files or not all(_sha256_value(value) for value in files.values()):
            findings.append("upstream file hashes are missing or malformed")
        if not _sha256_value(upstream.get("registry_sha256")):
            findings.append("upstream registry hash is missing or malformed")

    components = transition.get("components")
    if not isinstance(components, dict):
        findings.append("method state transition components missing")
        components = {}
    else:
        for name in ("student_model", "optimizer", "teacher", "anchor", "memory", "robust_bn", "recovery"):
            if not isinstance(components.get(name), dict):
                findings.append(f"method state transition component missing: {name}")
        anchor = components.get("anchor", {})
        if anchor.get("mutated") is True or anchor.get("before_digest") != anchor.get("after_digest"):
            findings.append("immutable anchor mutated")
        optimizer_component = components.get("optimizer", {})
        if optimizer_component.get("step_count") != semantics.get("optimizer_step_count"):
            findings.append("optimizer step count disagrees with algorithm semantics")

    if method == "eata":
        if not isinstance(semantics.get("fisher_samples"), int) or semantics.get("fisher_samples", 0) <= 0:
            findings.append("EATA Fisher importance was not estimated")
        if not _sha256_value(semantics.get("fisher_digest")):
            findings.append("EATA Fisher digest is missing")
        if semantics.get("nonredundant_samples", 0) > semantics.get("reliable_samples", -1):
            findings.append("EATA nonredundant sample count exceeds reliable sample count")
        if not _sha256_value(semantics.get("eata_probability_ema_digest")):
            findings.append("EATA probability EMA digest is missing")
    elif method == "sar":
        first = semantics.get("sam_first_steps")
        second = semantics.get("sam_second_steps")
        rollbacks = semantics.get("sam_perturbation_rollbacks")
        if not isinstance(first, int) or first <= 0 or first != second or second != rollbacks:
            findings.append("SAR two-step SAM perturbation/rollback counts are inconsistent")
        if semantics.get("optimizer_step_count") != second:
            findings.append("SAR optimizer count disagrees with second SAM steps")
        if "entropy_ema" not in semantics:
            findings.append("SAR entropy EMA recovery state is missing")
    elif method == "cotta":
        steps = semantics.get("optimizer_step_count")
        if semantics.get("optimizer_type") != "Adam":
            findings.append("CoTTA optimizer must be Adam")
        if semantics.get("teacher_updates") != steps:
            findings.append("CoTTA teacher update count disagrees with optimizer steps")
        if semantics.get("teacher_before_digest") == semantics.get("teacher_after_digest"):
            findings.append("CoTTA teacher did not change")
        if not isinstance(semantics.get("augmentation_predictions"), int) or semantics.get("augmentation_predictions", 0) < 32:
            findings.append("CoTTA augmentation-averaged teacher path was not exercised")
        if not isinstance(semantics.get("stochastic_restore_trials"), int) or semantics.get("stochastic_restore_trials", 0) <= 0:
            findings.append("CoTTA stochastic restore was not exercised")
    elif method == "rotta":
        occupancy = semantics.get("memory_occupancy")
        capacity = semantics.get("memory_capacity")
        histogram = semantics.get("memory_class_histogram")
        if semantics.get("memory_type") != "class_balanced_timeliness_uncertainty":
            findings.append("RoTTA memory type is incorrect")
        if not isinstance(occupancy, int) or not isinstance(capacity, int) or occupancy <= 0 or occupancy > capacity:
            findings.append("RoTTA memory occupancy is invalid")
        if not isinstance(histogram, list) or sum(histogram) != occupancy:
            findings.append("RoTTA class-balanced memory histogram is inconsistent")
        if not isinstance(semantics.get("memory_max_age"), int) or semantics.get("memory_max_age", 0) <= 0:
            findings.append("RoTTA memory age ledger was not exercised")
        if not isinstance(semantics.get("robust_bn_layers"), int) or semantics.get("robust_bn_layers", 0) <= 0:
            findings.append("RoTTA robust BN state is missing")
        if semantics.get("teacher_updates") != semantics.get("optimizer_step_count"):
            findings.append("RoTTA teacher update count disagrees with optimizer steps")
    return findings


def inspect_shard(
    path: Path,
    *,
    expected_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    findings: list[str] = []
    try:
        shard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "findings": [f"unreadable shard: {exc}"]}
    episode = shard.get("episode", shard)
    if expected_provenance:
        for field in PROVENANCE_FIELDS:
            expected = expected_provenance.get(field)
            actual = shard.get(field)
            if expected is not None and actual != expected:
                findings.append(
                    f"provenance binding violation: {field} expected {expected!r}, got {actual!r}"
                )
    events = episode.get("events", {})
    predictor = events.get("predictor_step")
    outcome = events.get("outcome_step")
    if not isinstance(predictor, int) or not isinstance(outcome, int) or predictor >= outcome:
        findings.append("temporal availability violation: predictor event must precede outcome")
    transition = episode.get("state_transition", {})
    if transition.get("unexpected_mutations"):
        findings.append(f"unexpected state mutations: {transition['unexpected_mutations']}")
    findings.extend(_method_semantic_findings(shard, episode))
    if episode.get("labels_used_during_adaptation") is True or shard.get("labels_used_during_adaptation") is True:
        findings.append("labels were used during adaptation")
    for section, fields in {
        "return_outcome": ("accuracy_before_return_adaptation", "accuracy_after_return_adaptation"),
        "target_stream_evaluation": ("accuracy",),
    }.items():
        values = episode.get(section, {})
        for field in fields:
            value = values.get(field)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                findings.append(f"metric range violation: {section}.{field}")
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "shard": shard}


def canary_coverage_findings(
    observed: set[tuple[Any, ...]], campaign: dict[str, Any]
) -> list[str]:
    """Return missing semantic-canary cells for a campaign.

    The canary is deliberately smaller than a formal matrix, but it must
    exercise every registered method, both canonical return trajectories, and
    two distinct source-training seeds.  A caller may provide an explicit
    ``canary_trajectories``/``canary_seeds`` in the manifest; otherwise the
    first two registered seeds and the canonical A-B-A/A-B-C-A pair are used.
    """
    findings: list[str] = []
    methods = {str(value) for value in campaign.get("methods", [])}
    if not methods:
        findings.append("canary has no registered methods")
    trajectories = set(
        campaign.get("canary_trajectories", ("A-B-A", "A-B-C-A"))
    )
    registered_trajectories = set(campaign.get("trajectories", trajectories))
    missing_registered = trajectories - registered_trajectories
    if missing_registered:
        findings.append(
            f"canary trajectories are not registered in campaign: {sorted(missing_registered)}"
        )
    seeds = [int(value) for value in campaign.get("canary_seeds", campaign.get("seeds", [])[:2])]
    if len(set(seeds)) < 2:
        findings.append("canary requires at least two distinct source-training seeds")
    required_seeds = set(seeds[:2])
    if len(required_seeds) < 2:
        required_seeds = set(seeds)
    normalization_policies = {str(value) for value in campaign.get("policies", [])}
    normalization_campaign = bool(normalization_policies)
    observed_methods = {cell[0] for cell in observed}
    observed_trajectories = {cell[1] for cell in observed}
    observed_seeds = {int(cell[2]) for cell in observed}
    missing_methods = methods - observed_methods
    if missing_methods:
        findings.append(f"canary methods missing: {sorted(missing_methods)}")
    missing_trajectories = trajectories - observed_trajectories
    if missing_trajectories:
        findings.append(f"canary trajectories missing: {sorted(missing_trajectories)}")
    missing_seeds = required_seeds - observed_seeds
    if missing_seeds:
        findings.append(f"canary seeds missing: {sorted(missing_seeds)}")
    for method in sorted(methods):
        for trajectory in sorted(trajectories):
            for seed in sorted(required_seeds):
                if normalization_campaign:
                    present = any(
                        len(cell) >= 4
                        and cell[0] == method
                        and cell[1] == trajectory
                        and int(cell[2]) == seed
                        and str(cell[3]) == policy
                        for cell in observed
                        for policy in normalization_policies
                    )
                else:
                    present = (method, trajectory, seed) in observed
                if not present:
                    findings.append(
                        f"canary cell missing: method={method}, trajectory={trajectory}, seed={seed}"
                    )
    if normalization_campaign:
        observed_policies = {str(cell[3]) for cell in observed if len(cell) >= 4}
        missing_policies = normalization_policies - observed_policies
        if missing_policies:
            findings.append(f"canary normalization policies missing: {sorted(missing_policies)}")
        for method in sorted(methods):
            for trajectory in sorted(trajectories):
                for seed in sorted(required_seeds):
                    for policy in sorted(normalization_policies):
                        if not any(
                            len(cell) >= 4
                            and cell[0] == method
                            and cell[1] == trajectory
                            and int(cell[2]) == seed
                            and str(cell[3]) == policy
                            for cell in observed
                        ):
                            findings.append(
                                f"canary cell missing: method={method}, trajectory={trajectory}, seed={seed}, policy={policy}"
                            )
    return findings


def _status_and_shards(
    status_path: Path,
    result_root: Path,
    *,
    expected_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    findings: list[str] = []
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [], [f"unreadable status ledger: {exc}"]
    shards: list[dict[str, Any]] = []
    completed = status.get("completed", {})
    if not isinstance(completed, dict):
        return status, [], ["completed ledger is not an object"]
    for job_id, expected_hash in sorted(completed.items()):
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"missing shard: {job_id}")
            continue
        actual_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            findings.append(f"result hash mismatch: {job_id}")
            continue
        inspected = inspect_shard(path, expected_provenance=expected_provenance)
        findings.extend(f"{job_id}: {item}" for item in inspected["findings"])
        inspected["job_id"] = job_id
        shards.append(inspected["shard"])
    if status.get("failed"):
        findings.append(f"failed jobs retained in canary: {len(status['failed'])}")
    return status, shards, findings


def build_admission_ledger(
    status_path: Path,
    result_root: Path,
    campaign: dict[str, Any],
    *,
    expected_provenance: dict[str, str],
) -> dict[str, Any]:
    """Build the append-only campaign admission decision from canary shards."""
    status, shards, shard_findings = _status_and_shards(
        status_path, result_root, expected_provenance=expected_provenance
    )
    normalization_campaign = bool(campaign.get("policies"))
    observed = {
        ((str(shard.get("method")), str(shard.get("trajectory")), int(shard.get("seed")), str(shard.get("normalization_policy", shard.get("protocol_parameters", {}).get("normalization_policy")))) if normalization_campaign else (str(shard.get("method")), str(shard.get("trajectory")), int(shard.get("seed"))))
        for shard in shards
        if shard.get("method") is not None
        and shard.get("trajectory") is not None
        and shard.get("seed") is not None
    }
    coverage_findings = canary_coverage_findings(observed, campaign)
    temporal_findings = [
        finding
        for finding in shard_findings
        if "temporal availability" in finding or "pre-update" in finding
    ]
    mutation_findings = [
        finding
        for finding in shard_findings
        if "unexpected state mutations" in finding
        or "labels were used" in finding
        or "unexpected configuration mutations" in finding
        or "method state transition" in finding
        or "optimizer" in finding
        or "Fisher" in finding
        or "EATA" in finding
        or "SAR" in finding
        or "CoTTA" in finding
        or "RoTTA" in finding
        or "anchor" in finding
    ]
    provenance_findings = [
        finding
        for finding in shard_findings
        if "provenance" in finding or "result hash" in finding or "missing shard" in finding
    ]
    source_findings: list[str] = []
    source_shards = [shard for shard in shards if shard.get("method") == "source"]
    for shard in source_shards:
        episode = shard.get("episode", {})
        transition = episode.get("state_transition", {})
        if transition.get("unexpected_mutations"):
            source_findings.append(f"{shard.get('job_id', shard.get('method'))}: source unexpected mutation")
        if transition.get("before_digest") != transition.get("after_digest"):
            source_findings.append(f"{shard.get('job_id', shard.get('method'))}: source state digest changed")
        if episode.get("prospective_mechanism", {}).get("parameter_drift") != 0.0:
            source_findings.append(f"{shard.get('job_id', shard.get('method'))}: source parameter drift is nonzero")
        if transition.get("optimizer_step_count") != 0:
            source_findings.append(f"{shard.get('job_id', shard.get('method'))}: source optimizer step count is not zero")
    all_audited = bool(shards) and not shard_findings
    gates = {
        "source_immutability": {
            "status": "PASS" if source_shards and not source_findings else "FAIL",
            "checked_shards": len(source_shards),
            "findings": source_findings,
        },
        "method_state_transition": {
            "status": "PASS" if not mutation_findings else "FAIL",
            "findings": mutation_findings,
        },
        "temporal_availability": {
            "status": "PASS" if not temporal_findings and bool(shards) else "FAIL",
            "findings": temporal_findings,
        },
        "provenance_binding": {
            "status": "PASS" if not provenance_findings and bool(shards) else "FAIL",
            "findings": provenance_findings,
            "expected": expected_provenance,
        },
        "semantic_canary": {
            "status": "PASS" if not coverage_findings and status.get("status") == "COMPLETED" and not status.get("failed") else "FAIL",
            "observed_cells": len(observed),
            "coverage": {"methods": sorted({cell[0] for cell in observed}), "trajectories": sorted({cell[1] for cell in observed}), "seeds": sorted({cell[2] for cell in observed})},
            "findings": coverage_findings,
        },
        "early_sanity_audit": {
            "status": "PASS" if all_audited else "FAIL",
            "checked_shards": len(shards),
            "findings": shard_findings,
        },
    }
    failed = [name for name in REQUIRED_GATES if gates[name]["status"] != "PASS"]
    return {
        "schema_version": "1.0.0",
        "campaign_id": status.get("campaign_id"),
        "result_set": status.get("result_set"),
        "status": "FORMAL_ADMISSION_PASS" if not failed else "FORMAL_ADMISSION_FAIL",
        "gates": gates,
        "failed_gates": failed,
        "provenance": expected_provenance,
        "source_shards": len(source_shards),
    }


def audit_result_set(status_path: Path, result_root: Path) -> dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    checked = 0
    for job_id, expected_hash in status.get("completed", {}).items():
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"missing shard: {job_id}")
            continue
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            findings.append(f"result hash mismatch: {job_id}")
        result = inspect_shard(path)
        findings.extend(f"{job_id}: {item}" for item in result["findings"])
        checked += 1
    return {"status": "PASS" if not findings else "FAIL", "checked_shards": checked, "findings": findings}
