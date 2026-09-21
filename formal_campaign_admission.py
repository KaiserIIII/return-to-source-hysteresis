from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
HASH_FIELDS = (
    "runner_sha256",
    "supervisor_sha256",
    "manifest_sha256",
    "dataset_provenance_sha256",
)
METHODS = ("source", "tent", "anchor", "ema_restore", "periodic_reset")
TRAJECTORIES = ("A-B-A", "A-B-C-A")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def freeze_checkpoint_inventory(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    identities: dict[str, str] = {}
    findings: list[str] = []
    checkpoint_root = root / "checkpoints" / "expansion" / "formal"
    for dataset in manifest.get("datasets", []):
        for seed in manifest.get("seeds", []):
            path = checkpoint_root / f"{dataset}_s{int(seed):02d}.pt"
            if not path.is_file():
                findings.append(f"initial source checkpoint is missing: {path}")
                continue
            identities[f"{dataset}:s{int(seed):02d}"] = sha256(path)
    expected_count = len(manifest.get("datasets", [])) * len(manifest.get("seeds", []))
    if len(identities) != expected_count:
        findings.append(
            f"initial source checkpoint inventory is incomplete: {len(identities)}/{expected_count}"
        )
    return {
        "status": "PASS" if not findings else "FAIL",
        "checkpoint_sha256s": identities,
        "findings": findings,
    }


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(f"formal_runner_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_snapshot(model) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tensor_map_sha256(values: dict[str, Any], *, bn_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        if bn_only and not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        value = values[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _changed_keys(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    changed = set(before) ^ set(after)
    for name in set(before) & set(after):
        if not before[name].equal(after[name]):
            changed.add(name)
    return changed


def _parameter_delta(model, before: dict[str, Any]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name in before:
            total += float((parameter.detach().cpu() - before[name]).pow(2).sum())
    return math.sqrt(total)


def temporal_findings(shard: dict[str, Any], name: str) -> list[str]:
    findings: list[str] = []
    episode = shard.get("episode", {})
    prospective = episode.get("prospective_mechanism", {})
    outcome = episode.get("return_outcome", {})
    measurement = prospective.get("measurement_step")
    outcome_step = outcome.get("measurement_step")
    if not isinstance(measurement, int) or not isinstance(outcome_step, int) or measurement >= outcome_step:
        findings.append(
            f"{name}: prospective predictor violates measurement_time < outcome_time "
            f"({measurement!r} !< {outcome_step!r})"
        )
    timing = episode.get("target_stream_evaluation", {}).get("timing")
    if timing != "pre_update_online_predictions":
        findings.append(f"{name}: target-stream metric is not explicitly pre-update")
    return findings


def run_semantic_canary(
    runner_path: Path,
    contracts: dict[str, Any],
    *,
    seeds: tuple[int, ...] = (901, 902),
    sample_count: int = 8,
) -> dict[str, Any]:
    runner = _load_runner(runner_path)
    declared = contracts.get("methods", {})
    findings: list[str] = []
    cases: list[dict[str, Any]] = []
    source_text = inspect.getsource(runner.run_recurrent_episode)
    if "optimizer.step" in source_text or "optim." in source_text:
        findings.append("episode runner contains an optimizer path that the canary cannot account for")

    for seed in seeds:
        generator = runner.torch.Generator().manual_seed(seed)
        x = runner.torch.rand(sample_count, 3, 8, 8, generator=generator)
        y = runner.torch.arange(sample_count) % 3
        for method in METHODS:
            contract = declared.get(method)
            if not isinstance(contract, dict):
                findings.append(f"missing method state-transition contract: {method}")
                continue
            for trajectory in TRAJECTORIES:
                model = runner.TinyConvNet(classes=3)
                captured: dict[str, Any] = {"train_modes": []}
                original_copy_module = runner.copy
                original_deepcopy = runner.copy.deepcopy
                original_adaptable = runner._adaptable

                def tracked_deepcopy(value):
                    clone = original_deepcopy(value)
                    captured["model"] = clone
                    captured["before"] = _tensor_snapshot(clone)
                    original_train = clone.train

                    def tracked_train(mode: bool = True):
                        captured["train_modes"].append(bool(mode))
                        return original_train(mode)

                    clone.train = tracked_train
                    return clone

                def tracked_adaptable(value):
                    selected = original_adaptable(value)
                    captured["adaptable_names"] = {name for name, _ in selected}
                    captured["configured"] = _tensor_snapshot(value)
                    return selected

                class CopyProxy:
                    @staticmethod
                    def deepcopy(value):
                        return tracked_deepcopy(value)

                runner.copy = CopyProxy()
                runner._adaptable = tracked_adaptable
                try:
                    episode = runner.run_recurrent_episode(
                        model,
                        x,
                        y,
                        method=method,
                        trajectory=trajectory,
                        shift_family="noise",
                        severity=1,
                        seed=seed,
                        batch_size=4,
                        adaptation_passes=1,
                    )
                finally:
                    runner.copy = original_copy_module
                    runner._adaptable = original_adaptable

                clone = captured["model"]
                before = captured["before"]
                configured = captured.get("configured", before)
                after = _tensor_snapshot(clone)
                configured_changes = _changed_keys(before, configured)
                trajectory_changes = _changed_keys(configured, after)
                bn_config_changes = {
                    name for name in configured_changes
                    if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
                }
                unexpected_config = configured_changes - bn_config_changes
                allowed_trajectory = set(captured.get("adaptable_names", set())) if contract.get("bn_affine") else set()
                unexpected_trajectory = trajectory_changes - allowed_trajectory
                state_before_hash = _tensor_map_sha256(before)
                state_after_hash = _tensor_map_sha256(after)
                bn_before_hash = _tensor_map_sha256(before, bn_only=True)
                bn_after_hash = _tensor_map_sha256(after, bn_only=True)
                delta = _parameter_delta(clone, before)
                entered_mutating_train_mode = any(captured["train_modes"])
                case_findings = temporal_findings({"episode": episode}, f"canary:{method}:{trajectory}:s{seed}")

                if method == "source":
                    if state_before_hash != state_after_hash:
                        case_findings.append("source state_dict changed")
                    if bn_before_hash != bn_after_hash:
                        case_findings.append("source BatchNorm buffers changed")
                    if delta != 0.0:
                        case_findings.append(f"source trainable parameter delta is {delta}")
                    if entered_mutating_train_mode:
                        case_findings.append("source entered state-mutating train mode")
                    if episode["prospective_mechanism"]["update_norm"] != 0.0:
                        case_findings.append("source update norm is nonzero")
                    if episode["prospective_mechanism"]["parameter_drift"] != 0.0:
                        case_findings.append("source parameter drift is nonzero")
                else:
                    if unexpected_config:
                        case_findings.append(f"unexpected configuration mutations: {sorted(unexpected_config)}")
                    expected_bn = contract.get("bn_running_statistics")
                    if bn_config_changes and expected_bn != "DISABLED_AT_CONFIGURATION":
                        case_findings.append(f"undeclared BatchNorm configuration mutation: {sorted(bn_config_changes)}")
                    if unexpected_trajectory:
                        case_findings.append(f"unexpected trajectory mutations: {sorted(unexpected_trajectory)}")

                case = {
                    "method": method,
                    "trajectory": trajectory,
                    "seed": seed,
                    "status": "PASS" if not case_findings else "FAIL",
                    "state_dict_before_sha256": state_before_hash,
                    "state_dict_after_sha256": state_after_hash,
                    "bn_buffers_before_sha256": bn_before_hash,
                    "bn_buffers_after_sha256": bn_after_hash,
                    "optimizer_step_count": 0,
                    "trainable_parameter_delta": delta,
                    "entered_mutating_train_mode": entered_mutating_train_mode,
                    "configured_mutations": sorted(configured_changes),
                    "trajectory_mutations": sorted(trajectory_changes),
                    "prospective_event_id": f"step:{episode['prospective_mechanism']['measurement_step']}",
                    "outcome_event_id": f"step:{episode['return_outcome']['measurement_step']}",
                    "findings": case_findings,
                }
                cases.append(case)
                findings.extend(case_findings)

    return {
        "status": "PASS" if not findings else "FAIL",
        "runner_sha256": sha256(runner_path),
        "case_count": len(cases),
        "coverage": {
            "methods": list(METHODS),
            "trajectories": list(TRAJECTORIES),
            "seeds": list(seeds),
            "shift_family": "noise",
            "severity": 1,
        },
        "cases": cases,
        "findings": findings,
    }


def validate_v1_fixture(value: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    required = {
        "SOURCE_BN_STATE_CONTAMINATION",
        "DRIFT_TEMPORAL_MISLABELING",
        "INCOMPLETE_RUNNER_DATASET_PROVENANCE",
    }
    if value.get("disposition") != "INVALID_FOR_PRIMARY_EVIDENCE":
        findings.append("formal_v1 disposition must remain INVALID_FOR_PRIMARY_EVIDENCE")
    if value.get("observed_jobs") != 1000:
        findings.append("formal_v1 observed job count must remain 1000")
    if set(value.get("reason_codes", [])) != required:
        findings.append("formal_v1 reason codes changed")
    if value.get("deletion_policy") != "PRESERVE_IMMUTABLY":
        findings.append("formal_v1 must be preserved immutably")
    for field in ("status_sha256", "manifest_sha256", "result_hash_list_sha256"):
        if not isinstance(value.get(field), str) or not value[field].startswith("sha256:"):
            findings.append(f"formal_v1 {field} is missing")
    return findings


def audit_shard_bindings(
    status: dict[str, Any],
    result_root: Path,
    *,
    expected: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    findings: list[str] = []
    bindings: list[dict[str, Any]] = []
    for field in HASH_FIELDS:
        if status.get(field) != expected.get(field):
            findings.append(f"campaign {field} mismatch")
    completed = status.get("completed", {})
    if not isinstance(completed, dict):
        return {"status": "FAIL", "findings": ["completed ledger is not an object"], "bindings": []}
    expected_protocol = {
        "batch_size": manifest.get("batch_size"),
        "adaptation_passes": manifest.get("adaptation_passes"),
        "adaptation_learning_rate": manifest.get("adaptation_learning_rate"),
    }
    for job_id, expected_result_hash in sorted(completed.items()):
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"{job_id}: result shard is missing")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_result_hash:
            findings.append(f"{job_id}: result hash mismatch")
            continue
        try:
            shard = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(f"{job_id}: unreadable JSON: {exc}")
            continue
        if shard.get("runner_sha256") != expected.get("runner_sha256"):
            findings.append(f"{job_id}: runner hash mismatch")
        if shard.get("dataset_provenance_sha256") != expected.get("dataset_provenance_sha256"):
            findings.append(f"{job_id}: dataset provenance mismatch")
        if shard.get("protocol_parameters") != expected_protocol:
            findings.append(f"{job_id}: protocol parameters mismatch")
        if not isinstance(shard.get("environment"), dict) or not shard["environment"].get("torch"):
            findings.append(f"{job_id}: environment identity is missing")
        checkpoint_hash = shard.get("source_checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 71 or not checkpoint_hash.startswith("sha256:"):
            findings.append(f"{job_id}: checkpoint identity is missing")
        findings.extend(temporal_findings(shard, job_id))
        bindings.append(
            {
                "job_id": job_id,
                "result_sha256": actual_hash,
                "runner_sha256": expected.get("runner_sha256"),
                "supervisor_sha256": expected.get("supervisor_sha256"),
                "manifest_sha256": expected.get("manifest_sha256"),
                "dataset_provenance_sha256": expected.get("dataset_provenance_sha256"),
                "source_checkpoint_sha256": checkpoint_hash,
            }
        )
    return {"status": "PASS" if not findings else "FAIL", "checked_shards": len(bindings), "bindings": bindings, "findings": findings}


def early_sanity_audit(status: dict[str, Any], result_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    findings = list(provenance.get("findings", []))
    checked = 0
    source_checked = 0
    methods: set[str] = set()
    for binding in provenance.get("bindings", []):
        shard = load_json(result_root / f"{binding['job_id']}.json")
        episode = shard["episode"]
        checked += 1
        methods.add(shard["method"])
        for section in ("checkpoint_source_evaluation", "configured_source_evaluation"):
            for metric in ("accuracy", "ece"):
                value = episode[section][metric]
                if not 0.0 <= value <= 1.0:
                    findings.append(f"{binding['job_id']}: {section}.{metric} outside [0,1]")
        for metric in ("accuracy_before_return_adaptation", "accuracy_after_return_adaptation"):
            value = episode["return_outcome"][metric]
            if not 0.0 <= value <= 1.0:
                findings.append(f"{binding['job_id']}: return_outcome.{metric} outside [0,1]")
        if shard["method"] == "source":
            source_checked += 1
            required_zero = (
                episode["configuration_gap"],
                episode["prospective_mechanism"]["gradient_norm_before_update"],
                episode["prospective_mechanism"]["update_norm"],
                episode["prospective_mechanism"]["parameter_drift"],
                episode["return_outcome"]["absolute_hysteresis_before_return_adaptation"],
                episode["return_outcome"]["update_induced_hysteresis_before_return_adaptation"],
            )
            if any(value != 0.0 for value in required_zero):
                findings.append(f"{binding['job_id']}: source immutability invariant failed")
    if checked < 5:
        findings.append("fewer than five formal shards are available for early sanity")
    if source_checked == 0:
        findings.append("early sanity has no source shard")
    return {
        "status": "PASS" if not findings else "FAIL",
        "checked_shards": checked,
        "source_shards": source_checked,
        "methods_observed": sorted(methods),
        "findings": findings,
    }


def scale_out_decision(assessment: dict[str, Any]) -> dict[str, Any]:
    required = ("semantic_canary", "temporal_availability", "provenance_binding", "early_sanity_audit")
    failed = [name for name in required if assessment.get(name, {}).get("status") != "PASS"]
    return {
        "status": "FORMAL_ADMISSION_PASS" if not failed else "FORMAL_ADMISSION_FAIL",
        "failed_gates": failed,
        "scale_out_authorized": not failed,
    }


def audit_current(root: Path = ROOT) -> dict[str, Any]:
    runner_path = root / "expansion_experiment.py"
    supervisor_path = root / "run_expansion_supervisor.py"
    manifest_path = root / "configs" / "expansion_campaign_v2.json"
    dataset_path = root / "configs" / "cifar_dataset_provenance_v2.json"
    protocol_path = root / "docs" / "formal_protocol_expansion_v2_1.md"
    status_path = root / "run_state" / "expansion_formal_v2_status.json"
    result_root = root / "results" / "expansion" / "formal_v2"
    contracts_path = root / "configs" / "tta_method_state_contracts.json"
    v1_path = root / "configs" / "formal_v1_integrity_regression.json"
    manifest = load_json(manifest_path)
    status = load_json(status_path)
    checkpoint_inventory = freeze_checkpoint_inventory(root, manifest)
    expected = {
        "runner_sha256": sha256(runner_path),
        "supervisor_sha256": sha256(supervisor_path),
        "manifest_sha256": sha256(manifest_path),
        "dataset_provenance_sha256": sha256(dataset_path),
    }
    bindings = audit_shard_bindings(status, result_root, expected=expected, manifest=manifest)
    semantic = run_semantic_canary(runner_path, load_json(contracts_path))
    temporal = {
        "status": "PASS" if not [item for item in bindings["findings"] if "time" in item or "pre-update" in item] else "FAIL",
        "checked_shards": bindings.get("checked_shards", 0),
        "findings": [item for item in bindings["findings"] if "time" in item or "pre-update" in item],
    }
    early = early_sanity_audit(status, result_root, bindings)
    environments = []
    for item in bindings.get("bindings", []):
        shard = load_json(result_root / f"{item['job_id']}.json")
        environments.append(shard["environment"])
        key = f"{shard.get('dataset')}:s{int(shard.get('seed')):02d}"
        expected_checkpoint = checkpoint_inventory["checkpoint_sha256s"].get(key)
        if expected_checkpoint is None:
            bindings["findings"].append(f"{item['job_id']}: checkpoint is absent from the frozen inventory")
        elif shard.get("source_checkpoint_sha256") != expected_checkpoint:
            bindings["findings"].append(f"{item['job_id']}: source checkpoint hash mismatch")
    bindings["findings"].extend(checkpoint_inventory["findings"])
    bindings["status"] = "PASS" if not bindings["findings"] else "FAIL"
    environment_identity = {
        "observed_environments": sorted({json.dumps(item, sort_keys=True) for item in environments}),
        "requirements_lock_sha256": sha256(root / "requirements-cuda-lock.txt"),
        "venv_configuration_sha256": sha256(root / ".venv-cuda" / "pyvenv.cfg"),
    }
    bindings["protocol_sha256"] = sha256(protocol_path)
    bindings["environment_sha256"] = json_sha256(environment_identity)
    bindings["checkpoint_sha256s"] = checkpoint_inventory["checkpoint_sha256s"]
    v1 = load_json(v1_path)
    v1_findings = validate_v1_fixture(v1)
    assessment = {
        "schema_version": "1.0.0",
        "campaign_id": "formal_v2",
        "evaluated_at": utc(),
        "admission_mode": "RETROSPECTIVE_FOR_ALREADY_RUNNING_FROZEN_CAMPAIGN",
        "formal_v1_regression": {"status": "PASS" if not v1_findings else "FAIL", "findings": v1_findings, "fixture_sha256": sha256(v1_path)},
        "source_immutability": {
            "status": "PASS" if semantic["status"] == "PASS" and all(case["status"] == "PASS" for case in semantic["cases"] if case["method"] == "source") else "FAIL",
            "canary_source_cases": len([case for case in semantic["cases"] if case["method"] == "source"]),
        },
        "method_state_transition_contracts": {"status": semantic["status"], "contracts_sha256": sha256(contracts_path)},
        "semantic_canary": semantic,
        "temporal_availability": temporal,
        "provenance_binding": bindings,
        "early_sanity_audit": early,
        "frozen_identities": {
            **expected,
            "protocol_sha256": sha256(protocol_path),
            "environment_sha256": json_sha256(environment_identity),
            "checkpoint_sha256s": checkpoint_inventory["checkpoint_sha256s"],
        },
    }
    assessment["scale_out"] = scale_out_decision(assessment)
    assessment["formal_campaign_admission"] = (
        "RETROSPECTIVELY_VERIFIED" if assessment["scale_out"]["scale_out_authorized"] and not v1_findings else "FAIL"
    )
    assessment["restart_required"] = assessment["formal_campaign_admission"] != "RETROSPECTIVELY_VERIFIED"
    return assessment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the scientific-semantic admission gate for a formal campaign.")
    parser.add_argument("command", choices=("audit-current", "check-admission"))
    parser.add_argument("--output", type=Path, default=ROOT / "run_state" / "formal_v2_admission.json")
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args(argv)
    if args.command == "audit-current":
        result = audit_current(ROOT)
        write_json(args.output, result)
        print(json.dumps({
            "formal_campaign_admission": result["formal_campaign_admission"],
            "restart_required": result["restart_required"],
            "output": str(args.output),
        }))
        return 0 if not result["restart_required"] else 1
    path = args.admission or args.output
    result = load_json(path)
    current = audit_current(ROOT)
    ok = (
        result.get("formal_campaign_admission") == "RETROSPECTIVELY_VERIFIED"
        and result.get("frozen_identities") == current.get("frozen_identities")
        and current.get("scale_out", {}).get("scale_out_authorized") is True
    )
    print(json.dumps({"status": "PASS" if ok else "FAIL", "admission": str(path)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
