from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from publication_campaign_admission import REQUIRED_GATES, inspect_shard


ROOT = Path(__file__).resolve().parent
PROVENANCE_FIELDS = (
    "runner_sha256",
    "supervisor_sha256",
    "manifest_sha256",
    "dataset_provenance_sha256",
    "external_source_registry_sha256",
    "clean_dataset_provenance_sha256",
    "method_engine_sha256",
    "method_registry_sha256",
    "method_contracts_sha256",
    "checkpoint_inventory_sha256",
    "environment_sha256",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def freeze_checkpoint_inventory(
    campaign: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for dataset in campaign["datasets"]:
        base_dataset = str(dataset).replace("_c", "")
        for seed_value in campaign["seeds"]:
            seed = int(seed_value)
            path = root / "checkpoints" / "expansion" / "formal" / f"{base_dataset}_s{seed:02d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing source checkpoint: {path}")
            files[f"{dataset}:s{seed:02d}"] = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {"files": files, "sha256": json_digest(files)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _paired_corruption(corruptions: list[str], corruption: str) -> str:
    index = corruptions.index(corruption)
    return corruptions[(index + 1) % len(corruptions)]


def _job(
    campaign: dict[str, Any],
    *,
    result_set: str,
    dataset: str,
    method: str,
    seed: int,
    trajectory: str,
    severity: int,
    corruption: str,
    paired_corruption: str,
) -> dict[str, Any]:
    job_id = (
        f"{campaign['campaign_id']}_{dataset}_s{seed:02d}_{method}_"
        f"{trajectory.replace('-', '')}_{corruption}_c{paired_corruption}_v{severity}"
    )
    output = ROOT / "results" / "external_validation" / result_set / f"{job_id}.json"
    return {
        "job_id": job_id,
        "campaign_id": campaign["campaign_id"],
        "result_set": result_set,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "trajectory": trajectory,
        "severity": severity,
        "corruption": corruption,
        "paired_corruption": paired_corruption,
        "output": str(output),
    }


def enumerate_formal_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    corruptions = list(campaign["corruptions"])
    for dataset in campaign["datasets"]:
        for corruption in corruptions:
            for severity in campaign["severities"]:
                for seed in campaign["seeds"]:
                    for method in campaign["methods"]:
                        for trajectory in campaign["formal_trajectories"]:
                            jobs.append(
                                _job(
                                    campaign,
                                    result_set=campaign["result_set"],
                                    dataset=dataset,
                                    method=method,
                                    seed=int(seed),
                                    trajectory=trajectory,
                                    severity=int(severity),
                                    corruption=corruption,
                                    paired_corruption=_paired_corruption(corruptions, corruption),
                                )
                            )
    return jobs


def enumerate_canary_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seeds = list(campaign["canary_seeds"])
    trajectories = list(campaign["canary_trajectories"])
    for dataset in campaign["datasets"]:
        for method in campaign["methods"]:
            combinations = [
                (int(seed), trajectory)
                for seed in seeds
                for trajectory in trajectories
            ]
            for index, (seed, trajectory) in enumerate(combinations):
                case = campaign["canary_representatives"][index]
                jobs.append(
                    _job(
                        campaign,
                        result_set=campaign["canary_result_set"],
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        trajectory=trajectory,
                        severity=int(case["severity"]),
                        corruption=case["corruption"],
                        paired_corruption=case["paired_corruption"],
                    )
                )
    return jobs


def enumerate_smoke_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    case = campaign["canary_representatives"][0]
    return [
        _job(
            campaign,
            result_set=campaign["smoke_result_set"],
            dataset=dataset,
            method=method,
            seed=int(campaign["canary_seeds"][0]),
            trajectory="A-B-A",
            severity=int(case["severity"]),
            corruption=case["corruption"],
            paired_corruption=case["paired_corruption"],
        )
        for dataset in campaign["datasets"]
        for method in campaign["methods"]
    ]


def resume_plan(
    jobs: list[dict[str, Any]],
    prior: dict[str, Any],
    result_root: Path,
    *,
    expected_provenance: dict[str, str],
) -> dict[str, Any]:
    if prior.get("status") == "BLOCKED_EXTERNAL_DATA" and not prior.get("completed"):
        prior = {}
    if prior.get("status") == "EARLY_SANITY_FAIL":
        raise ValueError(
            "refusing automatic resume after integrity failure; inspect the audit and use a new campaign identity if required"
        )
    if prior:
        for field, expected in expected_provenance.items():
            if prior.get(field) != expected:
                raise ValueError(
                    f"provenance mismatch for {field}: prior={prior.get(field)!r}, current={expected!r}"
                )
    valid: dict[str, str] = {}
    for job_id, expected_hash in prior.get("completed", {}).items():
        path = result_root / f"{job_id}.json"
        if path.is_file() and sha256(path) == expected_hash:
            valid[job_id] = expected_hash
    return {
        "completed": valid,
        "pending": [job for job in jobs if job["job_id"] not in valid],
        "preserved_failures": list(prior.get("preserved_failures", []))
        + list(prior.get("failed", [])),
    }


def external_coverage_findings(
    shards: list[dict[str, Any]], campaign: dict[str, Any]
) -> list[str]:
    findings: list[str] = []
    expected = {
        "dataset": set(campaign["datasets"]),
        "method": set(campaign["methods"]),
        "seed": set(int(value) for value in campaign["canary_seeds"]),
        "trajectory": set(campaign["canary_trajectories"]),
        "severity": set(int(value) for value in campaign["canary_severities"]),
    }
    for field, required in expected.items():
        observed = {shard.get(field) for shard in shards}
        missing = required - observed
        if missing:
            findings.append(f"canary {field} coverage missing: {sorted(missing)}")
    observed_cells = {
        (
            shard.get("dataset"),
            shard.get("method"),
            shard.get("seed"),
            shard.get("trajectory"),
        )
        for shard in shards
    }
    required_cells = {
        (dataset, method, int(seed), trajectory)
        for dataset in campaign["datasets"]
        for method in campaign["methods"]
        for seed in campaign["canary_seeds"]
        for trajectory in campaign["canary_trajectories"]
    }
    missing_cells = required_cells - observed_cells
    if missing_cells:
        findings.append(f"canary method-seed-trajectory cells missing: {len(missing_cells)}")
    categories = campaign["corruption_categories"]
    observed_categories = {
        categories.get(str(shard.get("corruption"))) for shard in shards
    }
    missing_categories = {"noise", "blur", "weather", "digital"} - observed_categories
    if missing_categories:
        findings.append(f"canary corruption category coverage missing: {sorted(missing_categories)}")
    for shard in shards:
        identity = shard.get("dataset_identity", {})
        if (
            identity.get("protocol") != "standard_cifar_c"
            or identity.get("data_source") != "REAL_OFFICIAL_CIFAR_C"
        ):
            findings.append(
                f"{shard.get('job_id', 'unknown')}: external evidence is not real standard_cifar_c"
            )
        if identity.get("external_dataset") != shard.get("dataset"):
            findings.append(f"{shard.get('job_id', 'unknown')}: dataset identity mismatch")
        for field in (
            "archive_sha256",
            "labels_file_sha256",
            "corruption_file_sha256",
            "paired_corruption_file_sha256",
        ):
            value = identity.get(field)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                findings.append(
                    f"{shard.get('job_id', 'unknown')}: missing dataset identity hash {field}"
                )
    return findings


def audit_shards(
    completed: dict[str, str],
    result_root: Path,
    expected_provenance: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    shards: list[dict[str, Any]] = []
    findings: list[str] = []
    for job_id, expected_hash in sorted(completed.items()):
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"missing shard: {job_id}")
            continue
        if sha256(path) != expected_hash:
            findings.append(f"result hash mismatch: {job_id}")
            continue
        inspected = inspect_shard(path, expected_provenance=expected_provenance)
        findings.extend(f"{job_id}: {value}" for value in inspected["findings"])
        shard = inspected["shard"]
        for field in PROVENANCE_FIELDS:
            if shard.get(field) != expected_provenance.get(field):
                findings.append(f"{job_id}: provenance binding violation: {field}")
        immutability = shard.get("episode", {}).get("external_source_model_immutability", {})
        if immutability.get("status") != "PASS" or immutability.get("before_digest") != immutability.get("after_digest"):
            findings.append(f"{job_id}: external source model immutability failed")
        if shard.get("method") == "source":
            transition = shard.get("episode", {}).get("state_transition", {})
            if transition.get("before_digest") != transition.get("after_digest"):
                findings.append(f"{job_id}: source state digest changed")
            if transition.get("optimizer_step_count") != 0:
                findings.append(f"{job_id}: source optimizer step count is not zero")
            drift = shard.get("episode", {}).get("prospective_mechanism", {}).get("parameter_drift")
            if drift != 0.0:
                findings.append(f"{job_id}: source parameter drift is nonzero")
        shards.append(shard)
    return shards, findings


def build_admission(
    state: dict[str, Any],
    result_root: Path,
    campaign: dict[str, Any],
    provenance: dict[str, str],
) -> dict[str, Any]:
    shards, audit_findings = audit_shards(state["completed"], result_root, provenance)
    coverage_findings = external_coverage_findings(shards, campaign)
    source_shards = [value for value in shards if value.get("method") == "source"]
    source_findings = [
        value for value in audit_findings if "source" in value.lower() or "optimizer step count" in value
    ]
    temporal_findings = [value for value in audit_findings if "temporal" in value]
    provenance_findings = [
        value for value in audit_findings
        if "provenance" in value or "hash mismatch" in value or "missing shard" in value
    ]
    method_findings = [
        value for value in audit_findings
        if value not in source_findings + temporal_findings + provenance_findings
    ]
    gates = {
        "source_immutability": {
            "status": "PASS" if source_shards and not source_findings else "FAIL",
            "checked_shards": len(source_shards),
            "findings": source_findings,
        },
        "method_state_transition": {
            "status": "PASS" if not method_findings and bool(shards) else "FAIL",
            "findings": method_findings,
        },
        "temporal_availability": {
            "status": "PASS" if not temporal_findings and bool(shards) else "FAIL",
            "findings": temporal_findings,
        },
        "provenance_binding": {
            "status": "PASS" if not provenance_findings and bool(shards) else "FAIL",
            "findings": provenance_findings,
            "expected": provenance,
        },
        "semantic_canary": {
            "status": "PASS" if not coverage_findings and state.get("status") == "COMPLETED" else "FAIL",
            "checked_shards": len(shards),
            "findings": coverage_findings,
        },
        "early_sanity_audit": {
            "status": "PASS" if not audit_findings and bool(shards) else "FAIL",
            "checked_shards": len(shards),
            "findings": audit_findings,
        },
    }
    failed = [name for name in REQUIRED_GATES if gates[name]["status"] != "PASS"]
    return {
        "schema_version": "1.0.0",
        "campaign_id": campaign["campaign_id"],
        "result_set": state["result_set"],
        "status": "FORMAL_ADMISSION_PASS" if not failed else "FORMAL_ADMISSION_FAIL",
        "scale_out_authorized": not failed,
        "gates": gates,
        "failed_gates": failed,
        "provenance": provenance,
        "created_at": utc(),
    }


def validate_admission(
    path: Path, campaign_id: str, provenance: dict[str, str]
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FORMAL_ADMISSION_FAIL", "scale_out_authorized": False, "findings": [str(exc)]}
    findings: list[str] = []
    if ledger.get("campaign_id") != campaign_id:
        findings.append("campaign ID mismatch")
    if ledger.get("status") != "FORMAL_ADMISSION_PASS":
        findings.append("canary admission did not pass")
    if ledger.get("provenance") != provenance:
        findings.append("frozen provenance changed after canary")
    failed = [
        name for name in REQUIRED_GATES
        if ledger.get("gates", {}).get(name, {}).get("status") != "PASS"
    ]
    if failed:
        findings.append(f"required gates failed: {failed}")
    return {
        "status": "FORMAL_ADMISSION_PASS" if not findings else "FORMAL_ADMISSION_FAIL",
        "scale_out_authorized": not findings,
        "findings": findings,
    }


def _run_one(
    job: dict[str, Any],
    campaign: dict[str, Any],
    provenance: dict[str, str],
    *,
    data_root: Path,
    dataset_provenance: Path,
    device: str,
    max_test_samples: int,
    smoke: bool,
    canary: bool,
    checkpoint_inventory: dict[str, Any],
) -> dict[str, Any]:
    base_dataset = job["dataset"].replace("_c", "")
    checkpoint = ROOT / "checkpoints" / "expansion" / "formal" / f"{base_dataset}_s{job['seed']:02d}.pt"
    checkpoint_key = f"{job['dataset']}:s{job['seed']:02d}"
    expected_checkpoint_sha256 = checkpoint_inventory["files"][checkpoint_key]["sha256"]
    command = [
        sys.executable,
        str(ROOT / "external_validation_experiment.py"),
        "--job-id", job["job_id"],
        "--campaign-id", job["campaign_id"],
        "--dataset", job["dataset"],
        "--corruption", job["corruption"],
        "--paired-corruption", job["paired_corruption"],
        "--severity", str(job["severity"]),
        "--seed", str(job["seed"]),
        "--method", job["method"],
        "--trajectory", job["trajectory"],
        "--source-checkpoint", str(checkpoint),
        "--data-root", str(data_root),
        "--dataset-provenance", str(dataset_provenance),
        "--output", job["output"],
        "--batch-size", str(campaign["batch_size"]),
        "--pass-count", str(campaign["pass_count"]),
        "--learning-rate", str(campaign["learning_rate"]),
        "--normalization-policy", campaign["normalization_policy"],
        "--device", device,
        "--expected-source-checkpoint-sha256", expected_checkpoint_sha256,
    ]
    for field in (
        "manifest_sha256",
        "supervisor_sha256",
        "dataset_provenance_sha256",
        "external_source_registry_sha256",
        "clean_dataset_provenance_sha256",
        "method_engine_sha256",
        "checkpoint_inventory_sha256",
        "environment_sha256",
    ):
        command += ["--" + field.replace("_", "-"), provenance[field]]
    if max_test_samples:
        command += ["--max-test-samples", str(max_test_samples)]
    if smoke:
        command.append("--smoke")
    if canary:
        command.append("--canary")
    log_root = ROOT / "logs" / "external_validation" / job["result_set"] / "jobs"
    log_root.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (log_root / f"{job['job_id']}.stdout.log").write_text(process.stdout, encoding="utf-8")
    (log_root / f"{job['job_id']}.stderr.log").write_text(process.stderr, encoding="utf-8")
    output = Path(job["output"])
    return {
        "ok": process.returncode == 0 and output.is_file(),
        "job": job,
        "sha256": sha256(output) if output.is_file() else None,
        "error": (
            (process.stderr or process.stdout)[-4000:]
            or f"child process exited with return code {process.returncode}"
        ),
    }


def run_bounded(
    jobs: list[dict[str, Any]],
    submit_job,
    *,
    workers: int,
):
    """Yield completed jobs while keeping at most ``workers`` in flight."""
    iterator = iter(jobs)
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures: dict[Any, dict[str, Any]] = {}
    try:
        for _ in range(max(1, workers)):
            try:
                job = next(iterator)
            except StopIteration:
                break
            futures[pool.submit(submit_job, job)] = job
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                futures.pop(future)
                yield future.result()
                try:
                    job = next(iterator)
                except StopIteration:
                    continue
                futures[pool.submit(submit_job, job)] = job
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/external_validation_cifar_c_v1.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--job-list",
        type=Path,
        default=None,
        help="restrict execution to an explicit JSON list of job IDs (recovery campaigns only)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    campaign = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = (
        enumerate_smoke_jobs(campaign) if args.smoke
        else enumerate_canary_jobs(campaign) if args.canary
        else enumerate_formal_jobs(campaign)
    )
    if args.job_list is not None:
        if not args.formal:
            raise ValueError("--job-list is only valid for formal campaigns")
        requested = json.loads(args.job_list.read_text(encoding="utf-8"))
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ValueError("job list must be a JSON array of job IDs")
        by_id = {job["job_id"]: job for job in jobs}
        missing = [job_id for job_id in requested if job_id not in by_id]
        if missing:
            raise ValueError(f"job list contains unknown IDs: {missing[:3]}")
        jobs = [by_id[job_id] for job_id in requested]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    result_set = (
        campaign["smoke_result_set"] if args.smoke
        else campaign["canary_result_set"] if args.canary
        else campaign["result_set"]
    )
    status_path = ROOT / "run_state" / f"{result_set}_status.json"
    result_root = ROOT / "results" / "external_validation" / result_set
    dataset_provenance = ROOT / campaign["dataset_provenance"]
    data_root = ROOT / campaign["data_root"]
    required_files = {
        "dataset_provenance": dataset_provenance,
        "external_source_registry": ROOT / campaign["external_source_registry"],
        "clean_dataset_provenance": ROOT / "configs/cifar_dataset_provenance_v2.json",
        "method_engine": ROOT / "publication_expansion_experiment.py",
        "method_registry": ROOT / "configs/modern_baseline_upstreams.json",
        "method_contracts": ROOT / "configs/tta_method_state_contracts.json",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        write_json(status_path, {
            "schema_version": "1.0.0",
            "status": "BLOCKED_EXTERNAL_DATA",
            "campaign_id": campaign["campaign_id"],
            "result_set": result_set,
            "missing": missing,
            "created_at": utc(),
        })
        print(json.dumps({"status": "BLOCKED_EXTERNAL_DATA", "missing": missing}))
        return 2
    provenance = {
        "runner_sha256": sha256(ROOT / "external_validation_experiment.py"),
        "supervisor_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.config),
        "dataset_provenance_sha256": sha256(dataset_provenance),
        "external_source_registry_sha256": sha256(required_files["external_source_registry"]),
        "clean_dataset_provenance_sha256": sha256(required_files["clean_dataset_provenance"]),
        "method_engine_sha256": sha256(required_files["method_engine"]),
        "method_registry_sha256": sha256(required_files["method_registry"]),
        "method_contracts_sha256": sha256(required_files["method_contracts"]),
    }
    checkpoint_inventory = freeze_checkpoint_inventory(campaign)
    from external_validation_experiment import runtime_environment_identity

    environment_identity = runtime_environment_identity(args.device)
    provenance["checkpoint_inventory_sha256"] = checkpoint_inventory["sha256"]
    provenance["environment_sha256"] = json_digest(environment_identity)
    prior = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.is_file() else {}
    plan = resume_plan(jobs, prior, result_root, expected_provenance=provenance)
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "RUNNING",
        "campaign_id": campaign["campaign_id"],
        "result_set": result_set,
        "evidence_label": "SMOKE" if args.smoke else ("SEMANTIC_CANARY" if args.canary else "FORMAL_EXTERNAL_VALIDATION"),
        "started_at": prior.get("started_at", utc()),
        "last_heartbeat": utc(),
        **provenance,
        "device": args.device,
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "environment_identity": environment_identity,
        "checkpoint_inventory": checkpoint_inventory,
        "total_jobs": len(jobs),
        "completed": plan["completed"],
        "failed": [],
        "preserved_failures": plan["preserved_failures"],
        "remaining_jobs": len(plan["pending"]),
    }
    write_json(status_path, state)
    if args.formal:
        admission_path = ROOT / "run_state" / f"{campaign['canary_result_set']}_admission.json"
        admission = validate_admission(admission_path, campaign["campaign_id"], provenance)
        if not admission["scale_out_authorized"]:
            state.update({"status": "BLOCKED_CANARY_REQUIRED", "admission": admission})
            write_json(status_path, state)
            return 2
        state["admission"] = admission
        write_json(status_path, state)
    default_limit = 256 if args.smoke or args.canary else 0
    sample_limit = args.max_test_samples or default_limit
    def submit_job(job: dict[str, Any]) -> dict[str, Any]:
        return _run_one(
            job,
            campaign,
            provenance,
            data_root=data_root,
            dataset_provenance=dataset_provenance,
            device=args.device,
            max_test_samples=sample_limit,
            smoke=args.smoke,
            canary=args.canary,
            checkpoint_inventory=checkpoint_inventory,
        )

    pending_jobs = list(plan["pending"])
    bootstrap_count = min(12, len(pending_jobs)) if args.formal else 0
    phases = (
        [pending_jobs[:bootstrap_count], pending_jobs[bootstrap_count:]]
        if args.formal
        else [pending_jobs]
    )
    for phase_index, phase_jobs in enumerate(phases):
        for result in run_bounded(phase_jobs, submit_job, workers=args.workers):
            job = result["job"]
            if result["ok"] and result["sha256"]:
                state["completed"][job["job_id"]] = result["sha256"]
            else:
                state["failed"].append({"job_id": job["job_id"], "error": result["error"]})
            state["remaining_jobs"] = len(jobs) - len(state["completed"])
            state["last_heartbeat"] = utc()
            write_json(status_path, state)
            if args.formal and state["failed"]:
                audit = {
                    "schema_version": "1.0.0",
                    "status": "FAIL",
                    "checked_shards": len(state["completed"]),
                    "findings": [f"formal job failed: {job['job_id']}"],
                    "created_at": utc(),
                }
                write_json(ROOT / "run_state" / f"{result_set}_early_sanity_audit.json", audit)
                state.update({"status": "EARLY_SANITY_FAIL", "early_sanity_audit": audit})
                write_json(status_path, state)
                return 3
            if args.formal and phase_index == 0 and result["ok"]:
                _shards, single_findings = audit_shards(
                    {job["job_id"]: result["sha256"]}, result_root, provenance
                )
                if single_findings:
                    audit = {
                        "schema_version": "1.0.0",
                        "status": "FAIL",
                        "checked_shards": 1,
                        "findings": single_findings,
                        "created_at": utc(),
                    }
                    write_json(ROOT / "run_state" / f"{result_set}_early_sanity_audit.json", audit)
                    state.update({"status": "EARLY_SANITY_FAIL", "early_sanity_audit": audit})
                    write_json(status_path, state)
                    return 3
        if args.formal and phase_index == 0 and phase_jobs:
            _, findings = audit_shards(state["completed"], result_root, provenance)
            audit = {
                "schema_version": "1.0.0",
                "status": "PASS" if not findings else "FAIL",
                "checked_shards": len(state["completed"]),
                "findings": findings,
                "created_at": utc(),
            }
            write_json(ROOT / "run_state" / f"{result_set}_early_sanity_audit.json", audit)
            if findings:
                state.update({"status": "EARLY_SANITY_FAIL", "early_sanity_audit": audit})
                write_json(status_path, state)
                return 3
    state["status"] = "COMPLETED" if not state["failed"] and len(state["completed"]) == len(jobs) else "PARTIAL"
    state["finished_at"] = utc()
    write_json(status_path, state)
    if args.canary and state["status"] == "COMPLETED":
        admission = build_admission(state, result_root, campaign, provenance)
        admission_path = ROOT / "run_state" / f"{result_set}_admission.json"
        write_json(admission_path, admission)
        state["admission_path"] = str(admission_path)
        state["admission_status"] = admission["status"]
        state["status"] = "COMPLETED" if admission["scale_out_authorized"] else "CANARY_ADMISSION_FAIL"
        write_json(status_path, state)
    print(json.dumps({
        "status": state["status"],
        "completed": len(state["completed"]),
        "failed": len(state["failed"]),
        "status_path": str(status_path),
    }))
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
