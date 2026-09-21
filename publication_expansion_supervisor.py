from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REQUIRED_GATES = (
    "source_immutability",
    "method_state_transition",
    "temporal_availability",
    "provenance_binding",
    "semantic_canary",
    "early_sanity_audit",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def campaign_result_set(campaign: dict[str, Any], smoke: bool, canary: bool = False) -> str:
    if canary:
        return str(campaign.get("canary_result_set", f"{campaign['result_set']}_canary"))
    return str(campaign["smoke_result_set"] if smoke else campaign["result_set"])


def enumerate_jobs(name: str, campaign: dict[str, Any], *, smoke: bool = False, canary: bool = False) -> list[dict[str, Any]]:
    result_set = campaign_result_set(campaign, smoke, canary)
    methods = list(campaign.get("methods", ["source", "tent"]))
    trajectories = list(campaign.get("trajectories", ["A-B-A", "A-B-C-A"]))
    seeds = [int(x) for x in campaign.get("seeds", campaign.get("fresh_seeds", [0, 1]))]
    datasets = list(campaign.get("datasets", ["cifar10"]))
    policies = list(campaign.get("policies", ["bn_affine_frozen_stats"]))
    pass_counts = list(campaign.get("pass_counts", [1]))
    learning_rates = list(campaign.get("learning_rates", [0.001]))
    severities = list(campaign.get("severities", [1]))
    if smoke or canary:
        seeds, trajectories, severities = seeds[:2], trajectories[:2], severities[:1]
        # Normalization canaries must exercise every registered policy; a
        # first-policy shortcut would make the admission gate scientifically
        # blind to the very factor under test.
        if not (canary and name == "normalization_ablation_v1"):
            policies = policies[:1]
        pass_counts, learning_rates = pass_counts[:1], learning_rates[:1]
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                for trajectory in trajectories:
                    for severity in severities:
                        for policy in policies:
                            for pass_count in pass_counts:
                                for learning_rate in learning_rates:
                                    job_id = f"{name}_{dataset}_s{seed:02d}_{method}_{trajectory.replace('-', '')}_{policy}_p{pass_count}_lr{learning_rate:g}_v{severity}"
                                    output = ROOT / "results" / "publication_expansion" / result_set / f"{job_id}.json"
                                    # Every normalization-ablation cell is bound to an immutable,
                                    # policy-specific checkpoint so the factor identity is auditable.
                                    policy_checkpoint_required = name == "normalization_ablation_v1"
                                    policy_checkpoint_path = ROOT / "checkpoints" / "expansion" / "normalization" / f"{dataset}_s{seed:02d}_{policy}.pt" if policy_checkpoint_required else ROOT / "checkpoints" / "expansion" / "formal" / f"{dataset}_s{seed:02d}.pt"
                                    jobs.append({"job_id": job_id, "campaign_id": name, "result_set": result_set, "dataset": dataset, "seed": seed, "method": method, "trajectory": trajectory, "shift_family": "noise", "severity": severity, "normalization_policy": policy, "policy_checkpoint_required": policy_checkpoint_required, "policy_checkpoint_path": str(policy_checkpoint_path), "pass_count": pass_count, "learning_rate": learning_rate, "output": str(output)})
    return jobs


def scale_out_decision(assessment: dict[str, Any]) -> dict[str, Any]:
    failed = [gate for gate in REQUIRED_GATES if assessment.get(gate, {}).get("status") != "PASS"]
    return {"status": "FORMAL_ADMISSION_PASS" if not failed else "FORMAL_ADMISSION_FAIL", "failed_gates": failed, "scale_out_authorized": not failed}


def validate_canary_admission(
    admission_path: Path, campaign_id: str, expected_provenance: dict[str, str]
) -> dict[str, Any]:
    """Validate the immutable canary ledger before allowing formal scale-out."""
    findings: list[str] = []
    try:
        ledger = json.loads(admission_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FORMAL_ADMISSION_FAIL", "failed_gates": ["semantic_canary"], "scale_out_authorized": False, "findings": [f"unreadable admission ledger: {exc}"]}
    if ledger.get("campaign_id") != campaign_id:
        findings.append(f"admission campaign mismatch: {ledger.get('campaign_id')!r} != {campaign_id!r}")
    if ledger.get("status") != "FORMAL_ADMISSION_PASS":
        findings.append(f"admission status is {ledger.get('status')!r}")
    if ledger.get("provenance") != expected_provenance:
        findings.append("admission provenance does not match current frozen identities")
    gates = ledger.get("gates", {})
    failed = [gate for gate in REQUIRED_GATES if gates.get(gate, {}).get("status") != "PASS"]
    if failed:
        findings.append(f"admission gates are not all PASS: {failed}")
    return {
        "status": "FORMAL_ADMISSION_PASS" if not findings else "FORMAL_ADMISSION_FAIL",
        "failed_gates": failed,
        "scale_out_authorized": not findings,
        "findings": findings,
        "admission_path": str(admission_path),
    }


def finalize_canary_admission(
    status_path: Path,
    result_root: Path,
    state: dict[str, Any],
    campaign: dict[str, Any],
    provenance: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    """Persist terminal canary state before deriving its admission ledger."""
    from publication_campaign_admission import build_admission_ledger

    completed_state = dict(state)
    completed_state["status"] = "COMPLETED"
    completed_state["semantic_canary_passed"] = False
    completed_state["finished_at"] = utc()
    write_json(status_path, completed_state)

    admission_path = status_path.parent / f"publication_{completed_state.get('result_set', campaign.get('result_set', completed_state.get('campaign_id')))}_admission.json"
    admission = build_admission_ledger(
        status_path, result_root, campaign, expected_provenance=provenance
    )
    admission["runner_sha256"] = provenance.get("runner_sha256")
    write_json(admission_path, admission)
    completed_state["admission_path"] = str(admission_path)
    completed_state["admission_status"] = admission["status"]
    completed_state["semantic_canary_passed"] = admission["status"] == "FORMAL_ADMISSION_PASS"
    completed_state["status"] = "COMPLETED" if completed_state["semantic_canary_passed"] else "CANARY_ADMISSION_FAIL"
    write_json(status_path, completed_state)
    return completed_state, admission_path


def resume_plan(all_jobs: list[dict[str, Any]], prior: dict[str, Any], result_root: Path, *, expected_provenance: dict[str, str] | None = None) -> dict[str, Any]:
    if expected_provenance and prior:
        for field, expected in expected_provenance.items():
            if prior.get(field) != expected:
                raise ValueError(f"provenance mismatch for {field}: prior={prior.get(field)!r}, current={expected!r}")
    completed = prior.get("completed", {})
    valid: dict[str, str] = {}
    if isinstance(completed, dict):
        for job_id, expected_hash in completed.items():
            path = result_root / f"{job_id}.json"
            if path.is_file() and sha256(path) == expected_hash:
                valid[job_id] = expected_hash
    return {"pending": [job for job in all_jobs if job["job_id"] not in valid], "completed": valid, "preserved_failures": list(prior.get("preserved_failures", [])) + list(prior.get("failed", []))}


def _run_one(
    job: dict[str, Any],
    checkpoint: Path,
    provenance: str | None,
    device: str,
    smoke: bool,
    canary: bool,
    manifest_sha256: str,
    supervisor_sha256: str,
) -> dict[str, Any]:
    output = Path(job["output"])
    command = [sys.executable, str(ROOT / "publication_expansion_experiment.py"), "run", "--job-id", job["job_id"], "--campaign-id", job["campaign_id"], "--dataset", job["dataset"], "--seed", str(job["seed"]), "--method", job["method"], "--trajectory", job["trajectory"], "--shift-family", job["shift_family"], "--severity", str(job["severity"]), "--source-checkpoint", str(checkpoint), "--output", str(output), "--batch-size", "128", "--adaptation-passes", str(job["pass_count"]), "--pass-count", str(job["pass_count"]), "--learning-rate", str(job["learning_rate"]), "--normalization-policy", job["normalization_policy"], "--device", device]
    if provenance:
        command += ["--dataset-provenance-sha256", provenance]
    command += ["--manifest-sha256", manifest_sha256, "--supervisor-sha256", supervisor_sha256]
    if smoke:
        command += ["--max-test-samples", "256", "--smoke"]
    if canary:
        command += ["--max-test-samples", "256", "--canary"]
    log_dir = ROOT / "logs" / "publication_expansion" / job["result_set"] / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (log_dir / f"{job['job_id']}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{job['job_id']}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {"ok": proc.returncode == 0, "job": job, "output": output, "sha256": sha256(output) if output.is_file() else None, "error": (proc.stderr or proc.stdout)[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "publication_expansion_campaigns.json")
    parser.add_argument("--status", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    campaign = config[args.campaign]
    if args.smoke and args.canary:
        parser.error("--smoke and --canary are mutually exclusive")
    jobs = enumerate_jobs(args.campaign, campaign, smoke=args.smoke, canary=args.canary)
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    result_set = campaign_result_set(campaign, args.smoke, args.canary)
    status_path = args.status or ROOT / "run_state" / f"publication_{result_set}_status.json"
    result_root = ROOT / "results" / "publication_expansion" / result_set
    provenance_file = ROOT / "configs" / "cifar_dataset_provenance_v2.json"
    provenance = {"manifest_sha256": sha256(args.config), "runner_sha256": sha256(ROOT / "publication_expansion_experiment.py"), "supervisor_sha256": sha256(Path(__file__)), "dataset_provenance_sha256": sha256(provenance_file) if provenance_file.is_file() else None, "method_registry_sha256": sha256(ROOT / "configs" / "modern_baseline_upstreams.json"), "method_contracts_sha256": sha256(ROOT / "configs" / "tta_method_state_contracts.json")}
    prior = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.is_file() else {}
    plan = resume_plan(jobs, prior, result_root, expected_provenance=provenance if args.resume else None)
    state: dict[str, Any] = {"schema_version": "1.0.0", "status": "RUNNING", "campaign_id": args.campaign, "result_set": result_set, "evidence_label": "SEMANTIC_CANARY" if args.canary else ("SMOKE" if args.smoke else "FORMAL_PUBLICATION_EXPANSION"), "started_at": prior.get("started_at", utc()), "last_heartbeat": utc(), "manifest": str(args.config), **provenance, "device": args.device, "python": sys.executable, "host": {"python": platform.python_version(), "platform": platform.platform()}, "total_jobs": len(jobs), "completed": plan["completed"], "failed": [], "preserved_failures": plan["preserved_failures"], "remaining_jobs": len(plan["pending"]), "semantic_canary_passed": False}
    write_json(status_path, state)
    if not args.smoke and not args.canary:
        canary_result_set = campaign_result_set(campaign, smoke=False, canary=True)
        admission_path = ROOT / "run_state" / f"publication_{canary_result_set}_admission.json"
        admission = validate_canary_admission(admission_path, args.campaign, provenance)
        if not admission["scale_out_authorized"]:
            state.update({"status": "BLOCKED_CANARY_REQUIRED", "admission": admission})
            write_json(status_path, state)
            return 2
        state["semantic_canary_passed"] = True
        state["admission"] = admission
        write_json(status_path, state)
    source_checkpoints = {(j["dataset"], j["seed"]): ROOT / "checkpoints" / "expansion" / "formal" / f"{j['dataset']}_s{j['seed']:02d}.pt" for j in plan["pending"]}
    checkpoints: dict[tuple[str, int, str], Path] = {}
    if args.campaign == "normalization_ablation_v1":
        from publication_expansion_experiment import materialize_normalization_checkpoint

        for job in plan["pending"]:
            source_path = source_checkpoints[(job["dataset"], job["seed"])]
            target_path = Path(job["policy_checkpoint_path"])
            if source_path.is_file():
                materialize_normalization_checkpoint(
                    source_path, target_path, job["normalization_policy"]
                )
            checkpoints[(job["dataset"], job["seed"], job["normalization_policy"])] = target_path
    else:
        for job in plan["pending"]:
            checkpoints[(job["dataset"], job["seed"], job["normalization_policy"])] = source_checkpoints[(job["dataset"], job["seed"])]
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        state.update({"status": "BLOCKED_MISSING_CHECKPOINT", "missing_checkpoints": missing})
        write_json(status_path, state)
        return 2
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run_one, job, checkpoints[(job["dataset"], job["seed"], job["normalization_policy"])], provenance["dataset_provenance_sha256"], args.device, args.smoke, args.canary, provenance["manifest_sha256"], provenance["supervisor_sha256"]): job for job in plan["pending"]}
        for future in as_completed(futures):
            result = future.result()
            if result["ok"] and result["sha256"]:
                state["completed"][result["job"]["job_id"]] = result["sha256"]
            else:
                state["failed"].append({"job_id": result["job"]["job_id"], "error": result["error"]})
            state["remaining_jobs"] = len(jobs) - len(state["completed"])
            state["last_heartbeat"] = utc()
            write_json(status_path, state)
    state["status"] = "COMPLETED" if not state["failed"] and len(state["completed"]) == len(jobs) else "PARTIAL"
    if args.canary and state["status"] == "COMPLETED":
        state, admission_path = finalize_canary_admission(
            status_path, result_root, state, campaign, provenance
        )
    else:
        state["finished_at"] = utc()
        write_json(status_path, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"]), "status_path": str(status_path)}))
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
