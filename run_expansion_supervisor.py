from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def jobs(manifest: dict) -> list[dict]:
    values: list[dict] = []
    for dataset in manifest["datasets"]:
        for seed in manifest["seeds"]:
            for method in manifest["methods"]:
                for trajectory in manifest["trajectories"]:
                    for family in manifest["shift_families"]:
                        for severity in manifest["severities"]:
                            job_id = f"{dataset}_s{seed:02d}_{method}_{trajectory.replace('-', '')}_{family}_v{severity}"
                            values.append({"job_id": job_id, "dataset": dataset, "seed": seed, "method": method, "trajectory": trajectory, "shift_family": family, "severity": severity})
    return values


def campaign_result_set(manifest: dict, *, smoke: bool) -> str:
    value = str(manifest.get("smoke_result_set", "smoke")) if smoke else str(manifest.get("result_set", "formal"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe result_set: {value}")
    return value


def resume_plan(
    all_jobs: list[dict],
    prior: dict,
    result_root: Path,
    *,
    expected_provenance: dict[str, str | None] | None = None,
) -> dict:
    if expected_provenance and prior.get("completed"):
        for field, expected in expected_provenance.items():
            if prior.get(field) != expected:
                raise ValueError(
                    f"provenance mismatch for {field}: prior={prior.get(field)!r}, current={expected!r}; use a new result_set"
                )
    valid: set[str] = set()
    completed = prior.get("completed", {})
    for job_id, expected_hash in completed.items() if isinstance(completed, dict) else []:
        path = result_root / f"{job_id}.json"
        if path.is_file() and sha256(path) == expected_hash:
            valid.add(job_id)
    return {
        "pending": [job for job in all_jobs if job["job_id"] not in valid],
        "completed": {job_id: completed[job_id] for job_id in sorted(valid)},
        "preserved_failures": list(prior.get("preserved_failures", [])) + list(prior.get("failed", [])),
    }


def run_process(command: list[str], stdout: Path, stderr: Path, retries: int) -> dict:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    last_error = ""
    while attempts <= retries:
        attempts += 1
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        stdout.write_text(proc.stdout, encoding="utf-8")
        stderr.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode == 0:
            return {"ok": True, "attempts": attempts}
        last_error = (proc.stderr or proc.stdout or f"exit={proc.returncode}")[-4000:]
    return {"ok": False, "attempts": attempts, "error": last_error}


def ensure_source(dataset: str, seed: int, manifest: dict, *, smoke: bool, python: str, device: str) -> dict:
    suffix = "smoke" if smoke else "formal"
    checkpoint = ROOT / "checkpoints" / "expansion" / suffix / f"{dataset}_s{seed:02d}.pt"
    if checkpoint.is_file():
        return {"ok": True, "checkpoint": checkpoint, "sha256": sha256(checkpoint), "reused": True}
    command = [python, str(ROOT / "expansion_experiment.py"), "train-source", "--dataset", dataset, "--seed", str(seed), "--output", str(checkpoint), "--epochs", str(1 if smoke else manifest["model"]["source_epochs"]), "--batch-size", str(manifest["model"]["batch_size"]), "--max-samples", str(1024 if smoke else 0), "--device", device]
    log_root = ROOT / "logs" / "expansion" / suffix
    outcome = run_process(command, log_root / f"train_{dataset}_s{seed:02d}.stdout.log", log_root / f"train_{dataset}_s{seed:02d}.stderr.log", int(manifest.get("retries", 2)))
    return outcome | {"checkpoint": checkpoint, "sha256": sha256(checkpoint) if checkpoint.is_file() else None, "reused": False}


def run_one(job: dict, manifest: dict, checkpoint: Path, *, smoke: bool, python: str, device: str) -> dict:
    result_set = campaign_result_set(manifest, smoke=smoke)
    output = ROOT / "results" / "expansion" / result_set / f"{job['job_id']}.json"
    log_root = ROOT / "logs" / "expansion" / result_set / "jobs"
    command = [python, str(ROOT / "expansion_experiment.py"), "run", "--job-id", job["job_id"], "--dataset", job["dataset"], "--seed", str(job["seed"]), "--method", job["method"], "--trajectory", job["trajectory"], "--shift-family", job["shift_family"], "--severity", str(job["severity"]), "--source-checkpoint", str(checkpoint), "--output", str(output), "--batch-size", str(manifest["batch_size"]), "--adaptation-passes", str(manifest["adaptation_passes"]), "--learning-rate", str(manifest["adaptation_learning_rate"]), "--max-test-samples", str(256 if smoke else 0), "--device", device]
    provenance_file = manifest.get("dataset_provenance_file")
    if provenance_file:
        command.extend(["--dataset-provenance-sha256", sha256(ROOT / provenance_file)])
    if smoke:
        command.append("--smoke")
    outcome = run_process(command, log_root / f"{job['job_id']}.stdout.log", log_root / f"{job['job_id']}.stderr.log", int(manifest.get("retries", 2)))
    return outcome | {"job": job, "output": output, "sha256": sha256(output) if output.is_file() else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "expansion_campaign_v1.json")
    parser.add_argument("--status", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = read(args.manifest)
    python = sys.executable
    result_set = campaign_result_set(manifest, smoke=args.smoke)
    status_path = args.status or ROOT / "run_state" / f"expansion_{result_set}_status.json"
    result_root = ROOT / "results" / "expansion" / result_set
    all_jobs = jobs(manifest)
    if args.smoke:
        all_jobs = [job for job in all_jobs if job["dataset"] == "cifar10" and job["seed"] == 0 and job["method"] in {"source", "tent"} and job["trajectory"] == "A-B-A" and job["shift_family"] == "noise" and job["severity"] == 1]
    if args.max_jobs:
        all_jobs = all_jobs[: args.max_jobs]
    current_provenance = {
        "manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(ROOT / "expansion_experiment.py"),
        "supervisor_sha256": sha256(Path(__file__)),
        "dataset_provenance_sha256": sha256(ROOT / manifest["dataset_provenance_file"]) if manifest.get("dataset_provenance_file") else None,
    }
    prior = read(status_path) if args.resume and status_path.is_file() else {}
    plan = resume_plan(
        all_jobs,
        prior,
        result_root,
        expected_provenance=current_provenance if args.resume else None,
    )
    state = {
        "schema_version": "1.0.0",
        "status": "RUNNING",
        "campaign": "cross-dataset recurrent TTA hysteresis",
        "evidence_label": "SMOKE" if args.smoke else manifest.get("evidence_label", "FORMAL_EXPANSION"),
        "result_set": result_set,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_at": prior.get("started_at", utc()),
        "last_heartbeat": utc(),
        "manifest": str(args.manifest),
        **current_provenance,
        "python": python,
        "device": args.device,
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "total_jobs": len(all_jobs),
        "completed": plan["completed"],
        "failed": [],
        "preserved_failures": plan["preserved_failures"],
        "remaining_jobs": len(plan["pending"]),
        "workers": max(1, args.workers),
    }
    write(status_path, state)
    source_checkpoints: dict[tuple[str, int], Path] = {}
    for dataset, seed in sorted({(job["dataset"], job["seed"]) for job in plan["pending"]}):
        source = ensure_source(dataset, seed, manifest, smoke=args.smoke, python=python, device=args.device)
        state["last_heartbeat"] = utc()
        if not source["ok"]:
            state["failed"].append({"job_id": f"source_{dataset}_s{seed:02d}", **source})
            state["status"] = "FAILED"
            write(status_path, state)
            return 1
        source_checkpoints[(dataset, seed)] = source["checkpoint"]
        write(status_path, state)
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for job in plan["pending"]:
            future = pool.submit(run_one, job, manifest, source_checkpoints[(job["dataset"], job["seed"])], smoke=args.smoke, python=python, device=args.device)
            futures[future] = job
        for future in as_completed(futures):
            outcome = future.result()
            job = outcome["job"]
            if outcome["ok"] and outcome["sha256"]:
                state["completed"][job["job_id"]] = outcome["sha256"]
            else:
                state["failed"].append({"job_id": job["job_id"], "attempts": outcome.get("attempts"), "error": outcome.get("error")})
            state["remaining_jobs"] = len(all_jobs) - len(state["completed"]) - len([item for item in state["failed"] if item.get("job_id") in {job["job_id"] for job in all_jobs}])
            state["last_heartbeat"] = utc()
            write(status_path, state)
    state["status"] = "COMPLETED" if not state["failed"] and len(state["completed"]) == len(all_jobs) else "PARTIAL"
    state["finished_at"] = utc()
    state["remaining_jobs"] = max(0, len(all_jobs) - len(state["completed"]))
    write(status_path, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"]), "status_path": str(status_path)}))
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
