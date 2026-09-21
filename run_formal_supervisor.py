from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jobs(manifest: dict) -> list[dict]:
    output = []
    for seed in manifest["seeds"]:
        for method in manifest["methods"]:
            for trajectory in manifest["trajectories"]:
                for family in manifest["shift_families"]:
                    for severity in manifest["severities"]:
                        job_id = f"s{seed:02d}_{method}_{trajectory.replace('-', '')}_{family}_v{severity}"
                        output.append({"job_id": job_id, "seed": seed, "method": method, "trajectory": trajectory, "shift_family": family, "severity": severity})
    return output


def base_status(manifest_path: Path, total: int, status: str, started: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "status": status,
        "campaign": "Adaptation Hysteresis under Recurrent Distribution Shifts",
        "supervisor_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_at": started,
        "last_heartbeat": utc(),
        "total_jobs": total,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "remaining_jobs": total,
        "current_job": None,
        "failed": [],
        "stdout": str(ROOT / "logs" / "formal_supervisor.stdout.log"),
        "stderr": str(ROOT / "logs" / "formal_supervisor.stderr.log"),
        "results_root": str(ROOT / "results" / "formal"),
        "checkpoint": str(ROOT / "run_state" / "formal_experiments_checkpoint.json"),
        "manifest": str(manifest_path),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
    }


def run_one(job: dict, output: Path, ckpt: Path, stdout_path: Path, stderr_path: Path, retries: int) -> dict:
    command = [sys.executable, str(ROOT / "formal_experiment.py"), "--job-id", job["job_id"], "--method", job["method"], "--trajectory", job["trajectory"], "--shift-family", job["shift_family"], "--severity", str(job["severity"]), "--seed", str(job["seed"]), "--output", str(output), "--checkpoint", str(ckpt)]
    attempts = 0
    last_error = ""
    while attempts <= retries:
        attempts += 1
        proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode == 0 and output.is_file() and ckpt.is_file():
            return {"job": job, "ok": True, "attempts": attempts, "stdout": str(stdout_path), "stderr": str(stderr_path)}
        last_error = (proc.stderr or proc.stdout or f"exit={proc.returncode}")[-4000:]
    return {"job": job, "ok": False, "attempts": attempts, "error": last_error, "stdout": str(stdout_path), "stderr": str(stderr_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "formal_experiment_manifest.json")
    parser.add_argument("--status", type=Path, default=ROOT / "run_state" / "formal_experiments_status.json")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all jobs")
    parser.add_argument("--workers", type=int, default=1, help="parallel subprocess workers")
    parser.add_argument("--resume", action="store_true", help="resume from existing status/checkpoint")
    args = parser.parse_args()
    manifest = read(args.manifest)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    queue = jobs(manifest)
    if args.max_jobs > 0:
        queue = queue[:args.max_jobs]
    started = utc()
    prior = read(args.status) if args.resume and args.status.is_file() else {}
    prior_completed = {str(item) for item in prior.get("completed_job_ids", [])}
    if not prior_completed and args.resume:
        prior_completed = {str(item) for item in read(ROOT / "run_state" / "formal_experiments_checkpoint.json").get("completed_job_ids", [])}
    queue = [job for job in queue if job["job_id"] not in prior_completed]
    state = base_status(args.manifest, len(queue) + len(prior_completed), "RUNNING", str(prior.get("started_at") or started))
    if args.resume or int(args.workers) > 1:
        state["stdout"] = str(ROOT / "logs" / "formal_supervisor_parallel.stdout.log")
        state["stderr"] = str(ROOT / "logs" / "formal_supervisor_parallel.stderr.log")
    state["resumed_at"] = started if prior else None
    state["completed_jobs"] = len(prior_completed)
    state["completed_job_ids"] = sorted(prior_completed)
    state["workers"] = max(1, int(args.workers))
    state["manifest_sha256"] = "sha256:" + manifest_hash
    args.status.parent.mkdir(parents=True, exist_ok=True)
    write(args.status, state)
    log_root = ROOT / "logs" / "formal_jobs"
    result_root = ROOT / "results" / "formal"
    checkpoint_root = ROOT / "checkpoints" / "formal"
    log_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema_version": "1.0.0", "manifest_sha256": "sha256:" + manifest_hash, "next_index": len(prior_completed), "updated_at": utc(), "completed_job_ids": sorted(prior_completed), "failed_job_ids": list(prior.get("failed_job_ids", []))}
    write(Path(state["checkpoint"]), checkpoint)
    state["remaining_jobs"] = len(queue)
    write(args.status, state)
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for job in queue:
            output = result_root / f"{job['job_id']}.json"
            ckpt = checkpoint_root / f"{job['job_id']}.pt"
            stdout_path = log_root / f"{job['job_id']}.stdout.log"
            stderr_path = log_root / f"{job['job_id']}.stderr.log"
            futures[pool.submit(run_one, job, output, ckpt, stdout_path, stderr_path, int(manifest.get("retries", 1)))] = job
        state["current_job"] = [job["job_id"] for job in queue[:max(1, int(args.workers))]]
        state["last_heartbeat"] = utc()
        write(args.status, state)
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job = result["job"]
            state["last_heartbeat"] = utc()
            if result["ok"]:
                state["completed_jobs"] += 1
                state.setdefault("completed_job_ids", []).append(job["job_id"])
                checkpoint["completed_job_ids"].append(job["job_id"])
            else:
                state["failed_jobs"] += 1
                state["failed"].append(result | {"config": job})
                checkpoint["failed_job_ids"].append(job["job_id"])
            checkpoint["next_index"] = len(checkpoint["completed_job_ids"]) + len(checkpoint["failed_job_ids"])
            checkpoint["updated_at"] = utc()
            state["remaining_jobs"] = len(queue) - index
            state["current_job"] = [item["job_id"] for item in queue if item["job_id"] not in set(checkpoint["completed_job_ids"]) and item["job_id"] not in set(checkpoint["failed_job_ids"])][:max(1, int(args.workers))]
            write(Path(state["checkpoint"]), checkpoint)
            write(args.status, state)
    state["current_job"] = None
    state["last_heartbeat"] = utc()
    state["status"] = "COMPLETED" if state["failed_jobs"] == 0 else ("PARTIAL" if state["completed_jobs"] else "FAILED")
    state["finished_at"] = utc()
    state["remaining_jobs"] = 0
    write(args.status, state)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
