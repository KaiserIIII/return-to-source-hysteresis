"""Build an auditable recovery campaign for an interrupted CIFAR-C run.

The original campaign remains immutable and is never resumed after an early
sanity failure.  This helper creates a new identity and an explicit job list
containing only jobs that did not produce valid v1 shards.
"""

from __future__ import annotations

import json
from pathlib import Path

from external_validation_supervisor import enumerate_formal_jobs


ROOT = Path(__file__).resolve().parent
SOURCE_CONFIG = ROOT / "configs" / "external_validation_cifar_c_v1.json"
SOURCE_STATUS = ROOT / "run_state" / "external_validation_cifar_c_v1_status.json"
RECOVERY_CONFIG = ROOT / "configs" / "external_validation_cifar_c_recovery_v1.json"
RECOVERY_JOBS = ROOT / "run_state" / "external_validation_cifar_c_recovery_v1_job_list.json"
RECOVERY_REPORT = ROOT / "run_state" / "external_validation_cifar_c_recovery_v1_plan.json"


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    status = json.loads(SOURCE_STATUS.read_text(encoding="utf-8"))
    source_jobs = enumerate_formal_jobs(source)
    completed = set(status.get("completed", {}))
    pending_source = [job for job in source_jobs if job["job_id"] not in completed]

    recovery = dict(source)
    recovery["campaign_id"] = "external_validation_cifar_c_recovery_v1"
    recovery["result_set"] = "external_validation_cifar_c_recovery_v1"
    recovery["smoke_result_set"] = "external_validation_cifar_c_recovery_v1_smoke"
    recovery["canary_result_set"] = "external_validation_cifar_c_recovery_v1_canary"
    recovery["recovery_parent_campaign"] = source["campaign_id"]
    recovery["recovery_reason"] = "v1 early sanity failure; execute only uncompleted shards"

    recovery_jobs = enumerate_formal_jobs(recovery)
    by_suffix = {
        job["job_id"].removeprefix(recovery["campaign_id"] + "_"): job["job_id"]
        for job in recovery_jobs
    }
    requested = []
    for job in pending_source:
        suffix = job["job_id"].removeprefix(source["campaign_id"] + "_")
        try:
            requested.append(by_suffix[suffix])
        except KeyError as exc:
            raise RuntimeError(f"could not map pending job {job['job_id']}") from exc

    RECOVERY_CONFIG.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
    RECOVERY_JOBS.write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "1.0.0",
        "status": "PLANNED",
        "parent_campaign_id": source["campaign_id"],
        "recovery_campaign_id": recovery["campaign_id"],
        "parent_status": status.get("status"),
        "parent_completed": len(completed),
        "recovery_job_count": len(requested),
        "parent_failed": status.get("failed", []),
        "source_config": str(SOURCE_CONFIG),
        "source_status": str(SOURCE_STATUS),
        "recovery_config": str(RECOVERY_CONFIG),
        "recovery_job_list": str(RECOVERY_JOBS),
    }
    RECOVERY_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
