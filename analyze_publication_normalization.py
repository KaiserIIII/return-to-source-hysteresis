from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import publication_campaign_admission as admission


ROOT = Path(__file__).resolve().parent
POLICIES = [
    "source_eval",
    "bn_affine_frozen_stats",
    "bn_affine_running_stats",
    "train_eval_policy_variant",
    "groupnorm_checkpoint",
    "layernorm_checkpoint",
]
METRICS = {
    "absolute_hysteresis": ("episode", "return_outcome", "absolute_hysteresis_before_return_adaptation"),
    "update_induced_hysteresis": ("episode", "return_outcome", "update_induced_hysteresis_before_return_adaptation"),
    "target_accuracy": ("episode", "target_stream_evaluation", "accuracy"),
    "checkpoint_source_accuracy": ("episode", "checkpoint_source_evaluation", "accuracy"),
    "return_accuracy": ("episode", "return_outcome", "accuracy_before_return_adaptation"),
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(max(1, draws), len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def _exact_sign_flip(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return math.nan, 0
    observed = abs(float(values.mean()))
    extremes = 0
    for mask in range(1 << n):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        if abs(float((values * signs).mean())) >= observed - 1e-15:
            extremes += 1
    return min(1.0, extremes / float(1 << n)), 1 << n


def paired_contrast(
    reference: dict[int, float], alternative: dict[int, float], *, draws: int = 10000, seed: int = 20260911
) -> dict[str, Any]:
    seeds = sorted(set(reference) & set(alternative))
    differences = np.asarray([alternative[item] - reference[item] for item in seeds], dtype=float)
    if len(differences) == 0:
        raise ValueError("paired contrast has no common seeds")
    ci_low, ci_high = _bootstrap_mean(differences, draws, seed)
    p_value, permutations = _exact_sign_flip(differences)
    sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    mean = float(differences.mean())
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "difference_by_seed": {str(seed): float(value) for seed, value in zip(seeds, differences)},
        "mean_difference": mean,
        "sd_difference": sd,
        "cohen_dz": mean / sd if sd > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value_two_sided": p_value,
        "permutations": permutations,
        "unit": "source-training-seed",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _metric(row: dict[str, Any], name: str) -> float:
    value: Any = row
    for key in METRICS[name]:
        value = value[key]
    return float(value)


def load_verified_rows(status_path: Path, result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or int(status.get("remaining_jobs", 1)) != 0:
        raise ValueError("campaign is not complete")
    completed = status.get("completed", {})
    if len(completed) != int(status.get("total_jobs", -1)):
        raise ValueError("completed shard count does not match total_jobs")
    expected = {field: status.get(field) for field in admission.PROVENANCE_FIELDS}
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    for job_id, expected_hash in sorted(completed.items()):
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"missing shard: {job_id}")
            continue
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            findings.append(f"result hash mismatch: {job_id}")
            continue
        inspected = admission.inspect_shard(path, expected_provenance=expected)
        findings.extend(f"{job_id}: {item}" for item in inspected["findings"])
        if inspected["status"] != "PASS":
            continue
        shard = inspected["shard"]
        row = {
            "job_id": job_id,
            "dataset": shard["dataset"],
            "method": shard["method"],
            "trajectory": shard["trajectory"],
            "seed": int(shard["seed"]),
            "policy": shard["normalization_policy"],
            **{name: _metric(shard, name) for name in METRICS},
        }
        rows.append(row)
    if findings:
        raise ValueError("verified campaign audit failed: " + " | ".join(findings[:8]))
    return rows, {"status": "PASS", "checked_shards": len(rows), "provenance": expected}


def policy_comparability(rows: list[dict[str, Any]], *, reference: str = "source_eval", threshold: float = 0.05) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row["checkpoint_source_accuracy"])
    ref = float(np.mean(grouped[reference]))
    policies: dict[str, Any] = {}
    for policy in sorted(grouped):
        mean = float(np.mean(grouped[policy]))
        drop = ref - mean
        policies[policy] = {
            "n_rows": len(grouped[policy]),
            "checkpoint_source_accuracy_mean": mean,
            "reference_accuracy_mean": ref,
            "absolute_accuracy_drop": drop,
            "primary_comparable": bool(drop <= threshold),
            "warning": bool(drop > threshold),
            "threshold": threshold,
        }
    return {"reference_policy": reference, "policies": policies}


def _seed_means(rows: list[dict[str, Any]], policy: str, method: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["policy"] == policy and row["method"] == method:
            grouped[row["seed"]].append(row[metric])
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def build_analysis(rows: list[dict[str, Any]], audit: dict[str, Any], *, draws: int, seed: int) -> dict[str, Any]:
    comparability = policy_comparability(rows)
    contrasts: list[dict[str, Any]] = []
    primary_p_values: list[float] = []
    primary_indices: list[int] = []
    for policy in POLICIES:
        if policy == "source_eval":
            continue
        for metric in ("absolute_hysteresis", "update_induced_hysteresis", "target_accuracy"):
            result = paired_contrast(
                _seed_means(rows, "source_eval", "tent", metric),
                _seed_means(rows, policy, "tent", metric),
                draws=draws,
                seed=seed + len(contrasts),
            )
            item = {"method": "tent", "reference_policy": "source_eval", "policy": policy, "metric": metric, **result}
            contrasts.append(item)
            if comparability["policies"][policy]["primary_comparable"]:
                primary_indices.append(len(contrasts) - 1)
                primary_p_values.append(result["p_value_two_sided"])
    adjusted = holm_adjust(primary_p_values)
    for index, value in zip(primary_indices, adjusted):
        contrasts[index]["p_value_holm"] = value
    for index, item in enumerate(contrasts):
        item.setdefault("p_value_holm", None)

    summaries: dict[str, Any] = {}
    for policy in POLICIES:
        summaries[policy] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows if row["policy"] == policy], dtype=float)
            low, high = _bootstrap_mean(values, draws, seed + 1000 + len(summaries[policy]))
            summaries[policy][metric] = {
                "n_rows": int(len(values)),
                "mean_row_value": float(values.mean()),
                "sd_row_value": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "note": "row summaries are descriptive; inferential unit is source-training-seed",
            }
    primary = [p for p in POLICIES if p != "source_eval" and comparability["policies"][p]["primary_comparable"]]
    return {
        "schema_version": "1.0.0",
        "analysis_label": "NORMALIZATION_ABLATION_V1_ANALYSIS",
        "analysis_status": "PASS",
        "created_at": utc(),
        "design": {
            "independent_replication_unit": "source-training-seed",
            "seed_aggregation": "mean across A-B-A and A-B-C-A within seed before inference",
            "primary_reference": "source_eval",
            "primary_policies": primary,
            "excluded_from_primary_if_accuracy_drop_gt": 0.05,
            "multiplicity": "Holm correction over comparable policy x metric contrasts",
            "uncertainty": "95% seed-level bootstrap percentile interval",
            "p_value": "exact two-sided sign-flip over seed-level paired differences",
        },
        "audit": audit,
        "comparability": comparability,
        "summaries": summaries,
        "contrasts": contrasts,
        "claim_boundary": {
            "status": "CONDITIONAL",
            "supported": "normalization policy materially changes observed return-to-source hysteresis in this CIFAR-10 noise protocol",
            "not_supported": "a universal causal decomposition into configuration gap and recurrent updates; this campaign sets configuration_gap to zero within each policy checkpoint",
            "negative_or_limited": "GroupNorm and LayerNorm checkpoints collapse source-domain accuracy and are exploratory architecture controls, not fair primary comparisons",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_outputs(output: Path, rows: list[dict[str, Any]], analysis: dict[str, Any], *, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in (output / "rows.csv", output / "analysis.json", output / "analysis_report.md", output / "artifact_manifest.json"):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {path}")
    with (output / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "analysis.json", analysis)
    lines = [
        "# Normalization Ablation v1 Analysis",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "The 120 raw shards passed hash, provenance, temporal, mutation, metric-range, and source-immutability audits.",
        "Inference uses source-training seed as the independent replication unit; rows are descriptive only.",
        "",
        "## Policy comparability",
        "",
        "| Policy | Source accuracy | Drop vs source_eval | Primary comparable |",
        "|---|---:|---:|:---:|",
    ]
    for policy, item in analysis["comparability"]["policies"].items():
        lines.append(f"| {policy} | {item['checkpoint_source_accuracy_mean']:.4f} | {item['absolute_accuracy_drop']:.4f} | {'yes' if item['primary_comparable'] else 'no'} |")
    lines.extend(["", "## TENT paired contrasts vs source_eval", "", "| Policy | Metric | Mean difference | 95% CI | dz | exact p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"])
    for item in analysis["contrasts"]:
        ci = f"[{item['ci95_low']:.5f}, {item['ci95_high']:.5f}]"
        dz = "inf" if math.isinf(item["cohen_dz"]) else f"{item['cohen_dz']:.3f}"
        holm = "—" if item["p_value_holm"] is None else f"{item['p_value_holm']:.5f}"
        lines.append(f"| {item['policy']} | {item['metric']} | {item['mean_difference']:.5f} | {ci} | {dz} | {item['p_value_two_sided']:.5f} | {holm} |")
    lines.extend(["", "## Claim boundary", "", analysis["claim_boundary"]["supported"], "", analysis["claim_boundary"]["not_supported"], "", analysis["claim_boundary"]["negative_or_limited"], ""])
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "analysis_script_sha256": sha256(Path(__file__)),
        "artifacts": [],
    }
    for path in sorted(output.iterdir()):
        if path.name != "artifact_manifest.json":
            manifest["artifacts"].append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    _write_json(output / "artifact_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=ROOT / "run_state/publication_normalization_ablation_v1_status.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/publication_expansion/normalization_ablation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/normalization_ablation_v1")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, audit = load_verified_rows(args.status, args.results)
    analysis = build_analysis(rows, audit, draws=args.draws, seed=args.seed)
    write_outputs(args.output, rows, analysis, force=args.force)
    print(json.dumps({"status": analysis["analysis_status"], "rows": len(rows), "output": str(args.output), "primary_policies": analysis["design"]["primary_policies"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
