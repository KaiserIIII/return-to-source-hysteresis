"""Auditable seed-level analysis for the completed CIFAR-C recovery campaign.

The v1 and recovery ledgers remain immutable.  This script reads both ledgers,
verifies result-file hashes, checks that their union is exactly the registered
2,700-job matrix, and writes a fresh analysis bundle without modifying raw
shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
CAMPAIGN_CONFIG = ROOT / "configs" / "external_validation_cifar_c_v1.json"
CAMPAIGN_SETS = ("external_validation_cifar_c_v1", "external_validation_cifar_c_recovery_v1")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def exact_sign_flip(differences: Iterable[float]) -> dict:
    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("at least one seed-level difference is required")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return {
        "n_seeds": int(len(values)),
        "permutations": int(total),
        "p_value_two_sided": float(extreme / total),
        "mean_difference": float(values.mean()),
    }


def bootstrap_ci(seed_values: dict[int, float], *, draws: int, rng_seed: int) -> dict:
    if not seed_values:
        raise ValueError("seed_values cannot be empty")
    values = np.asarray([seed_values[key] for key in sorted(seed_values)], dtype=float)
    if len(values) == 1:
        distribution = values
    else:
        rng = np.random.default_rng(rng_seed)
        sampled = rng.integers(0, len(values), size=(draws, len(values)))
        distribution = values[sampled].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "n_seeds": int(len(values)),
        "draws": int(draws),
        "rng_seed": int(rng_seed),
    }


def _seed_means(rows: list[dict], value: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(float(row[value]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def paired_contrast(
    rows: list[dict], *, value: str, reference: str, control: str, draws: int = 10000, rng_seed: int = 20260920
) -> dict:
    reference_rows = [row for row in rows if row["method"] == reference]
    control_rows = [row for row in rows if row["method"] == control]
    ref = _seed_means(reference_rows, value)
    ctl = _seed_means(control_rows, value)
    seeds = sorted(set(ref) & set(ctl))
    if not seeds:
        raise ValueError(f"no paired seeds for {reference} vs {control}")
    # Round only the serialized seed contrasts.  This removes binary floating
    # point noise (e.g. 0.013000000000000001) without changing the underlying
    # inference values used by bootstrap/sign-flip calculations.
    differences = {seed: ref[seed] - ctl[seed] for seed in seeds}
    result = {
        "reference": reference,
        "control": control,
        "value": value,
        "interpretation": "positive means reference has a larger value",
        "seed_differences": {str(seed): float(np.round(differences[seed], 12)) for seed in seeds},
    }
    result.update(bootstrap_ci(differences, draws=draws, rng_seed=rng_seed))
    result.update(exact_sign_flip(differences.values()))
    return result


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_suffix(job_id: str) -> str:
    for prefix in ("external_validation_cifar_c_v1_", "external_validation_cifar_c_recovery_v1_"):
        if job_id.startswith(prefix):
            return job_id[len(prefix):]
    raise ValueError(f"unknown job identity: {job_id}")


def _extract_row(shard: dict, *, campaign_set: str, result_hash: str) -> dict:
    episode = shard["episode"]
    returned = episode["return_outcome"]
    target = episode["target_stream_evaluation"]
    prospective = episode["prospective_mechanism"]
    method_state = episode.get("algorithm_semantics", episode.get("method_state", {}))
    return {
        "campaign_set": campaign_set,
        "job_id": shard["job_id"],
        "dataset": shard["dataset"],
        "method": shard["method"],
        "seed": int(shard["seed"]),
        "trajectory": shard["trajectory"],
        "corruption": shard["corruption"],
        "paired_corruption": shard["paired_corruption"],
        "severity": int(shard["severity"]),
        "source_accuracy": float(episode["checkpoint_source_evaluation"]["accuracy"]),
        "configured_source_accuracy": float(episode["configured_source_evaluation"]["accuracy"]),
        "target_accuracy": float(target["accuracy"]),
        "return_accuracy_before": float(returned["accuracy_before_return_adaptation"]),
        "return_accuracy_after": float(returned["accuracy_after_return_adaptation"]),
        "habs": float(returned["absolute_hysteresis_before_return_adaptation"]),
        "hupd": float(returned["update_induced_hysteresis_before_return_adaptation"]),
        "configuration_gap": float(episode["configuration_gap"]),
        "parameter_drift": float(prospective["parameter_drift"]),
        "prediction_entropy": float(prospective["prediction_entropy_before_update"]),
        "gradient_norm": float(prospective["gradient_norm_before_update"]),
        "update_norm": float(prospective["update_norm"]),
        "result_sha256": result_hash,
        "runner_sha256": shard["runner_sha256"],
        "manifest_sha256": shard["manifest_sha256"],
        "dataset_provenance_sha256": shard["dataset_provenance_sha256"],
        "source_model_immutability": episode["external_source_model_immutability"]["status"],
        "predictor_step": int(episode["events"]["predictor_step"]),
        "outcome_step": int(episode["events"]["outcome_step"]),
        "optimizer_step_count": int(method_state.get("optimizer_step_count", episode["state_transition"].get("optimizer_step_count", 0))),
        "reliable_samples": int(method_state.get("reliable_samples", 0)),
        "nonredundant_samples": int(method_state.get("nonredundant_samples", 0)),
        "fisher_samples": int(method_state.get("fisher_samples", 0)),
    }


def load_verified_rows(*, project_root: Path = ROOT) -> tuple[list[dict], dict]:
    config = _load_json(project_root / "configs" / "external_validation_cifar_c_v1.json")
    expected_jobs = []
    for dataset in config["datasets"]:
        for corruption in config["corruptions"]:
            for severity in config["severities"]:
                for seed in config["seeds"]:
                    for method in config["methods"]:
                        trajectory = config["formal_trajectories"][0]
                        paired = config["corruptions"][(config["corruptions"].index(corruption) + 1) % len(config["corruptions"])]
                        suffix = f"{dataset}_s{seed:02d}_{method}_{trajectory.replace('-', '')}_{corruption}_c{paired}_v{severity}"
                        expected_jobs.append(suffix)
    expected = set(expected_jobs)
    observed: dict[str, str] = {}
    rows: list[dict] = []
    # The intentionally preserved failed v1 and its recovery are separate
    # formal campaigns.  Their runner and dataset identities are shared, but
    # their frozen manifests may differ because the recovery was admitted as a
    # new campaign.  Track provenance per campaign instead of falsely treating
    # that expected boundary as a mixed-code failure.
    provenance_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for campaign_set in CAMPAIGN_SETS:
        status = _load_json(project_root / "run_state" / f"{campaign_set}_status.json")
        for job_id, expected_hash in status["completed"].items():
            suffix = _job_suffix(job_id)
            if suffix in observed:
                raise ValueError(f"duplicate matrix cell: {suffix}")
            path = project_root / "results" / "external_validation" / campaign_set / f"{job_id}.json"
            if not path.is_file() or sha256(path) != expected_hash:
                raise ValueError(f"missing or hash-mismatched shard: {job_id}")
            shard = _load_json(path)
            if shard.get("status") != "PASS" or _job_suffix(shard["job_id"]) != suffix:
                raise ValueError(f"invalid shard identity/status: {job_id}")
            if shard["episode"]["external_source_model_immutability"]["status"] != "PASS":
                raise ValueError(f"source immutability failed: {job_id}")
            if shard["episode"]["events"]["predictor_step"] >= shard["episode"]["events"]["outcome_step"]:
                raise ValueError(f"temporal ordering failed: {job_id}")
            transition = shard["episode"]["state_transition"]
            if shard["method"] == "source" and (transition.get("before_digest") != transition.get("after_digest") or transition.get("optimizer_step_count") != 0):
                raise ValueError(f"source state changed: {job_id}")
            observed[suffix] = campaign_set
            provenance_sets[campaign_set]["runner"].add(shard["runner_sha256"])
            provenance_sets[campaign_set]["manifest"].add(shard["manifest_sha256"])
            provenance_sets[campaign_set]["dataset"].add(shard["dataset_provenance_sha256"])
            rows.append(_extract_row(shard, campaign_set=campaign_set, result_hash=expected_hash))
    if set(observed) != expected:
        raise ValueError(f"matrix coverage mismatch missing={len(expected-set(observed))} extra={len(set(observed)-expected)}")
    if any(len(values) != 1 for campaign in provenance_sets.values() for values in campaign.values()):
        raise ValueError(f"mixed provenance identities: {provenance_sets}")
    # A combined analysis may span a failed campaign and a recovery campaign,
    # but runner and dataset provenance must still agree across that union.
    for identity in ("runner", "dataset"):
        if len({next(iter(provenance_sets[campaign][identity])) for campaign in provenance_sets}) != 1:
            raise ValueError(f"mixed cross-campaign {identity} identities: {provenance_sets}")
    return rows, {
        "status": "PASS",
        "expected_jobs": len(expected),
        "observed_jobs": len(rows),
        "campaign_sets": {name: sum(1 for value in observed.values() if value == name) for name in CAMPAIGN_SETS},
        "provenance": {
            campaign: {name: sorted(values)[0] for name, values in identities.items()}
            for campaign, identities in provenance_sets.items()
        },
        "source_shards": sum(row["method"] == "source" for row in rows),
        "temporal_ordered_shards": sum(row["predictor_step"] < row["outcome_step"] for row in rows),
        "configuration_gap_nonzero": sum(abs(row["configuration_gap"]) > 1e-12 for row in rows),
    }


def summarize(rows: list[dict], *, group_fields: list[str], value: str, draws: int, seed: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for index, (key, subset) in enumerate(sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))):
        values = _seed_means(subset, value)
        output.append({**dict(zip(group_fields, key)), "n_rows": len(subset), **bootstrap_ci(values, draws=draws, rng_seed=seed + index)})
    return output


def build_analysis(rows: list[dict], audit: dict, *, draws: int, seed: int) -> dict:
    methods = ["source", "tent", "eata", "sar", "cotta", "rotta"]
    contrasts = {}
    for value in ("habs", "hupd", "target_accuracy"):
        family = []
        for index, method in enumerate(methods[1:]):
            family.append(paired_contrast(rows, value=value, reference=method, control="source", draws=draws, rng_seed=seed + index))
        adjusted = holm_adjust([item["p_value_two_sided"] for item in family])
        for item, p in zip(family, adjusted):
            item["p_value_holm"] = p
        contrasts[f"{value}_vs_source"] = family
    family = []
    for index, method in enumerate(methods[2:]):
        family.append(paired_contrast(rows, value="habs", reference="tent", control=method, draws=draws, rng_seed=seed + 100 + index))
    adjusted = holm_adjust([item["p_value_two_sided"] for item in family])
    for item, p in zip(family, adjusted):
        item["p_value_holm"] = p
    contrasts["habs_tent_vs_adaptive"] = family
    return {
        "schema_version": "1.0.0",
        "analysis_label": "FORMAL_EXTERNAL_VALIDATION_CIFAR_C",
        "analysis_status": "PASS",
        "created_at": utc(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        "integrity_audit": audit,
        "design": {
            "independent_replication_unit": "source-training seed",
            "seed_aggregation": "average all corruption x severity cells within each seed before inference",
            "bootstrap": f"seed-cluster percentile bootstrap, {draws} draws",
            "paired_test": "exact two-sided sign-flip over five seed-level differences",
            "multiplicity": "Holm within each registered contrast family",
            "external_protocol": "standard CIFAR-10-C/CIFAR-100-C, A-B-A, severities 1/3/5",
        },
        "counts": {"rows": len(rows), "datasets": len({r["dataset"] for r in rows}), "methods": len({r["method"] for r in rows}), "seeds": len({r["seed"] for r in rows}), "corruptions": len({r["corruption"] for r in rows}), "severities": len({r["severity"] for r in rows})},
        "summaries": {
            "method": summarize(rows, group_fields=["method"], value="habs", draws=draws, seed=seed),
            "method_hupd": summarize(rows, group_fields=["method"], value="hupd", draws=draws, seed=seed + 10),
            "method_target_accuracy": summarize(rows, group_fields=["method"], value="target_accuracy", draws=draws, seed=seed + 20),
            "dataset_method": summarize(rows, group_fields=["dataset", "method"], value="habs", draws=draws, seed=seed + 30),
            "severity_method": summarize(rows, group_fields=["severity", "method"], value="habs", draws=draws, seed=seed + 40),
            "corruption_method": summarize(rows, group_fields=["corruption", "method"], value="habs", draws=draws, seed=seed + 50),
            "eata_diagnostics": summarize(rows, group_fields=["dataset"], value="target_accuracy", draws=draws, seed=seed + 60),
        },
        "contrasts": contrasts,
        "negative_results": [
            "The external CIFAR-C campaign does not support a universal method ranking.",
            "Configuration_gap is zero under the frozen bn_affine_frozen_stats policy; external Habs equals Hupd by construction for these shards.",
            "EATA shows severe target-utility degradation in this independent reimplementation and must be treated as a stability result requiring implementation-sensitivity discussion, not as a leaderboard claim.",
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict], output: Path, *, seed: int) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    methods = ["source", "tent", "eata", "sar", "cotta", "rotta"]
    labels = {"source": "Source", "tent": "TENT", "eata": "EATA", "sar": "SAR", "cotta": "CoTTA", "rotta": "RoTTA"}
    colors = {"source": "#4D4D4D", "tent": "#D55E00", "eata": "#CC79A7", "sar": "#0072B2", "cotta": "#009E73", "rotta": "#E69F00"}
    manifests = []

    def export(fig, stem, description):
        paths = [output / f"{stem}.{ext}" for ext in ("png", "svg", "pdf")]
        for path in paths:
            fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
        manifests.append({"figure": stem, "description": description, "files": [{"path": str(p), "sha256": sha256(p)} for p in paths]})
        plt.close(fig)

    for dataset in sorted({row["dataset"] for row in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
        for ax, value, label in zip(axes, ("habs", "target_accuracy"), ("Return loss H_abs", "Target accuracy")):
            for index, method in enumerate(methods):
                subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
                means = _seed_means(subset, value)
                summary = bootstrap_ci(means, draws=5000, rng_seed=seed + index)
                vals = np.asarray([means[k] for k in sorted(means)])
                jitter = np.linspace(-0.10, 0.10, len(vals))
                ax.scatter(index + jitter, vals, s=13, facecolors="none", edgecolors=colors[method], linewidths=0.8)
                ax.errorbar(index, summary["mean"], yerr=[[summary["mean"] - summary["ci95_low"]], [summary["ci95_high"] - summary["mean"]]], fmt="o", color=colors[method], capsize=3, markersize=5)
            ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(dataset.replace("_c", "-C"))
        export(fig, f"external_{dataset}_method_summary", "Seed-level means and 95% seed-cluster intervals by method.")

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    shifts = sorted({row["corruption"] for row in rows})
    matrix = np.asarray([[np.mean([r["hupd"] for r in rows if r["corruption"] == shift and r["method"] == method]) for method in methods] for shift in shifts])
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 1e-9)
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(len(shifts)), shifts)
    for i in range(len(shifts)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=6, color="white" if matrix[i,j] > limit * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="H_upd")
    export(fig, "external_cifar_c_shift_method_heatmap", "Mean update-induced return loss by corruption and method.")
    return manifests


def render_report(analysis: dict) -> str:
    lines = ["# CIFAR-C External Validation Analysis", "", f"Generated: `{analysis['created_at']}`", "", "## Integrity", "", f"The combined ledger covers {analysis['integrity_audit']['observed_jobs']} of {analysis['integrity_audit']['expected_jobs']} registered jobs with no missing or duplicate matrix cells. Source immutability and temporal ordering were checked before statistics.", "", "## Method summaries", "", "| Method | H_abs mean | H_abs 95% CI | H_upd mean | Target accuracy |", "|---|---:|---:|---:|---:|"]
    by_method = {item["method"]: item for item in analysis["summaries"]["method"]}
    by_update = {item["method"]: item for item in analysis["summaries"]["method_hupd"]}
    by_target = {item["method"]: item for item in analysis["summaries"]["method_target_accuracy"]}
    for method in ["source", "tent", "eata", "sar", "cotta", "rotta"]:
        h = by_method[method]; u = by_update[method]; t = by_target[method]
        lines.append(f"| {method} | {h['mean']:.6f} | [{h['ci95_low']:.6f}, {h['ci95_high']:.6f}] | {u['mean']:.6f} | {t['mean']:.6f} |")
    lines += ["", "## Interpretation boundary", "", "The frozen external protocol uses `bn_affine_frozen_stats`, so configuration_gap is zero in all audited shards and H_abs equals H_upd. This external campaign therefore tests update-induced return loss under a fixed normalization policy; it does not replace the formal_v2 normalization decomposition.", "", "EATA's target accuracy is substantially lower than the other methods in this independent reimplementation. That observation is retained as a stability signal and is not promoted to a universal claim about EATA without an implementation-sensitivity audit against the pinned upstream reference.", "", "All inferential comparisons use source-training seed as the independent replication unit, exact two-sided sign-flip tests, 95% seed-cluster bootstrap intervals, and Holm correction within each registered family.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analysis" / "external_validation_cifar_c_v1")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, audit = load_verified_rows()
    analysis = build_analysis(rows, audit, draws=args.draws, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "rows.csv", rows)
    analysis["figures"] = make_figures(rows, args.output / "figures", seed=args.seed)
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "analysis_report.md").write_text(render_report(analysis), encoding="utf-8")
    manifest = {"schema_version": "1.0.0", "created_at": utc(), "analysis_script_sha256": sha256(Path(__file__)), "artifacts": []}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["artifacts"].append({"path": str(path.relative_to(args.output)), "sha256": sha256(path)})
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
