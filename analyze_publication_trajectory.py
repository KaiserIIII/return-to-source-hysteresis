from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import publication_campaign_admission as admission


ROOT = Path(__file__).resolve().parent
METHODS = ["source", "tent", "eata", "sar", "cotta", "rotta"]
ADAPTIVE_METHODS = ["tent", "eata", "sar", "cotta", "rotta"]
TRAJECTORIES = ["A-B-A", "A-B-C-A", "A-B-A-B-A", "A-C-B-A"]
PASS_COUNTS = [1, 2, 4]
LEARNING_RATES = [0.0001, 0.001, 0.01]
SEEDS = [0, 1, 2, 3, 4]
METRICS = {
    "absolute_hysteresis": ("episode", "return_outcome", "absolute_hysteresis_before_return_adaptation"),
    "update_induced_hysteresis": ("episode", "return_outcome", "update_induced_hysteresis_before_return_adaptation"),
    "target_accuracy": ("episode", "target_stream_evaluation", "accuracy"),
    "return_accuracy": ("episode", "return_outcome", "accuracy_before_return_adaptation"),
    "parameter_drift": ("episode", "prospective_mechanism", "parameter_drift"),
}
CONTRAST_SPECS = [
    ("trajectory", "A-B-A", "A-B-C-A", "additional_shift"),
    ("trajectory", "A-B-C-A", "A-C-B-A", "order_reversal"),
    ("trajectory", "A-B-A", "A-B-A-B-A", "recurrence"),
    ("pass_count", 1, 2, "pass_1_to_2"),
    ("pass_count", 1, 4, "pass_1_to_4"),
    ("learning_rate", 0.001, 0.0001, "lr_mid_to_low"),
    ("learning_rate", 0.001, 0.01, "lr_mid_to_high"),
]


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
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(max(1, draws), len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _exact_sign_flip(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot test an empty paired sample")
    observed = abs(float(values.mean()))
    extremes = 0
    permutations = 1 << len(values)
    for mask in range(permutations):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(values))])
        if abs(float((values * signs).mean())) >= observed - 1e-15:
            extremes += 1
    return min(1.0, extremes / float(permutations)), permutations


def _paired_summary(
    reference: dict[int, float],
    alternative: dict[int, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    paired_seeds = sorted(set(reference) & set(alternative))
    if not paired_seeds:
        raise ValueError("paired contrast has no common seeds")
    differences = np.asarray(
        [round(alternative[item] - reference[item], 15) for item in paired_seeds],
        dtype=float,
    )
    ci_low, ci_high = _bootstrap_mean(differences, draws, seed)
    p_value, permutations = _exact_sign_flip(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {
        "n_seeds": len(paired_seeds),
        "seeds": paired_seeds,
        "difference_by_seed": {
            str(item): float(value) for item, value in zip(paired_seeds, differences)
        },
        "mean_difference": mean,
        "sd_difference": sd,
        "cohen_dz": mean / sd if sd > 0 else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value_two_sided": p_value,
        "permutations": permutations,
        "unit": "source-training-seed",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _seed_factor_means(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    metric: str,
    factor: str,
    value: Any,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row[factor] == value:
            grouped[int(row["seed"])].append(float(row[metric]))
    return {item: float(np.mean(values)) for item, values in grouped.items()}


def factor_contrast(
    rows: list[dict[str, Any]],
    *,
    method: str,
    metric: str,
    factor: str,
    reference: Any,
    alternative: Any,
    draws: int = 10000,
    seed: int = 20260918,
) -> dict[str, Any]:
    result = _paired_summary(
        _seed_factor_means(rows, method=method, metric=metric, factor=factor, value=reference),
        _seed_factor_means(rows, method=method, metric=metric, factor=factor, value=alternative),
        draws=draws,
        seed=seed,
    )
    return {
        "method": method,
        "metric": metric,
        "factor": factor,
        "reference": reference,
        "alternative": alternative,
        **result,
    }


def design_coverage(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    trajectories: list[str],
    pass_counts: list[int],
    learning_rates: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    expected = {
        (method, trajectory, int(pass_count), float(learning_rate), int(seed))
        for method in methods
        for trajectory in trajectories
        for pass_count in pass_counts
        for learning_rate in learning_rates
        for seed in seeds
    }
    observed = {
        (
            str(row["method"]),
            str(row["trajectory"]),
            int(row["pass_count"]),
            float(row["learning_rate"]),
            int(row["seed"]),
        )
        for row in rows
    }
    missing = sorted(expected - observed, key=str)
    unexpected = sorted(observed - expected, key=str)
    return {
        "status": "PASS" if not missing and not unexpected and len(rows) == len(expected) else "FAIL",
        "expected_cells": len(expected),
        "observed_cells": len(rows),
        "unique_observed_cells": len(observed),
        "missing_cells": [list(item) for item in missing],
        "unexpected_cells": [list(item) for item in unexpected],
    }


def _metric(shard: dict[str, Any], name: str) -> float:
    value: Any = shard
    for key in METRICS[name]:
        value = value[key]
    return float(value)


def load_verified_rows(status_path: Path, result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or int(status.get("remaining_jobs", 1)) != 0:
        raise ValueError("campaign is not complete")
    completed = status.get("completed", {})
    if not isinstance(completed, dict) or len(completed) != int(status.get("total_jobs", -1)):
        raise ValueError("completed shard count does not match total_jobs")
    if status.get("failed"):
        raise ValueError("campaign contains unresolved failed jobs")
    expected = {field: status.get(field) for field in admission.PROVENANCE_FIELDS}
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    for job_id, expected_hash in sorted(completed.items()):
        path = result_root / f"{job_id}.json"
        if not path.is_file():
            findings.append(f"missing shard: {job_id}")
            continue
        if sha256(path) != expected_hash:
            findings.append(f"result hash mismatch: {job_id}")
            continue
        inspected = admission.inspect_shard(path, expected_provenance=expected)
        findings.extend(f"{job_id}: {item}" for item in inspected["findings"])
        if inspected["status"] != "PASS":
            continue
        shard = inspected["shard"]
        rows.append(
            {
                "job_id": job_id,
                "dataset": shard["dataset"],
                "method": shard["method"],
                "trajectory": shard["trajectory"],
                "seed": int(shard["seed"]),
                "normalization_policy": shard["normalization_policy"],
                "pass_count": int(shard["protocol_parameters"]["pass_count"]),
                "learning_rate": float(shard["protocol_parameters"]["adaptation_learning_rate"]),
                **{name: _metric(shard, name) for name in METRICS},
            }
        )
    if findings:
        raise ValueError("verified campaign audit failed: " + " | ".join(findings[:8]))
    return rows, {
        "status": "PASS",
        "checked_shards": len(rows),
        "provenance": expected,
        "status_sha256": sha256(status_path),
    }


def _seed_summary(
    rows: list[dict[str, Any]],
    *,
    method: str,
    metric: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            grouped[int(row["seed"])].append(float(row[metric]))
    seed_means = {item: float(np.mean(values)) for item, values in grouped.items()}
    values = np.asarray([seed_means[item] for item in sorted(seed_means)], dtype=float)
    low, high = _bootstrap_mean(values, draws, seed)
    return {
        "n_rows": sum(len(values_) for values_ in grouped.values()),
        "n_seeds": len(values),
        "seed_means": {str(item): seed_means[item] for item in sorted(seed_means)},
        "mean": float(values.mean()),
        "sd_across_seed_means": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "unit": "source-training-seed",
    }


def build_analysis(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    coverage = design_coverage(
        rows,
        methods=METHODS,
        trajectories=TRAJECTORIES,
        pass_counts=PASS_COUNTS,
        learning_rates=LEARNING_RATES,
        seeds=SEEDS,
    )
    if coverage["status"] != "PASS":
        raise ValueError("trajectory factorial design is incomplete")

    summaries: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        summaries[method] = {
            metric: _seed_summary(
                rows,
                method=method,
                metric=metric,
                draws=draws,
                seed=seed + 1000 + method_index * len(METRICS) + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }

    contrasts: list[dict[str, Any]] = []
    contrast_seed = seed
    for method in ADAPTIVE_METHODS:
        for metric in ("absolute_hysteresis", "update_induced_hysteresis", "target_accuracy"):
            for factor, reference, alternative, label in CONTRAST_SPECS:
                item = factor_contrast(
                    rows,
                    method=method,
                    metric=metric,
                    factor=factor,
                    reference=reference,
                    alternative=alternative,
                    draws=draws,
                    seed=contrast_seed,
                )
                item["contrast"] = label
                item["p_value_holm"] = None
                contrasts.append(item)
                contrast_seed += 1

    for metric in ("absolute_hysteresis", "update_induced_hysteresis", "target_accuracy"):
        indices = [index for index, item in enumerate(contrasts) if item["metric"] == metric]
        adjusted = holm_adjust([contrasts[index]["p_value_two_sided"] for index in indices])
        for index, value in zip(indices, adjusted):
            contrasts[index]["p_value_holm"] = value

    source_rows = [row for row in rows if row["method"] == "source"]
    source_control = {
        "status": "PASS"
        if max(abs(row["absolute_hysteresis"]) for row in source_rows) <= 1e-12
        and max(abs(row["update_induced_hysteresis"]) for row in source_rows) <= 1e-12
        and max(abs(row["parameter_drift"]) for row in source_rows) <= 1e-12
        else "FAIL",
        "n_rows": len(source_rows),
        "max_abs_hysteresis": max(abs(row["absolute_hysteresis"]) for row in source_rows),
        "max_abs_update_induced_hysteresis": max(
            abs(row["update_induced_hysteresis"]) for row in source_rows
        ),
        "max_parameter_drift": max(abs(row["parameter_drift"]) for row in source_rows),
    }
    return {
        "schema_version": "1.0.0",
        "analysis_label": "TRAJECTORY_ABLATION_GPU_V1_ANALYSIS",
        "analysis_status": "PASS" if source_control["status"] == "PASS" else "FAIL",
        "created_at": utc(),
        "design": {
            "independent_replication_unit": "source-training-seed",
            "n_independent_seeds": 5,
            "seed_aggregation": "average all nuisance-factor cells within each source-training seed before paired inference",
            "primary_endpoints": ["absolute_hysteresis", "update_induced_hysteresis"],
            "secondary_endpoint": "target_accuracy",
            "planned_contrasts": [item[3] for item in CONTRAST_SPECS],
            "multiplicity": "Holm correction separately across all 35 planned contrasts for each endpoint",
            "uncertainty": "95% percentile bootstrap over five source-training seed means",
            "p_value": "exact two-sided sign-flip over five paired seed differences; minimum attainable nonzero p is 0.0625",
            "confirmatory_boundary": "CIFAR-10, noise severity 1, frozen-BN-affine policy, independently reimplemented methods",
        },
        "audit": audit,
        "coverage": coverage,
        "source_control": source_control,
        "summaries": summaries,
        "contrasts": contrasts,
        "claim_boundary": {
            "status": "CONDITIONAL",
            "supported": "the factorial campaign estimates how trajectory order, recurrence, pass count, and learning rate change return behavior under the registered CIFAR-10 protocol",
            "not_supported": "generalization to standard CIFAR-C corruptions, CIFAR-100-C, natural shifts, or a universal mechanism",
            "inference_limit": "with five independent source-training seeds, exact two-sided sign-flip tests cannot attain p < 0.05; effect sizes and uncertainty intervals carry the interpretation",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(output: Path, contrasts: list[dict[str, Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "tent": "#1f77b4",
        "eata": "#2ca02c",
        "sar": "#d62728",
        "cotta": "#9467bd",
        "rotta": "#8c564b",
    }
    figure_paths: list[Path] = []
    groups = [
        ("trajectory", ["additional_shift", "order_reversal", "recurrence"], "Trajectory contrasts"),
        ("hyperparameter", ["pass_1_to_2", "pass_1_to_4", "lr_mid_to_low", "lr_mid_to_high"], "Pass/LR contrasts"),
    ]
    for file_stem, labels, title in groups:
        fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.3), sharey=True)
        if len(labels) == 1:
            axes = [axes]
        for axis, label in zip(axes, labels):
            subset = [
                item
                for item in contrasts
                if item["metric"] == "absolute_hysteresis" and item["contrast"] == label
            ]
            subset.sort(key=lambda item: ADAPTIVE_METHODS.index(item["method"]))
            y = np.arange(len(subset))
            means = np.asarray([item["mean_difference"] for item in subset])
            lower = means - np.asarray([item["ci95_low"] for item in subset])
            upper = np.asarray([item["ci95_high"] for item in subset]) - means
            for index, item in enumerate(subset):
                axis.errorbar(
                    means[index],
                    y[index],
                    xerr=np.asarray([[lower[index]], [upper[index]]]),
                    fmt="o",
                    color=palette[item["method"]],
                    capsize=3,
                    markersize=5,
                    linewidth=1.4,
                )
            axis.axvline(0.0, color="#333333", linewidth=0.9, linestyle="--")
            axis.set_title(label.replace("_", " "))
            axis.set_xlabel("Paired difference in Habs")
            axis.grid(axis="x", color="#dddddd", linewidth=0.6)
            axis.set_yticks(y, [item["method"].upper() for item in subset])
        axes[0].set_ylabel("Method")
        fig.suptitle(title)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{file_stem}_habs.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            figure_paths.append(path)
        plt.close(fig)
    return figure_paths


def _result_root_digest(completed: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for job_id, value in sorted(completed.items()):
        digest.update(job_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def write_outputs(
    output: Path,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    status_path: Path,
    force: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output / "rows.csv",
        output / "contrasts.csv",
        output / "analysis.json",
        output / "analysis_report.md",
        output / "campaign_freeze.json",
        output / "trajectory_habs.png",
        output / "trajectory_habs.pdf",
        output / "hyperparameter_habs.png",
        output / "hyperparameter_habs.pdf",
        output / "artifact_manifest.json",
    ]
    for path in artifacts:
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {path}")

    _write_csv(output / "rows.csv", rows)
    flat_contrasts = [
        {
            key: value
            for key, value in item.items()
            if key not in {"seeds", "difference_by_seed"}
        }
        for item in analysis["contrasts"]
    ]
    _write_csv(output / "contrasts.csv", flat_contrasts)
    _write_json(output / "analysis.json", analysis)
    _write_figures(output, analysis["contrasts"])

    primary = [item for item in analysis["contrasts"] if item["metric"] == "absolute_hysteresis"]
    lines = [
        "# Trajectory Ablation GPU v1 Analysis",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "All 1,080 formal shards passed hash, provenance, temporal-order, mutation-contract, metric-range, and source-immutability checks.",
        "Inference uses five source-training seeds as the independent replication units. Experimental rows are not treated as independent samples.",
        "",
        "## Method summaries",
        "",
        "| Method | Mean Habs | 95% CI | Mean Hupd | Mean target accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = analysis["summaries"][method]
        habs = summary["absolute_hysteresis"]
        lines.append(
            f"| {method.upper()} | {habs['mean']:.6f} | [{habs['ci95_low']:.6f}, {habs['ci95_high']:.6f}] | "
            f"{summary['update_induced_hysteresis']['mean']:.6f} | {summary['target_accuracy']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Planned Habs contrasts",
            "",
            "Positive values mean the alternative condition has greater return-to-source loss.",
            "",
            "| Method | Contrast | Mean difference | 95% CI | Exact p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in primary:
        lines.append(
            f"| {item['method'].upper()} | {item['contrast']} | {item['mean_difference']:.6f} | "
            f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] | "
            f"{item['p_value_two_sided']:.4f} | {item['p_value_holm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            analysis["claim_boundary"]["supported"],
            "",
            analysis["claim_boundary"]["not_supported"],
            "",
            analysis["claim_boundary"]["inference_limit"],
            "",
        ]
    )
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    freeze = {
        "schema_version": "1.0.0",
        "campaign_id": status["campaign_id"],
        "status": "FROZEN_PRIMARY_RESULTS",
        "frozen_at": utc(),
        "status_sha256": sha256(status_path),
        "result_root_digest": _result_root_digest(status["completed"]),
        "shard_count": len(status["completed"]),
        "provenance": {field: status.get(field) for field in admission.PROVENANCE_FIELDS},
        "device": status.get("device"),
        "preserved_interrupted_jobs": status.get("preserved_failures", []),
        "unresolved_failed_jobs": status.get("failed", []),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    _write_json(output / "campaign_freeze.json", freeze)

    manifest = {
        "schema_version": "1.0.0",
        "analysis_script_sha256": sha256(Path(__file__)),
        "artifacts": [],
    }
    for path in sorted(output.iterdir()):
        if path.name != "artifact_manifest.json":
            manifest["artifacts"].append(
                {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            )
    _write_json(output / "artifact_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=Path,
        default=ROOT / "run_state/publication_trajectory_ablation_gpu_v1_status.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/publication_expansion/trajectory_ablation_gpu_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/analysis/trajectory_ablation_gpu_v1",
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, audit = load_verified_rows(args.status, args.results)
    analysis = build_analysis(rows, audit, draws=args.draws, seed=args.seed)
    write_outputs(args.output, rows, analysis, status_path=args.status, force=args.force)
    print(
        json.dumps(
            {
                "status": analysis["analysis_status"],
                "rows": len(rows),
                "contrasts": len(analysis["contrasts"]),
                "output": str(args.output),
                "source_control": analysis["source_control"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if analysis["analysis_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
