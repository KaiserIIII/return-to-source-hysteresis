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
PREDICTORS = [
    "prediction_entropy_before_update",
    "gradient_norm_before_update",
    "update_norm",
    "parameter_drift",
]

# The frozen V2.1 runner predates the manifest's corrected evidence label. The
# raw shards remain immutable; this narrowly scoped alias keeps the mismatch
# visible in the provenance audit instead of silently rewriting shard metadata.
EVIDENCE_LABEL_ALIASES = {
    ("FORMAL_EXPANSION", "FORMAL_EXPANSION_CORRECTED"): "frozen_runner_metadata_label",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=no_duplicate_keys,
    )


def audit_evidence_labels(observed_labels: Iterable[str], expected_label: str) -> dict:
    labels = sorted({str(label) for label in observed_labels})
    if labels == [expected_label]:
        return {
            "status": "PASS",
            "expected_label": expected_label,
            "observed_labels": labels,
            "alias_applied": False,
        }
    if len(labels) == 1 and (labels[0], expected_label) in EVIDENCE_LABEL_ALIASES:
        return {
            "status": "CONDITIONAL_METADATA_ALIAS",
            "expected_label": expected_label,
            "observed_labels": labels,
            "alias_applied": True,
            "alias_reason": EVIDENCE_LABEL_ALIASES[(labels[0], expected_label)],
            "raw_shards_unchanged": True,
        }
    raise ValueError(
        f"evidence label mismatch: expected {expected_label!r}, observed {labels!r}"
    )


def atomic_write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force for a verified regeneration")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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
        extreme += statistic >= observed - 1e-15
    return {
        "mean_difference": float(values.mean()),
        "n_seeds": int(len(values)),
        "permutations": total,
        "p_value_two_sided": extreme / total,
    }


def cluster_bootstrap_ci(seed_values: dict[int, float], *, draws: int, rng_seed: int) -> dict:
    if not seed_values:
        raise ValueError("seed_values cannot be empty")
    values = np.asarray([seed_values[key] for key in sorted(seed_values)], dtype=float)
    rng = np.random.default_rng(rng_seed)
    sampled = rng.integers(0, len(values), size=(draws, len(values)))
    distribution = values[sampled].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "n_seeds": int(len(values)),
        "draws": int(draws),
        "rng_seed": int(rng_seed),
    }


def expected_jobs(manifest: dict) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for dataset in manifest["datasets"]:
        for seed in manifest["seeds"]:
            for method in manifest["methods"]:
                for trajectory in manifest["trajectories"]:
                    for family in manifest["shift_families"]:
                        for severity in manifest["severities"]:
                            job_id = f"{dataset}_s{seed:02d}_{method}_{trajectory.replace('-', '')}_{family}_v{severity}"
                            output[job_id] = {
                                "dataset": dataset,
                                "seed": seed,
                                "method": method,
                                "trajectory": trajectory,
                                "shift_family": family,
                                "severity": severity,
                            }
    return output


def verify_dataset_provenance(project_root: Path, manifest_path: Path) -> dict:
    value = read_json(manifest_path)
    checked = []
    for item in value["files"]:
        path = project_root / item["path"]
        if not path.is_file():
            raise ValueError(f"dataset provenance file is missing: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"dataset size mismatch: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"dataset hash mismatch: {path}")
        checked.append({"path": item["path"], "sha256": actual})
    return {"manifest_sha256": sha256(manifest_path), "files_checked": len(checked)}


def _check_finite(value: object, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{location}[{index}]")


def load_verified_results(
    status_path: Path,
    manifest_path: Path,
    results_root: Path,
    *,
    project_root: Path = ROOT,
) -> tuple[list[dict], dict]:
    status = read_json(status_path)
    if status.get("status") != "COMPLETED":
        raise ValueError(f"campaign is not complete: {status.get('status')}")
    manifest = read_json(manifest_path)
    if status.get("result_set") != manifest.get("result_set"):
        raise ValueError("result_set differs between status and manifest")

    dataset_manifest = project_root / manifest["dataset_provenance_file"]
    current_provenance = {
        "manifest_sha256": sha256(manifest_path),
        "runner_sha256": sha256(project_root / "expansion_experiment.py"),
        "supervisor_sha256": sha256(project_root / "run_expansion_supervisor.py"),
        "dataset_provenance_sha256": sha256(dataset_manifest),
    }
    for field, expected in current_provenance.items():
        if status.get(field) != expected:
            raise ValueError(
                f"campaign provenance mismatch for {field}: status={status.get(field)!r}, current={expected!r}"
            )
    dataset_audit = verify_dataset_provenance(project_root, dataset_manifest)

    expected = expected_jobs(manifest)
    completed = status.get("completed", {})
    if set(completed) != set(expected):
        missing = sorted(set(expected) - set(completed))
        extra = sorted(set(completed) - set(expected))
        raise ValueError(f"job ledger mismatch: missing={missing[:5]}, extra={extra[:5]}")
    if status.get("failed"):
        raise ValueError(f"campaign has current failures: {len(status['failed'])}")

    checkpoint_hashes: dict[tuple[str, int], str] = {}
    rows: list[dict] = []
    observed_evidence_labels: set[str] = set()
    for job_id, factors in sorted(expected.items()):
        path = results_root / f"{job_id}.json"
        if not path.is_file():
            raise ValueError(f"missing result file: {path}")
        actual_hash = sha256(path)
        if completed[job_id] != actual_hash:
            raise ValueError(f"result hash mismatch: {job_id}")
        result = read_json(path)
        _check_finite(result, job_id)
        for field, expected_value in factors.items():
            if result.get(field) != expected_value:
                raise ValueError(f"factor mismatch for {job_id}.{field}")
        if result.get("job_id") != job_id or result.get("status") != "PASS":
            raise ValueError(f"invalid result identity/status: {job_id}")
        observed_evidence_labels.add(str(result.get("evidence_label")))
        if result.get("runner_sha256") != current_provenance["runner_sha256"]:
            raise ValueError(f"runner hash mismatch in result: {job_id}")
        if result.get("dataset_provenance_sha256") != current_provenance["dataset_provenance_sha256"]:
            raise ValueError(f"dataset provenance mismatch in result: {job_id}")
        expected_parameters = {
            "batch_size": manifest["batch_size"],
            "adaptation_passes": manifest["adaptation_passes"],
            "adaptation_learning_rate": manifest["adaptation_learning_rate"],
        }
        if result.get("protocol_parameters") != expected_parameters:
            raise ValueError(f"protocol parameter mismatch: {job_id}")
        episode = result["episode"]
        if episode.get("labels_used_during_adaptation") is not False:
            raise ValueError(f"labels used during adaptation: {job_id}")
        target = episode["target_stream_evaluation"]
        if target.get("timing") != "pre_update_online_predictions":
            raise ValueError(f"target metric timing mismatch: {job_id}")
        batches = math.ceil(10000 / manifest["batch_size"])
        stages = 1 if factors["trajectory"] == "A-B-A" else 2
        expected_steps = batches * stages * manifest["adaptation_passes"]
        if episode.get("adaptation_steps_before_return") != expected_steps:
            raise ValueError(f"adaptation step mismatch: {job_id}")
        if target.get("examples") != 10000 * stages * manifest["adaptation_passes"]:
            raise ValueError(f"target example count mismatch: {job_id}")
        expected_sequence = [factors["shift_family"]]
        if stages == 2:
            expected_sequence.append(manifest["abca_c_domain"])
        if episode.get("shift_sequence") != expected_sequence:
            raise ValueError(f"shift sequence mismatch: {job_id}")
        mechanism_step = episode["prospective_mechanism"]["measurement_step"]
        return_step = episode["return_outcome"]["measurement_step"]
        if mechanism_step != expected_steps or return_step != expected_steps + 1:
            raise ValueError(f"prospective timing mismatch: {job_id}")

        checkpoint_key = (factors["dataset"], factors["seed"])
        if checkpoint_key not in checkpoint_hashes:
            checkpoint_path = (
                project_root
                / "checkpoints"
                / "expansion"
                / "formal"
                / f"{factors['dataset']}_s{factors['seed']:02d}.pt"
            )
            checkpoint_hashes[checkpoint_key] = sha256(checkpoint_path)
        if result.get("source_checkpoint_sha256") != checkpoint_hashes[checkpoint_key]:
            raise ValueError(f"source checkpoint mismatch: {job_id}")

        checkpoint_source = episode["checkpoint_source_evaluation"]
        configured_source = episode["configured_source_evaluation"]
        returned = episode["return_outcome"]
        total_loss = checkpoint_source["accuracy"] - returned["accuracy_before_return_adaptation"]
        update_loss = configured_source["accuracy"] - returned["accuracy_before_return_adaptation"]
        if not math.isclose(total_loss, returned["absolute_hysteresis_before_return_adaptation"], abs_tol=1e-12):
            raise ValueError(f"primary endpoint arithmetic mismatch: {job_id}")
        if not math.isclose(update_loss, returned["update_induced_hysteresis_before_return_adaptation"], abs_tol=1e-12):
            raise ValueError(f"update endpoint arithmetic mismatch: {job_id}")
        if factors["method"] == "source" and (
            abs(total_loss) > 1e-12 or abs(update_loss) > 1e-12
        ):
            raise ValueError(f"static source invariant failed: {job_id}")

        mechanism = episode["prospective_mechanism"]
        rows.append(
            factors
            | {
                "job_id": job_id,
                "checkpoint_source_accuracy": checkpoint_source["accuracy"],
                "configured_source_accuracy": configured_source["accuracy"],
                "configuration_gap": episode["configuration_gap"],
                "return_accuracy_before": returned["accuracy_before_return_adaptation"],
                "return_accuracy_after": returned["accuracy_after_return_adaptation"],
                "absolute_hysteresis": total_loss,
                "update_induced_hysteresis": update_loss,
                "target_accuracy": target["accuracy"],
                "target_entropy": target["entropy"],
                "source_ece": checkpoint_source["ece"],
                "wall_clock_seconds": episode["wall_clock_seconds"],
                **{name: mechanism[name] for name in PREDICTORS},
                "result_sha256": actual_hash,
            }
        )

    label_audit = audit_evidence_labels(observed_evidence_labels, manifest["evidence_label"])
    return rows, {
        "status_sha256": sha256(status_path),
        "manifest_sha256": current_provenance["manifest_sha256"],
        "runner_sha256": current_provenance["runner_sha256"],
        "supervisor_sha256": current_provenance["supervisor_sha256"],
        "dataset_provenance_sha256": current_provenance["dataset_provenance_sha256"],
        "dataset_files_checked": dataset_audit["files_checked"],
        "result_files_checked": len(rows),
        "checkpoint_files_checked": len(checkpoint_hashes),
        "preserved_historical_failures": len(status.get("preserved_failures", [])),
        "evidence_label_audit": label_audit,
    }


def _seed_means(rows: list[dict], value: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(float(row[value]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def grouped_summaries(
    rows: list[dict], value: str, group_fields: list[str], *, draws: int, rng_seed: int
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for index, (key, subset) in enumerate(sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))):
        summary = cluster_bootstrap_ci(
            _seed_means(subset, value), draws=draws, rng_seed=rng_seed + index
        )
        output.append({**dict(zip(group_fields, key)), "n_rows": len(subset), **summary})
    return output


def paired_contrasts(
    rows: list[dict],
    value: str,
    *,
    reference: str,
    controls: list[str],
    draws: int,
    rng_seed: int,
) -> list[dict]:
    condition_fields = ["dataset", "seed", "trajectory", "shift_family", "severity"]
    lookup = {
        tuple(row[field] for field in condition_fields) + (row["method"],): float(row[value])
        for row in rows
    }
    output = []
    for index, control in enumerate(controls):
        per_seed: dict[int, list[float]] = defaultdict(list)
        for key in sorted({item[:-1] for item in lookup}):
            left = lookup.get(key + (reference,))
            right = lookup.get(key + (control,))
            if left is not None and right is not None:
                per_seed[int(key[1])].append(left - right)
        seed_differences = {seed: float(np.mean(values)) for seed, values in per_seed.items()}
        bootstrap = cluster_bootstrap_ci(
            seed_differences, draws=draws, rng_seed=rng_seed + index
        )
        permutation = exact_sign_flip(seed_differences.values())
        values = np.asarray(list(seed_differences.values()), dtype=float)
        paired_d = float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) else None
        output.append(
            {
                "reference": reference,
                "control": control,
                "interpretation": "positive means reference has larger loss",
                "paired_effect_dz": paired_d,
                **bootstrap,
                **permutation,
            }
        )
    adjusted = holm_adjust([item["p_value_two_sided"] for item in output])
    for item, value_adjusted in zip(output, adjusted):
        item["p_value_holm"] = value_adjusted
    return output


def _design_matrix(
    rows: list[dict],
    *,
    categorical_levels: dict[str, list[str]] | None = None,
    predictor_stats: dict[str, tuple[float, float]] | None = None,
    include_predictors: bool,
) -> tuple[np.ndarray, dict[str, list[str]], dict[str, tuple[float, float]], list[str]]:
    categorical = ["dataset", "method", "trajectory", "shift_family"]
    if categorical_levels is None:
        categorical_levels = {
            field: sorted({str(row[field]) for row in rows}) for field in categorical
        }
    if predictor_stats is None:
        predictor_stats = {}
        for field in PREDICTORS:
            values = np.asarray([float(row[field]) for row in rows])
            scale = float(values.std(ddof=0)) or 1.0
            predictor_stats[field] = (float(values.mean()), scale)
    columns = ["intercept", "severity"]
    columns.extend(
        f"{field}={level}"
        for field in categorical
        for level in categorical_levels[field][1:]
    )
    if include_predictors:
        columns.extend(PREDICTORS)
    matrix = []
    for row in rows:
        values = [1.0, float(row["severity"])]
        values.extend(
            1.0 if str(row[field]) == level else 0.0
            for field in categorical
            for level in categorical_levels[field][1:]
        )
        if include_predictors:
            values.extend(
                (float(row[field]) - predictor_stats[field][0]) / predictor_stats[field][1]
                for field in PREDICTORS
            )
        matrix.append(values)
    return np.asarray(matrix, dtype=float), categorical_levels, predictor_stats, columns


def mechanism_screen(rows: list[dict], *, train_seeds: set[int], held_seeds: set[int]) -> dict:
    from scipy.stats import spearmanr

    usable = [row for row in rows if row["method"] != "source"]
    train = [row for row in usable if int(row["seed"]) in train_seeds]
    held = [row for row in usable if int(row["seed"]) in held_seeds]
    if not train or not held:
        raise ValueError("mechanism screen requires both training and held-seed rows")
    y_train = np.asarray([row["update_induced_hysteresis"] for row in train])
    y_held = np.asarray([row["update_induced_hysteresis"] for row in held])
    x_control, levels, stats, control_columns = _design_matrix(
        train, include_predictors=False
    )
    x_full, _, _, full_columns = _design_matrix(train, include_predictors=True)
    held_control, _, _, _ = _design_matrix(
        held,
        categorical_levels=levels,
        predictor_stats=stats,
        include_predictors=False,
    )
    held_full, _, _, _ = _design_matrix(
        held,
        categorical_levels=levels,
        predictor_stats=stats,
        include_predictors=True,
    )
    beta_control = np.linalg.lstsq(x_control, y_train, rcond=None)[0]
    beta_full = np.linalg.lstsq(x_full, y_train, rcond=None)[0]
    prediction_control = held_control @ beta_control
    prediction_full = held_full @ beta_full

    def metrics(prediction: np.ndarray) -> dict:
        residual = y_held - prediction
        denominator = float(np.sum((y_held - y_held.mean()) ** 2))
        rho = spearmanr(prediction, y_held)
        return {
            "mae": float(np.mean(np.abs(residual))),
            "r2": 1.0 - float(np.sum(residual**2)) / denominator if denominator else None,
            "spearman_rho": float(rho.statistic),
            "spearman_p_rowwise_reference": float(rho.pvalue),
        }

    control_metrics = metrics(prediction_control)
    full_metrics = metrics(prediction_full)
    delta_r2 = (
        full_metrics["r2"] - control_metrics["r2"]
        if full_metrics["r2"] is not None and control_metrics["r2"] is not None
        else None
    )
    improves = (
        delta_r2 is not None
        and delta_r2 > 0.02
        and full_metrics["mae"] < control_metrics["mae"]
        and full_metrics["spearman_rho"] >= 0.2
    )
    return {
        "status": "SINGLE_SEED_SCREEN_ONLY" if improves else "NOT_SUPPORTED",
        "claim_policy": "No mechanism claim is permitted from one held seed; execute fresh held seeds 5-9.",
        "training_seeds": sorted(train_seeds),
        "held_seeds": sorted(held_seeds),
        "n_train_rows": len(train),
        "n_held_rows": len(held),
        "control_model": {"columns": control_columns, "metrics": control_metrics},
        "predictor_model": {
            "columns": full_columns,
            "coefficients": {name: float(value) for name, value in zip(full_columns, beta_full)},
            "metrics": full_metrics,
        },
        "incremental_r2": delta_r2,
        "incremental_mae_reduction": control_metrics["mae"] - full_metrics["mae"],
        "held_observed": y_held.tolist(),
        "held_predicted_control": prediction_control.tolist(),
        "held_predicted_full": prediction_full.tolist(),
    }


def build_analysis(rows: list[dict], audit: dict, *, draws: int, rng_seed: int) -> dict:
    adaptive = [row for row in rows if row["method"] != "source"]
    methods = sorted({row["method"] for row in rows})
    controls = [method for method in methods if method not in {"source", "tent"}]
    total_by_method = grouped_summaries(
        rows, "absolute_hysteresis", ["method"], draws=draws, rng_seed=rng_seed
    )
    update_by_method = grouped_summaries(
        rows,
        "update_induced_hysteresis",
        ["method"],
        draws=draws,
        rng_seed=rng_seed + 100,
    )
    target_by_method = grouped_summaries(
        rows, "target_accuracy", ["method"], draws=draws, rng_seed=rng_seed + 200
    )
    stratified = {
        "total_loss_by_dataset_method": grouped_summaries(
            rows,
            "absolute_hysteresis",
            ["dataset", "method"],
            draws=draws,
            rng_seed=rng_seed + 300,
        ),
        "update_loss_by_shift_method": grouped_summaries(
            rows,
            "update_induced_hysteresis",
            ["shift_family", "method"],
            draws=draws,
            rng_seed=rng_seed + 500,
        ),
        "update_loss_by_trajectory_method": grouped_summaries(
            rows,
            "update_induced_hysteresis",
            ["trajectory", "method"],
            draws=draws,
            rng_seed=rng_seed + 700,
        ),
    }
    negative_results = [
        {"method": item["method"], "endpoint": "update_induced_hysteresis", "reason": "95% seed-cluster CI includes zero", **item}
        for item in update_by_method
        if item["ci95_low"] <= 0 <= item["ci95_high"]
    ]
    return {
        "schema_version": "1.0.0",
        "analysis_label": "FORMAL_EXPANSION_V2_ANALYSIS",
        "analysis_status": (
            "PASS_WITH_METADATA_ALIAS"
            if audit.get("evidence_label_audit", {}).get("alias_applied")
            else "PASS"
        ),
        "created_at": utc(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "provenance_audit": audit,
        "design": {
            "unit_of_independent_replication": "source-training seed",
            "bootstrap": f"seed-cluster percentile bootstrap, {draws} draws",
            "paired_test": "exact two-sided sign-flip test on seed-level paired means",
            "multiplicity": "Holm within each predefined contrast family",
            "rowwise_p_values": "descriptive only because rows within seed are dependent",
        },
        "counts": {
            "rows": len(rows),
            "adaptive_rows": len(adaptive),
            "seeds": len({row["seed"] for row in rows}),
            "datasets": len({row["dataset"] for row in rows}),
        },
        "summaries": {
            "total_loss_by_method": total_by_method,
            "update_loss_by_method": update_by_method,
            "target_accuracy_by_method": target_by_method,
            **stratified,
        },
        "contrasts": {
            "tent_minus_controls_total_loss": paired_contrasts(
                rows,
                "absolute_hysteresis",
                reference="tent",
                controls=controls,
                draws=draws,
                rng_seed=rng_seed + 1000,
            ),
            "tent_minus_controls_update_loss": paired_contrasts(
                rows,
                "update_induced_hysteresis",
                reference="tent",
                controls=controls,
                draws=draws,
                rng_seed=rng_seed + 1100,
            ),
            "tent_minus_controls_target_accuracy": paired_contrasts(
                rows,
                "target_accuracy",
                reference="tent",
                controls=controls,
                draws=draws,
                rng_seed=rng_seed + 1200,
            ),
        },
        "mechanism_screen": mechanism_screen(
            rows, train_seeds={0, 1, 2, 3}, held_seeds={4}
        ),
        "negative_results": negative_results,
    }


def write_rows(path: Path, rows: list[dict], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_figures(rows: list[dict], analysis: dict, figure_root: Path, *, force: bool) -> list[dict]:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    method_order = ["source", "tent", "anchor", "ema_restore", "periodic_reset"]
    labels = {
        "source": "Source",
        "tent": "TENT",
        "anchor": "Anchor",
        "ema_restore": "EMA restore",
        "periodic_reset": "Periodic reset",
    }
    colors = {
        "source": "#4D4D4D",
        "tent": "#D55E00",
        "anchor": "#CC79A7",
        "ema_restore": "#0072B2",
        "periodic_reset": "#009E73",
    }
    manifests = []

    def export(fig, stem: str, description: str) -> None:
        paths = [figure_root / f"{stem}.{extension}" for extension in ("png", "svg", "pdf")]
        for path in paths:
            if path.exists() and not force:
                raise FileExistsError(f"refusing to overwrite {path}")
        fig.savefig(paths[0], dpi=300)
        fig.savefig(paths[1])
        fig.savefig(paths[2])
        manifests.append(
            {
                "figure": stem,
                "description": description,
                "files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
            }
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    endpoints = [
        ("absolute_hysteresis", "Checkpoint-to-return loss"),
        ("update_induced_hysteresis", "Update-induced return loss"),
    ]
    for panel, (value, ylabel) in zip(axes, endpoints):
        for index, method in enumerate(method_order):
            subset = [row for row in rows if row["method"] == method]
            seeds = _seed_means(subset, value)
            seed_values = np.asarray([seeds[key] for key in sorted(seeds)])
            summary = cluster_bootstrap_ci(seeds, draws=5000, rng_seed=20260907 + index)
            jitter = np.linspace(-0.09, 0.09, len(seed_values))
            panel.scatter(
                index + jitter,
                seed_values,
                s=15,
                facecolors="none",
                edgecolors=colors[method],
                linewidths=0.8,
            )
            panel.errorbar(
                index,
                summary["mean"],
                yerr=[[summary["mean"] - summary["ci95_low"]], [summary["ci95_high"] - summary["mean"]]],
                fmt="o",
                color=colors[method],
                capsize=3,
                markersize=5,
            )
        panel.axhline(0, color="#777777", linewidth=0.8)
        panel.set_xticks(range(len(method_order)), [labels[item] for item in method_order], rotation=28, ha="right")
        panel.set_ylabel(ylabel)
        panel.grid(axis="y", alpha=0.2)
    export(
        fig,
        "expansion_return_loss",
        "Seed means and 95% seed-cluster bootstrap intervals for total and update-induced return loss.",
    )

    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    for method in method_order:
        subset = [row for row in rows if row["method"] == method]
        x = _seed_means(subset, "target_accuracy")
        y = _seed_means(subset, "absolute_hysteresis")
        x_values = np.asarray([x[key] for key in sorted(x)])
        y_values = np.asarray([y[key] for key in sorted(y)])
        ax.scatter(x_values, y_values, s=18, alpha=0.45, color=colors[method])
        ax.scatter(x_values.mean(), y_values.mean(), s=65, marker="X", color=colors[method], label=labels[method])
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Pre-update online target accuracy")
    ax.set_ylabel("Checkpoint-to-return loss")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7)
    export(
        fig,
        "expansion_target_return_pareto",
        "Target utility versus return loss; small points are independent seed means and X marks are method means.",
    )

    shifts = sorted({row["shift_family"] for row in rows})
    matrix = np.asarray(
        [
            [
                np.mean(
                    [
                        row["update_induced_hysteresis"]
                        for row in rows
                        if row["shift_family"] == shift and row["method"] == method
                    ]
                )
                for method in method_order
            ]
            for shift in shifts
        ]
    )
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 1e-9)
    fig, ax = plt.subplots(figsize=(6.2, 3.3), layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(method_order)), [labels[item] for item in method_order], rotation=28, ha="right")
    ax.set_yticks(range(len(shifts)), shifts)
    for row_index in range(len(shifts)):
        for column_index in range(len(method_order)):
            ax.text(column_index, row_index, f"{matrix[row_index, column_index]:.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Update-induced return loss")
    export(
        fig,
        "expansion_shift_method_heatmap",
        "Mean update-induced return loss by shift and method; diverging scale is centered at zero.",
    )

    mechanism = analysis["mechanism_screen"]
    observed = np.asarray(mechanism["held_observed"])
    predicted = np.asarray(mechanism["held_predicted_full"])
    low = min(float(observed.min()), float(predicted.min()))
    high = max(float(observed.max()), float(predicted.max()))
    fig, ax = plt.subplots(figsize=(5.2, 4.0), layout="constrained")
    ax.scatter(predicted, observed, s=15, alpha=0.5, color="#0072B2")
    ax.plot([low, high], [low, high], linestyle="--", color="#777777", linewidth=0.8)
    ax.set_xlabel("Predicted update-induced loss")
    ax.set_ylabel("Observed update-induced loss")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, -3), useMathText=True)
    ax.grid(alpha=0.2)
    export(
        fig,
        "expansion_mechanism_screen",
        "Seed-4 held-screen predictions; this panel is not confirmatory mechanism evidence.",
    )
    return manifests


def render_report(analysis: dict) -> str:
    lines = [
        "# Corrected Expansion Analysis",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "The analysis passed structural, hash, dataset, checkpoint, protocol, timing, and source-invariant checks before computing statistics.",
        "",
        "## Method summaries",
        "",
        "| Method | Total return loss (95% CI) | Update-induced loss (95% CI) | Target accuracy (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    total = {item["method"]: item for item in analysis["summaries"]["total_loss_by_method"]}
    update = {item["method"]: item for item in analysis["summaries"]["update_loss_by_method"]}
    target = {item["method"]: item for item in analysis["summaries"]["target_accuracy_by_method"]}
    for method in sorted(total):
        lines.append(
            f"| {method} | {total[method]['mean']:.4f} [{total[method]['ci95_low']:.4f}, {total[method]['ci95_high']:.4f}] | "
            f"{update[method]['mean']:.4f} [{update[method]['ci95_low']:.4f}, {update[method]['ci95_high']:.4f}] | "
            f"{target[method]['mean']:.4f} [{target[method]['ci95_low']:.4f}, {target[method]['ci95_high']:.4f}] |"
        )
    mechanism = analysis["mechanism_screen"]
    lines.extend(
        [
            "",
            "## Mechanism boundary",
            "",
            f"Status: `{mechanism['status']}`. {mechanism['claim_policy']}",
            "",
            "## Reporting policy",
            "",
            "All null and negative results remain in `analysis.json`. Row-wise p-values in the held-seed screen are descriptive because repeated conditions within a seed are dependent. Formal contrasts use seed-level pairing and Holm correction.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=ROOT / "run_state" / "expansion_formal_v2_status.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "expansion_campaign_v2.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "expansion" / "formal_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analysis" / "expansion_v2")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows, audit = load_verified_results(args.status, args.manifest, args.results)
    analysis = build_analysis(rows, audit, draws=args.draws, rng_seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "rows.csv", rows, force=args.force)
    figure_manifest = make_figures(rows, analysis, args.output / "figures", force=args.force)
    analysis["figures"] = figure_manifest
    atomic_write(
        args.output / "analysis.json",
        json.dumps(analysis, indent=2, allow_nan=False) + "\n",
        force=args.force,
    )
    atomic_write(args.output / "analysis_report.md", render_report(analysis), force=args.force)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": utc(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "artifacts": [
            {"path": str(path.relative_to(args.output)), "sha256": sha256(path)}
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        ],
    }
    atomic_write(
        args.output / "artifact_manifest.json",
        json.dumps(manifest, indent=2) + "\n",
        force=args.force,
    )
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
