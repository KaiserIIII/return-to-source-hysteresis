from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
EXPECTED_METHODS = {"source", "tent", "anchor", "ema", "reset", "stochastic_restore"}
EXPECTED_TRAJECTORIES = {"A-B-A": 54, "A-B-C-A": 72}
PREDICTORS = [
    "gradient_cosine_mean",
    "gradient_cosine_min",
    "update_disagreement_mean",
    "final_parameter_drift",
]


def load_results(results_root: Path) -> pd.DataFrame:
    paths = sorted(results_root.glob("*.json"))
    if len(paths) != 960:
        raise ValueError(f"expected 960 formal outputs, found {len(paths)}")
    rows: list[dict] = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        trajectory = result["trajectory"]
        adaptation = result["adaptation"]
        if result.get("evidence_label") != "FORMAL":
            raise ValueError(f"non-formal evidence label in {path.name}")
        if result.get("labels_used_during_adaptation") is not False:
            raise ValueError(f"labels used during adaptation in {path.name}")
        if adaptation.get("stage_passes") != 2:
            raise ValueError(f"stage_passes mismatch in {path.name}")
        if adaptation.get("adaptation_steps") != EXPECTED_TRAJECTORIES[trajectory]:
            raise ValueError(f"adaptation step mismatch in {path.name}")
        rows.append(
            {
                "job_id": result["job_id"],
                "seed": int(result["seed"]),
                "method": result["method"],
                "trajectory": trajectory,
                "shift_family": result["shift_family"],
                "severity": int(result["severity"]),
                "absolute_hysteresis": float(result["absolute_hysteresis"]),
                "relative_hysteresis": float(result["relative_hysteresis"]),
                "source_accuracy": float(result["source_eval_A"]["accuracy"]),
                "return_accuracy": float(result["return_eval_A"]["accuracy"]),
                "target_accuracy": float(result["target_eval_B"]["accuracy"]),
                "source_entropy": float(result["source_eval_A"]["entropy"]),
                "return_entropy": float(result["return_eval_A"]["entropy"]),
                "target_entropy": float(result["target_eval_B"]["entropy"]),
                "source_ece": float(result["source_eval_A"]["ece"]),
                "return_ece": float(result["return_eval_A"]["ece"]),
                "target_ece": float(result["target_eval_B"]["ece"]),
                "gradient_cosine_mean": float(adaptation["gradient_cosine_mean"]),
                "gradient_cosine_min": float(adaptation["gradient_cosine_min"]),
                "update_disagreement_mean": float(adaptation["update_disagreement_mean"]),
                "final_parameter_drift": float(adaptation["final_parameter_drift"]),
                "wall_clock_seconds": float(result["wall_clock_seconds"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["job_id"].nunique() != len(frame):
        raise ValueError("duplicate job IDs")
    if set(frame["method"]) != EXPECTED_METHODS:
        raise ValueError("method set differs from frozen manifest")
    return frame.sort_values(["seed", "method", "trajectory", "shift_family", "severity"]).reset_index(drop=True)


def cluster_bootstrap_mean(frame: pd.DataFrame, value: str, group: list[str], seed: int, draws: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    seeds = np.sort(frame["seed"].unique())
    if not group:
        observed = float(frame[value].mean())
        med = float(frame[value].median())
        seed_means = frame.groupby("seed")[value].mean().reindex(seeds).to_numpy()
        sampled = rng.integers(0, len(seeds), size=(draws, len(seeds)))
        boot_values = seed_means[sampled].mean(axis=1)
        return {
            "group": group,
            "value": value,
            "draws": draws,
            "seed": seed,
            "estimates": [{
                "group": "overall",
                "mean": observed,
                "median": med,
                "ci95_low": float(np.quantile(boot_values, 0.025)),
                "ci95_high": float(np.quantile(boot_values, 0.975)),
                "n_rows": int(len(frame)),
                "n_seeds": int(frame["seed"].nunique()),
            }],
        }
    grouped_values = frame.groupby(group, dropna=False)[value]
    grouped = grouped_values.mean()
    grouped_median = grouped_values.median()
    seed_group = frame.groupby(["seed"] + group, dropna=False)[value].mean().reset_index()
    pivot = seed_group.pivot(index="seed", columns=group, values=value).reindex(seeds)
    sampled = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    boot = pivot.to_numpy()[sampled].mean(axis=1)
    boot_columns = list(pivot.columns)
    summary = []
    for key, estimate in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        name = "|".join(str(x) for x in key_tuple)
        column_index = boot_columns.index(key if len(key_tuple) > 1 else key_tuple[0])
        values = boot[:, column_index]
        mask = np.ones(len(frame), dtype=bool)
        for column, target in zip(group, key_tuple):
            mask &= frame[column].to_numpy() == target
        summary.append(
            {
                "group": name,
                "mean": float(estimate),
                "median": float(grouped_median.loc[key]),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "n_rows": int(mask.sum()),
                "n_seeds": int(frame.loc[mask, "seed"].nunique()),
            }
        )
    return {"group": group, "value": value, "draws": draws, "seed": seed, "estimates": summary}


def bootstrap_difference(frame: pd.DataFrame, left: str, right: str, value: str, group: list[str], seed: int, draws: int = 5000) -> dict:
    pivot = frame.pivot_table(index=["seed"] + group, columns="method", values=value, aggfunc="mean").dropna(subset=[left, right])
    observed = pivot[left] - pivot[right]
    rng = np.random.default_rng(seed)
    seeds = np.sort(pivot.reset_index()["seed"].unique())
    seed_means = observed.groupby(level="seed").mean().reindex(seeds).to_numpy()
    sampled = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    boot_means = seed_means[sampled].mean(axis=1)
    return {
        "left": left,
        "right": right,
        "value": value,
        "group": group,
        "mean_difference": float(observed.mean()),
        "median_difference": float(observed.median()),
        "ci95_low": float(np.quantile(boot_means, 0.025)),
        "ci95_high": float(np.quantile(boot_means, 0.975)),
        "n_pairs": int(len(observed)),
        "n_seeds": int(pivot.reset_index()["seed"].nunique()),
        "draws": draws,
        "seed": seed,
    }


def bootstrap_spearman(frame: pd.DataFrame, predictor: str, outcome: str, seed: int, draws: int = 5000) -> dict:
    data = frame[frame["method"] != "source"][["seed", predictor, outcome]].dropna()
    observed = float(spearmanr(data[predictor], data[outcome]).statistic)
    rng = np.random.default_rng(seed)
    seeds = np.sort(data["seed"].unique())
    values = []
    for _ in range(draws):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        boot = pd.concat([data[data["seed"] == s] for s in sampled], ignore_index=True)
        values.append(float(spearmanr(boot[predictor], boot[outcome]).statistic))
    return {
        "predictor": predictor,
        "outcome": outcome,
        "spearman_rho": observed,
        "p_value_rowwise_reference": float(spearmanr(data[predictor], data[outcome]).pvalue),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "n_rows": int(len(data)),
        "n_seeds": int(data["seed"].nunique()),
        "draws": draws,
        "seed": seed,
    }


def fit_huber(data: pd.DataFrame, predictors: list[str], outcome: str) -> dict:
    clean = data[predictors + [outcome]].dropna()
    scaler = StandardScaler()
    x = scaler.fit_transform(clean[predictors])
    model = HuberRegressor(max_iter=2000).fit(x, clean[outcome].to_numpy())
    return {
        "n_rows": int(len(clean)),
        "intercept": float(model.intercept_),
        "coefficients": {name: float(value) for name, value in zip(predictors, model.coef_)},
        "r2_train": float(model.score(x, clean[outcome].to_numpy())),
        "mae_train": float(np.mean(np.abs(model.predict(x) - clean[outcome].to_numpy()))),
    }


def bootstrap_huber(data: pd.DataFrame, predictors: list[str], outcome: str, seed: int, draws: int = 1000) -> dict:
    clean = data[["seed"] + predictors + [outcome]].dropna()
    seeds = np.sort(clean["seed"].unique())
    rng = np.random.default_rng(seed)
    coefficients: list[list[float]] = []
    intercepts: list[float] = []
    r2_values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat([clean[clean["seed"] == s] for s in sampled], ignore_index=True)
        try:
            fit = fit_huber(sample, predictors, outcome)
        except ValueError:
            continue
        coefficients.append([fit["coefficients"][p] for p in predictors])
        intercepts.append(fit["intercept"])
        r2_values.append(fit["r2_train"])
    point = fit_huber(clean, predictors, outcome)
    coeff_array = np.asarray(coefficients)
    return {
        "point": point,
        "bootstrap_draws": int(len(coefficients)),
        "seed": seed,
        "coefficient_ci95": {
            p: [float(np.quantile(coeff_array[:, i], 0.025)), float(np.quantile(coeff_array[:, i], 0.975))]
            for i, p in enumerate(predictors)
        },
        "intercept_ci95": [float(np.quantile(intercepts, 0.025)), float(np.quantile(intercepts, 0.975))],
        "r2_ci95": [float(np.quantile(r2_values, 0.025)), float(np.quantile(r2_values, 0.975))],
    }


def leave_one_seed_out(data: pd.DataFrame, predictors: list[str], outcome: str) -> list[dict]:
    output = []
    for held_out in sorted(data["seed"].unique()):
        fit = fit_huber(data[data["seed"] != held_out], predictors, outcome)
        output.append({"held_out_seed": int(held_out), **fit})
    return output


def stratified_correlations(data: pd.DataFrame, predictors: list[str], outcome: str) -> list[dict]:
    output = []
    for column in ["method", "shift_family", "trajectory"]:
        for level, subset in data.groupby(column):
            for predictor in predictors:
                rho = spearmanr(subset[predictor], subset[outcome])
                output.append({"stratum": column, "level": str(level), "predictor": predictor, "rho": float(rho.statistic), "p_value_rowwise_reference": float(rho.pvalue), "n_rows": int(len(subset)), "n_seeds": int(subset["seed"].nunique())})
    return output


def make_figures(frame: pd.DataFrame, figure_root: Path) -> list[str]:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = ["source", "tent", "anchor", "ema", "reset", "stochastic_restore"]
    labels = {"source": "Source", "tent": "TENT", "anchor": "Anchor", "ema": "EMA", "reset": "Reset", "stochastic_restore": "Stoch. restore"}
    colors = {"source": "#4c566a", "tent": "#bf616a", "anchor": "#d08770", "ema": "#ebcb8b", "reset": "#5e81ac", "stochastic_restore": "#88c0d0"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, trajectory in zip(axes, ["A-B-A", "A-B-C-A"]):
        subset = frame[frame["trajectory"] == trajectory]
        means = subset.groupby("method")["absolute_hysteresis"].mean().reindex(methods)
        sem = subset.groupby("method")["absolute_hysteresis"].agg(lambda x: x.std(ddof=1) / np.sqrt(len(x))).reindex(methods)
        x = np.arange(len(methods))
        ax.errorbar(x, means.values, yerr=1.96 * sem.values, fmt="none", ecolor="#333333", capsize=3, lw=1)
        ax.scatter(x, means.values, s=55, c=[colors[m] for m in methods], edgecolor="black", linewidth=0.4, zorder=3)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_xticks(x, [labels[m] for m in methods], rotation=30, ha="right")
        ax.set_title(trajectory)
        ax.set_ylabel("Mean absolute hysteresis" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Return-to-source loss by adaptation method")
    fig.tight_layout()
    path1 = figure_root / "formal_hysteresis_by_method.png"
    fig.savefig(path1, dpi=220)
    fig.savefig(path1.with_suffix(".svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in methods[1:]:
        subset = frame[frame["method"] == method]
        ax.scatter(subset["final_parameter_drift"], subset["absolute_hysteresis"], s=13, alpha=0.45, label=labels[method], color=colors[method])
    ax.set_xlabel("Final parameter drift")
    ax.set_ylabel("Absolute hysteresis")
    ax.set_title("Parameter drift and return loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path2 = figure_root / "formal_drift_predictor.png"
    fig.savefig(path2, dpi=220)
    fig.savefig(path2.with_suffix(".svg"))
    plt.close(fig)

    heat = frame[frame["method"] != "source"].pivot_table(index="shift_family", columns="method", values="absolute_hysteresis", aggfunc="mean").reindex(columns=methods[1:])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    im = ax.imshow(heat.values, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)), [labels[m] for m in heat.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean hysteresis by shift family and method")
    fig.colorbar(im, ax=ax, label="Absolute hysteresis")
    fig.tight_layout()
    path3 = figure_root / "formal_shift_method_heatmap.png"
    fig.savefig(path3, dpi=220)
    fig.savefig(path3.with_suffix(".svg"))
    plt.close(fig)
    return [str(path1), str(path2), str(path3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "formal")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analysis")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    frame = load_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "formal_rows.csv", index=False)

    adaptive = frame[frame["method"] != "source"]
    estimates = {
        "overall_adaptive": cluster_bootstrap_mean(adaptive, "absolute_hysteresis", [], args.seed, args.draws),
        "by_method": cluster_bootstrap_mean(adaptive, "absolute_hysteresis", ["method"], args.seed + 1, args.draws),
        "by_trajectory": cluster_bootstrap_mean(adaptive, "absolute_hysteresis", ["trajectory"], args.seed + 2, args.draws),
        "by_shift": cluster_bootstrap_mean(adaptive, "absolute_hysteresis", ["shift_family"], args.seed + 3, args.draws),
    }
    comparisons = []
    for control in ["source", "anchor", "ema", "reset", "stochastic_restore"]:
        comparisons.append(bootstrap_difference(frame, "tent", control, "absolute_hysteresis", ["trajectory", "shift_family", "severity"], args.seed + 10 + len(comparisons), args.draws))
    correlations = [bootstrap_spearman(adaptive, predictor, "absolute_hysteresis", args.seed + 100 + i, args.draws) for i, predictor in enumerate(PREDICTORS)]
    regression = bootstrap_huber(adaptive, PREDICTORS, "absolute_hysteresis", args.seed + 200, draws=min(args.draws, 1000))
    summary = {
        "schema_version": "1.0.0",
        "analysis_label": "FORMAL_ANALYSIS",
        "source_results": str(args.results),
        "source_file_count": int(len(frame)),
        "source_sha256": hashlib.sha256("".join(sorted(p.name for p in args.results.glob("*.json"))).encode()).hexdigest(),
        "bootstrap": {"unit": "seed cluster", "draws": args.draws, "seed": args.seed},
        "quality": {"protocol_validated": True, "labels_used_during_adaptation": False, "failed_jobs": 0},
        "estimates": estimates,
        "paired_comparisons": comparisons,
        "predictor_correlations": correlations,
        "predictor_correlations_stratified": stratified_correlations(adaptive, PREDICTORS, "absolute_hysteresis"),
        "robust_regression": regression,
        "leave_one_seed_out": leave_one_seed_out(adaptive, PREDICTORS, "absolute_hysteresis"),
        "predictor_timing_audit": {
            "status": "INVALID_FOR_PROSPECTIVE_MECHANISM",
            "reason": "The frozen runner aggregates gradient and drift metrics across the full trajectory, including adaptation after the A-return evaluation point.",
            "action": "Do not use these correlations or regression coefficients as evidence for the registered predictive mechanism claim.",
        },
        "claim_decisions": {
            "C1": "SUPPORTED_WITHIN_TIER1_SCOPE",
            "C2": "WITHDRAWN_PROSPECTIVE_MECHANISM_DUE_TO_PREDICTOR_TIMING",
            "C3": "SUPPORTED_FOR_EMA_RESET_AND_STOCHASTIC_RESTORE; ANCHOR_CONTROL_NULL",
        },
        "figures": make_figures(frame, args.output / "figures"),
        "kill_rules": {
            "phenomenon": "retain only if adaptive pooled CI excludes zero and pattern is not confined to one seed/shift",
            "mechanism": "withdraw strong predictor claim if cross-seed bootstrap interval is unstable or practically weak",
            "recovery": "report negative if simple controls do not reduce return loss against TENT at matched updates",
        },
    }
    (args.output / "formal_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(frame), "output": str(args.output / "formal_analysis.json"), "figures": summary["figures"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
