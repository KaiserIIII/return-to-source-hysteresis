"""Auditable seed-level analysis for the completed modern-baseline campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: dict[int, float], *, draws: int, seed: int) -> dict:
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=float)
    if len(ordered) == 1:
        distribution = ordered
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(ordered), size=(draws, len(ordered)))
        distribution = ordered[indices].mean(axis=1)
    return {
        "mean": float(ordered.mean()),
        "sd": float(ordered.std(ddof=1)) if len(ordered) > 1 else 0.0,
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "n_seeds": int(len(ordered)),
        "draws": int(draws),
        "rng_seed": int(seed),
    }


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if abs(float(np.mean(np.asarray(values) * np.asarray(signs)))) >= observed - 1e-15:
            extreme += 1
    return float(extreme / (2 ** len(values)))


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        out[index] = running
    return out


def load_rows(project_root: Path = ROOT) -> tuple[list[dict], dict]:
    status_path = project_root / "run_state" / "publication_modern_baselines_v1_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or status.get("failed"):
        raise ValueError("modern_baselines_v1 is not a clean completed campaign")
    rows = []
    provenance = defaultdict(set)
    upstream = {}
    for job_id, expected_hash in status["completed"].items():
        path = project_root / "results" / "publication_expansion" / "modern_baselines_v1" / f"{job_id}.json"
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"missing or hash-mismatched shard: {job_id}")
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("status") != "PASS" or shard.get("job_id") != job_id:
            raise ValueError(f"invalid shard identity/status: {job_id}")
        episode = shard["episode"]
        if episode["events"]["predictor_step"] >= episode["events"]["outcome_step"]:
            raise ValueError(f"temporal ordering failed: {job_id}")
        transition = episode["state_transition"]
        if transition.get("unexpected_mutations"):
            raise ValueError(f"unexpected mutation: {job_id}")
        if shard["method"] == "source" and (
            transition.get("before_digest") != transition.get("after_digest")
            or transition.get("optimizer_step_count") != 0
        ):
            raise ValueError(f"source immutability failed: {job_id}")
        provenance["runner"].add(shard["runner_sha256"])
        provenance["manifest"].add(shard["manifest_sha256"])
        provenance["dataset"].add(shard["dataset_provenance_sha256"])
        provenance["registry"].add(shard["method_registry_sha256"])
        provider = episode["algorithm_semantics"].get("method_upstream", shard.get("method_upstream", {}))
        upstream.setdefault(shard["method"], shard.get("method_upstream", {}))
        outcome = episode["return_outcome"]
        rows.append({
            "job_id": job_id,
            "dataset": shard["dataset"],
            "method": shard["method"],
            "trajectory": shard["trajectory"],
            "seed": int(shard["seed"]),
            "habs": float(outcome["absolute_hysteresis_before_return_adaptation"]),
            "hupd": float(outcome["update_induced_hysteresis_before_return_adaptation"]),
            "target_accuracy": float(episode["target_stream_evaluation"]["accuracy"]),
            "parameter_drift": float(episode["prospective_mechanism"]["parameter_drift"]),
            "result_sha256": expected_hash,
            "runner_sha256": shard["runner_sha256"],
            "manifest_sha256": shard["manifest_sha256"],
            "dataset_provenance_sha256": shard["dataset_provenance_sha256"],
            "method_registry_sha256": shard["method_registry_sha256"],
            "implementation_status": episode["algorithm_semantics"].get("implementation_status", "UNKNOWN"),
        })
    if len(rows) != 60:
        raise ValueError(f"expected 60 rows, found {len(rows)}")
    if any(len(values) != 1 for values in provenance.values()):
        raise ValueError(f"mixed provenance identities: {dict(provenance)}")
    return rows, {
        "status": "PASS",
        "expected_jobs": 60,
        "observed_jobs": len(rows),
        "methods": sorted({row["method"] for row in rows}),
        "trajectories": sorted({row["trajectory"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "source_shards": sum(row["method"] == "source" for row in rows),
        "temporal_ordered_shards": len(rows),
        "provenance": {name: sorted(values)[0] for name, values in provenance.items()},
        "upstream": upstream,
    }


def summarize(rows: list[dict], field: str, *, draws: int, seed: int) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["method"]][row["seed"]].append(float(row[field]))
    output = []
    for index, method in enumerate(sorted(grouped)):
        seed_values = {seed_id: float(np.mean(values)) for seed_id, values in grouped[method].items()}
        output.append({"method": method, "field": field, **bootstrap(seed_values, draws=draws, seed=seed + index), "seed_values": {str(k): v for k, v in sorted(seed_values.items())}})
    return output


def build_analysis(rows: list[dict], audit: dict, *, draws: int, seed: int) -> dict:
    method_summaries = summarize(rows, "habs", draws=draws, seed=seed)
    method_target = summarize(rows, "target_accuracy", draws=draws, seed=seed + 20)
    method_hupd = summarize(rows, "hupd", draws=draws, seed=seed + 40)
    methods = ["tent", "eata", "sar", "cotta", "rotta"]
    source_habs = {x["seed"]: float(np.mean([r["habs"] for r in rows if r["method"] == "source" and r["seed"] == x["seed"]])) for x in rows if x["method"] == "source"}
    contrasts = []
    for method in methods:
        vals = {}
        for seed_id in sorted(source_habs):
            alt = [r["habs"] for r in rows if r["method"] == method and r["seed"] == seed_id]
            vals[seed_id] = float(np.mean(alt) - source_habs[seed_id])
        stats = bootstrap(vals, draws=draws, seed=seed + 100 + methods.index(method))
        stats.update({"method": method, "value": "habs", "seed_differences": {str(k): v for k, v in vals.items()}, "p_value_two_sided": exact_sign_flip(list(vals.values()))})
        contrasts.append(stats)
    adjusted = holm([item["p_value_two_sided"] for item in contrasts])
    for item, value in zip(contrasts, adjusted):
        item["p_value_holm"] = value
    return {
        "schema_version": "1.0.0",
        "analysis_label": "FORMAL_MODERN_BASELINES_V1",
        "analysis_status": "PASS",
        "created_at": utc(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        "integrity_audit": audit,
        "design": {"independent_replication_unit": "source-training seed", "seed_aggregation": "average both trajectories within seed before inference", "bootstrap": f"seed-cluster percentile bootstrap, {draws} draws", "paired_test": "exact two-sided sign-flip over five seed-level differences", "multiplicity": "Holm across five modern-baseline contrasts", "normalization_policy": "bn_affine_frozen_stats"},
        "counts": {"rows": len(rows), "methods": len({r["method"] for r in rows}), "trajectories": len({r["trajectory"] for r in rows}), "seeds": len({r["seed"] for r in rows})},
        "summaries": {"habs": method_summaries, "hupd": method_hupd, "target_accuracy": method_target, "trajectory": [{"trajectory": t, "habs": float(np.mean([r["habs"] for r in rows if r["trajectory"] == t])), "target_accuracy": float(np.mean([r["target_accuracy"] for r in rows if r["trajectory"] == t]))} for t in sorted({r["trajectory"] for r in rows})]},
        "contrasts": contrasts,
        "negative_results": ["The campaign does not support a universal method ranking.", "EATA shows severe instability under the registered independent reimplementation and fixed-normalization protocol; this is an implementation-sensitive signal, not a universal failure claim."],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analysis" / "modern_baselines_v1")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    rows, audit = load_rows()
    analysis = build_analysis(rows, audit, draws=args.draws, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis["artifacts"] = {"raw_status": str(ROOT / "run_state" / "publication_modern_baselines_v1_status.json"), "rows": "rows.csv"}
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    lines = ["# Modern Baseline Campaign Analysis", "", f"The verified campaign covers {audit['observed_jobs']} jobs: six methods, two trajectories, and five source seeds.", "", "| Method | H_abs | Target accuracy | H_upd |", "|---|---:|---:|---:|"]
    by_h = {item["method"]: item for item in analysis["summaries"]["habs"]}
    by_t = {item["method"]: item for item in analysis["summaries"]["target_accuracy"]}
    by_u = {item["method"]: item for item in analysis["summaries"]["hupd"]}
    for method in ["source", "tent", "eata", "sar", "cotta", "rotta"]:
        lines.append(f"| {method} | {by_h[method]['mean']:.6f} [{by_h[method]['ci95_low']:.6f}, {by_h[method]['ci95_high']:.6f}] | {by_t[method]['mean']:.6f} | {by_u[method]['mean']:.6f} |")
    lines += ["", "The methods are independent reimplementations bound to the pinned upstream registry. EATA's instability remains an implementation-sensitivity finding.", ""]
    (args.output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"schema_version": "1.0.0", "created_at": utc(), "analysis_script_sha256": sha256(Path(__file__)), "artifacts": []}
    for path in sorted(args.output.glob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["artifacts"].append({"path": path.name, "sha256": sha256(path)})
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
