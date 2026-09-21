from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
        self.bn = torch.nn.BatchNorm1d(64, affine=True, track_running_stats=False)
        self.fc2 = torch.nn.Linear(64, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def shift(x: torch.Tensor, family: str, severity: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    image = x.reshape(-1, 1, 8, 8)
    if family == "noise":
        out = image + (0.08 * severity) * torch.randn(image.shape, generator=g)
    elif family == "invert_noise":
        out = 1.0 - image + (0.08 * severity) * torch.randn(image.shape, generator=g)
    elif family == "rotation":
        out = torch.rot90(image, k=severity, dims=(-2, -1))
    elif family == "blur":
        out = torch.nn.functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        if severity == 2:
            out = torch.nn.functional.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
    else:
        raise ValueError(f"unknown shift family: {family}")
    return out.clamp(0.0, 1.0).reshape(-1, 64)


def entropy(model: Net, x: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(model(x), dim=1)
    return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()


def evaluate(model: Net, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
        conf = p.max(1).values
        correct = pred.eq(y)
        bins = torch.linspace(0, 1, 11)
        ece = torch.tensor(0.0)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                ece += mask.float().mean() * (conf[mask].mean() - correct[mask].float().mean()).abs()
        return {
            "accuracy": float(correct.float().mean()),
            "entropy": float((-(p * torch.log(p.clamp_min(1e-8))).sum(1)).mean()),
            "ece": float(ece),
        }


def vec(model: Net) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def adapt(model: Net, source_state: dict, stages: list[tuple[str, torch.Tensor]], method: str, cfg: dict) -> dict:
    model.train()
    source_vec = torch.cat([v.detach().flatten() for v in source_state.values()])
    eta = float(cfg["eta"])
    lam = float(cfg["anchor_lambda"])
    decay = float(cfg["ema_decay"])
    restore_p = float(cfg["restore_probability"])
    stage_passes = int(cfg.get("stage_passes", 1))
    if stage_passes < 1:
        raise ValueError("stage_passes must be at least 1")
    step = 0
    previous_grad = None
    gradient_cosines: list[float] = []
    update_disagreements: list[float] = []
    stage_records: list[dict] = []
    for pass_index in range(stage_passes):
        for stage_name, x in stages:
            before = vec(model)
            for start in range(0, len(x), 64):
                batch = x[start:start + 64]
                if method == "source":
                    step += 1
                    continue
                loss = entropy(model, batch)
                if method == "anchor":
                    loss = loss + lam * sum((p - source_state[n]).pow(2).mean() for n, p in model.named_parameters())
                grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=False)
                flat_grad = torch.cat([g.detach().flatten() for g in grads])
                if previous_grad is not None:
                    gradient_cosines.append(float(torch.nn.functional.cosine_similarity(flat_grad, previous_grad, dim=0)))
                    update_disagreements.append(float((flat_grad / (flat_grad.norm() + 1e-8) - previous_grad / (previous_grad.norm() + 1e-8)).norm()))
                previous_grad = flat_grad
                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p.add_(g, alpha=-eta)
                    if method == "ema":
                        for n, p in model.named_parameters():
                            p.mul_(decay).add_(source_state[n], alpha=1.0 - decay)
                    elif method == "stochastic_restore":
                        for n, p in model.named_parameters():
                            mask = torch.rand(p.shape) < restore_p
                            p.copy_(torch.where(mask, source_state[n], p))
                    elif method == "reset" and step % 8 == 7:
                        model.load_state_dict(source_state)
                step += 1
            after = vec(model)
            stage_records.append({"stage": f"{stage_name}_pass{pass_index + 1}", "parameter_drift": float((after - before).norm()), "steps": step})
    return {
        "stages": stage_records,
        "stage_passes": stage_passes,
        "gradient_cosine_mean": float(np.mean(gradient_cosines)) if gradient_cosines else 0.0,
        "gradient_cosine_min": float(np.min(gradient_cosines)) if gradient_cosines else 0.0,
        "update_disagreement_mean": float(np.mean(update_disagreements)) if update_disagreements else 0.0,
        "adaptation_steps": step,
        "final_parameter_drift": float((vec(model) - source_vec).norm()),
    }


def run(args: argparse.Namespace) -> dict:
    seed_all(args.seed)
    data = load_digits()
    x = torch.tensor(data.data / 16.0, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.30, random_state=123, stratify=y)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xtr), ytr)
        loss.backward()
        optimizer.step()
    source_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    a = shift(xte, args.shift_family, args.severity, 10000 + args.seed)
    b = shift(xte, "invert_noise" if args.shift_family != "invert_noise" else "noise", args.severity, 20000 + args.seed)
    c = shift(xte, "blur" if args.shift_family != "blur" else "rotation", args.severity, 30000 + args.seed)
    source_eval = evaluate(model, a, yte)
    trajectory = [("A_1", a), ("B", b), ("A_2", a)] if args.trajectory == "A-B-A" else [("A_1", a), ("B", b), ("C", c), ("A_2", a)]
    started = time.perf_counter()
    adaptation = adapt(model, source_state, trajectory, args.method, {"eta": 0.01, "anchor_lambda": 0.05, "ema_decay": 0.99, "restore_probability": 0.1, "stage_passes": 2})
    return_eval = evaluate(model, a, yte)
    target_eval = evaluate(model, b, yte)
    result = {
        "schema_version": "1.0.0",
        "evidence_label": "FORMAL",
        "job_id": args.job_id,
        "seed": args.seed,
        "method": args.method,
        "trajectory": args.trajectory,
        "shift_family": args.shift_family,
        "severity": args.severity,
        "dataset": "sklearn-digits",
        "started_at": now(),
        "source_eval_A": source_eval,
        "return_eval_A": return_eval,
        "target_eval_B": target_eval,
        "absolute_hysteresis": source_eval["accuracy"] - return_eval["accuracy"],
        "relative_hysteresis": (source_eval["accuracy"] - return_eval["accuracy"]) / max(source_eval["accuracy"], 1e-8),
        "adaptation": adaptation,
        "wall_clock_seconds": time.perf_counter() - started,
        "labels_used_during_adaptation": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "threads": 1},
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--trajectory", required=True, choices=["A-B-A", "A-B-C-A"])
    parser.add_argument("--shift-family", required=True, choices=["noise", "invert_noise", "rotation", "blur"])
    parser.add_argument("--severity", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({"job_id": args.job_id, "seed": args.seed, "method": args.method}, args.checkpoint)
    print(json.dumps({"status": "PASS", "job_id": args.job_id, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
