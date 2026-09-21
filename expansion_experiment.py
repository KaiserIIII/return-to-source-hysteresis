from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class TinyConvNet(torch.nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(8, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def cifar_resnet18(classes: int) -> torch.nn.Module:
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def shift(x: torch.Tensor, family: str, severity: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    if family == "noise":
        out = x + 0.08 * severity * torch.randn(x.shape, generator=generator, device=x.device)
    elif family == "brightness":
        out = x + 0.12 * severity
    elif family == "contrast":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        out = mean + (x - mean) * max(0.1, 1.0 - 0.25 * severity)
    elif family == "blur":
        out = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
        if severity > 1:
            out = torch.nn.functional.avg_pool2d(out, 3, stride=1, padding=1)
    elif family == "rotation":
        out = torch.rot90(x, k=severity, dims=(-2, -1))
    elif family == "channel_permutation":
        order = [1, 2, 0] if severity == 1 else [2, 0, 1]
        out = x[:, order]
    else:
        raise ValueError(f"unknown shift family: {family}")
    return out.clamp(0.0, 1.0)


def _metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    entropy_sum = 0.0
    ece_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)
            confidence, prediction = prob.max(1)
            hit = prediction.eq(yb)
            total += len(xb)
            correct += int(hit.sum())
            entropy_sum += float((-(prob * prob.clamp_min(1e-8).log()).sum(1)).sum())
            bins = torch.linspace(0, 1, 11, device=x.device)
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (confidence > lo) & (confidence <= hi)
                if mask.any():
                    ece_sum += float(mask.sum() * (confidence[mask].mean() - hit[mask].float().mean()).abs())
    return {"accuracy": correct / max(total, 1), "entropy": entropy_sum / max(total, 1), "ece": ece_sum / max(total, 1)}


def _adaptable(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) and module.affine:
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            selected.extend(((f"{module_name}.weight", module.weight), (f"{module_name}.bias", module.bias)))
    return selected


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    return -(probability * probability.clamp_min(1e-8).log()).sum(1).mean()


def run_recurrent_episode(
    source_model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str,
    trajectory: str,
    shift_family: str,
    severity: int,
    seed: int,
    batch_size: int,
    adaptation_passes: int,
    learning_rate: float = 0.001,
) -> dict:
    if method not in {"source", "tent", "anchor", "ema_restore", "periodic_reset"}:
        raise ValueError(f"unsupported method: {method}")
    seed_all(seed)
    model = copy.deepcopy(source_model)
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    checkpoint_source_eval = _metrics(model, x, y, batch_size)
    b = shift(x, shift_family, severity, 10000 + seed)
    c_family = "channel_permutation"
    c = shift(x, c_family, severity, 20000 + seed)
    shifted_stages = [("B", b)] if trajectory == "A-B-A" else [("B", b), ("C", c)]
    params = _adaptable(model) if method != "source" else []
    source_params = {name: parameter.detach().clone() for name, parameter in params}
    configured_source_eval = _metrics(model, x, y, batch_size) if params else checkpoint_source_eval
    prospective_entropy: list[float] = []
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    target_correct = 0
    target_examples = 0
    target_entropy_sum = 0.0
    step = 0
    started = time.perf_counter()
    if method == "source":
        model.eval()
    else:
        model.train()
    for _ in range(adaptation_passes):
        for _, stage_x in shifted_stages:
            for start in range(0, len(stage_x), batch_size):
                xb = stage_x[start : start + batch_size]
                yb = y[start : start + batch_size]
                logits = model(xb)
                loss = _entropy(logits)
                prospective_entropy.append(float(loss.detach()))
                with torch.no_grad():
                    probability = torch.softmax(logits.detach(), dim=1)
                    target_correct += int(probability.argmax(1).eq(yb).sum())
                    target_examples += len(yb)
                    target_entropy_sum += float(
                        (-(probability * probability.clamp_min(1e-8).log()).sum(1)).sum()
                    )
                if method == "source":
                    step += 1
                    continue
                if method == "anchor":
                    loss = loss + 0.05 * sum((parameter - source_params[name]).pow(2).mean() for name, parameter in params)
                grads = torch.autograd.grad(loss, tuple(parameter for _, parameter in params))
                flat_gradient = torch.cat([gradient.detach().flatten() for gradient in grads])
                gradient_norms.append(float(flat_gradient.norm()))
                before = torch.cat([parameter.detach().flatten() for _, parameter in params])
                with torch.no_grad():
                    for (_, parameter), gradient in zip(params, grads):
                        parameter.add_(gradient, alpha=-learning_rate)
                    if method == "ema_restore":
                        for name, parameter in params:
                            parameter.mul_(0.99).add_(source_params[name], alpha=0.01)
                    elif method == "periodic_reset" and step % 8 == 7:
                        for name, parameter in params:
                            parameter.copy_(source_params[name])
                after = torch.cat([parameter.detach().flatten() for _, parameter in params])
                update_norms.append(float((after - before).norm()))
                step += 1
    prospective_step = step
    drift_before_return = 0.0
    if params:
        drift_before_return = float(
            torch.cat(
                [(parameter.detach() - source_params[name]).flatten() for name, parameter in params]
            ).norm()
        )
    return_pre = _metrics(model, x, y, batch_size)
    return_step = step + 1
    return_post = return_pre
    if method != "source":
        model.train()
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            loss = _entropy(model(xb))
            grads = torch.autograd.grad(loss, tuple(parameter for _, parameter in params))
            with torch.no_grad():
                for (_, parameter), gradient in zip(params, grads):
                    parameter.add_(gradient, alpha=-learning_rate)
        return_post = _metrics(model, x, y, batch_size)
    return {
        "source_evaluation": configured_source_eval,
        "checkpoint_source_evaluation": checkpoint_source_eval,
        "configured_source_evaluation": configured_source_eval,
        "configuration_gap": checkpoint_source_eval["accuracy"] - configured_source_eval["accuracy"],
        "shift_sequence": [shift_family] if trajectory == "A-B-A" else [shift_family, c_family],
        "prospective_mechanism": {
            "measurement_step": prospective_step,
            "prediction_entropy_before_update": float(np.mean(prospective_entropy)) if prospective_entropy else 0.0,
            "gradient_norm_before_update": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "update_norm": float(np.mean(update_norms)) if update_norms else 0.0,
            "parameter_drift": drift_before_return,
        },
        "target_stream_evaluation": {
            "accuracy": target_correct / max(target_examples, 1),
            "entropy": target_entropy_sum / max(target_examples, 1),
            "examples": target_examples,
            "timing": "pre_update_online_predictions",
        },
        "return_outcome": {
            "measurement_step": return_step,
            "accuracy_before_return_adaptation": return_pre["accuracy"],
            "accuracy_after_return_adaptation": return_post["accuracy"],
            "absolute_hysteresis_before_return_adaptation": checkpoint_source_eval["accuracy"] - return_pre["accuracy"],
            "update_induced_hysteresis_before_return_adaptation": configured_source_eval["accuracy"] - return_pre["accuracy"],
        },
        "labels_used_during_adaptation": False,
        "adaptation_steps_before_return": step,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def load_cifar(dataset: str, root: Path, *, train: bool, download: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision.transforms import ToTensor

    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[dataset]
    value = cls(root=str(root), train=train, transform=ToTensor(), download=download)
    loader = torch.utils.data.DataLoader(value, batch_size=512, shuffle=False, num_workers=2)
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for image, label in loader:
        images.append(image)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def train_source(dataset: str, seed: int, output: Path, *, epochs: int, batch_size: int, max_samples: int, device: str) -> dict:
    seed_all(seed)
    x, y = load_cifar(dataset, ROOT / "data", train=True, download=True)
    if max_samples:
        x, y = x[:max_samples], y[:max_samples]
    classes = 10 if dataset == "cifar10" else 100
    model = cifar_resnet18(classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            seen += len(xb)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1)})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "seed": seed, "classes": classes, "model_state": model.state_dict(), "history": history}, output)
    return {"status": "PASS", "checkpoint": str(output), "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(), "history": history}


def run_job(args: argparse.Namespace) -> dict:
    seed_all(args.seed)
    checkpoint = torch.load(args.source_checkpoint, map_location=args.device, weights_only=False)
    model = cifar_resnet18(int(checkpoint["classes"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    x, y = load_cifar(args.dataset, ROOT / "data", train=False, download=True)
    if args.max_test_samples:
        x, y = x[: args.max_test_samples], y[: args.max_test_samples]
    episode = run_recurrent_episode(
        model,
        x,
        y,
        method=args.method,
        trajectory=args.trajectory,
        shift_family=args.shift_family,
        severity=args.severity,
        seed=args.seed,
        batch_size=args.batch_size,
        adaptation_passes=args.adaptation_passes,
        learning_rate=args.learning_rate,
    )
    return {
        "schema_version": "2.0.0",
        "status": "PASS",
        "evidence_label": "FORMAL_EXPANSION" if not args.smoke else "SMOKE",
        "job_id": args.job_id,
        "dataset": args.dataset,
        "model": "cifar_resnet18",
        "method": args.method,
        "trajectory": args.trajectory,
        "shift_family": args.shift_family,
        "severity": args.severity,
        "seed": args.seed,
        "source_checkpoint_sha256": "sha256:" + hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest(),
        "runner_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_provenance_sha256": getattr(args, "dataset_provenance_sha256", None),
        "protocol_parameters": {
            "batch_size": args.batch_size,
            "adaptation_passes": args.adaptation_passes,
            "adaptation_learning_rate": args.learning_rate,
        },
        "episode": episode,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "platform": platform.platform()},
        "created_at": utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-source")
    train.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--max-samples", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run = sub.add_parser("run")
    run.add_argument("--job-id", required=True)
    run.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--trajectory", choices=["A-B-A", "A-B-C-A"], required=True)
    run.add_argument("--shift-family", choices=["noise", "brightness", "contrast", "blur", "rotation"], required=True)
    run.add_argument("--severity", type=int, required=True)
    run.add_argument("--source-checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--adaptation-passes", type=int, default=1)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--max-test-samples", type=int, default=0)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--dataset-provenance-sha256")
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "train-source":
        result = train_source(args.dataset, args.seed, args.output, epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples, device=args.device)
    else:
        result = run_job(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite formal result: {args.output}")
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output"] = str(args.output)
        result["output_sha256"] = "sha256:" + hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
