from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from expansion_experiment import TinyConvNet, cifar_resnet18, load_cifar, shift

ROOT = Path(__file__).resolve().parent
METHODS = ("source", "tent", "eata", "sar", "cotta", "rotta")
TRAJECTORIES = ("A-B-A", "A-B-C-A", "A-B-A-B-A", "A-C-B-A")
NORMALIZATION_POLICIES = (
    "source_eval", "bn_affine_frozen_stats", "bn_affine_running_stats",
    "train_eval_policy_variant", "groupnorm_checkpoint", "layernorm_checkpoint",
)


class ChannelLayerNorm2d(torch.nn.Module):
    """LayerNorm over channels for NCHW feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = True
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_last = x.permute(0, 2, 3, 1)
        normalized = torch.nn.functional.layer_norm(
            channels_last, (self.num_channels,), self.weight, self.bias, self.eps
        )
        return normalized.permute(0, 3, 1, 2)


def _replace_child(module: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    """Replace a named child in Sequential/module containers without string eval."""
    if isinstance(module, torch.nn.Sequential) and name.isdigit():
        module[int(name)] = replacement
    elif isinstance(module, (torch.nn.ModuleList, torch.nn.ParameterDict)) and name.isdigit():
        module[int(name)] = replacement
    else:
        setattr(module, name, replacement)


def _normalization_module_type(module: torch.nn.Module) -> str | None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return "BatchNorm"
    if isinstance(module, torch.nn.GroupNorm):
        return "Groupnorm"
    if isinstance(module, (torch.nn.LayerNorm, ChannelLayerNorm2d)):
        return "Layernorm"
    return None


def _apply_normalization_policy(model: torch.nn.Module, policy: str) -> torch.nn.Module:
    """Materialize the registered normalization policy on a private model copy."""
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"unsupported normalization policy: {policy}")
    if policy not in {"groupnorm_checkpoint", "layernorm_checkpoint"}:
        return model

    target = "group" if policy == "groupnorm_checkpoint" else "layer"
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                continue
            if target == "group":
                groups = min(8, child.num_features)
                while groups > 1 and child.num_features % groups:
                    groups -= 1
                replacement = torch.nn.GroupNorm(
                    num_groups=max(1, groups),
                    num_channels=child.num_features,
                    eps=child.eps,
                    affine=child.affine,
                )
            else:
                replacement = ChannelLayerNorm2d(child.num_features, eps=child.eps)
            if child.affine and child.weight is not None and replacement.weight is not None:
                replacement.weight.data.copy_(child.weight.detach())
                replacement.bias.data.copy_(child.bias.detach())
            replacements.append((parent, child_name, replacement))
    for parent, child_name, replacement in replacements:
        _replace_child(parent, child_name, replacement)
    return model


def normalization_metadata(model: torch.nn.Module, policy: str, *, checkpoint_sha256: str | None = None) -> dict[str, Any]:
    module_types = sorted({value for _, module in model.named_modules() if (value := _normalization_module_type(module)) is not None})
    modules = [name for name, module in model.named_modules() if _normalization_module_type(module) is not None]
    return {
        "policy": policy,
        "module_types": module_types,
        "module_names": modules,
        "checkpoint_sha256": checkpoint_sha256,
        "independent_checkpoint": policy in {"groupnorm_checkpoint", "layernorm_checkpoint"},
    }


def materialize_normalization_checkpoint(source_checkpoint: Path, output: Path, policy: str) -> Path:
    """Create an immutable, policy-specific checkpoint with an auditable source binding."""
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"unsupported normalization policy: {policy}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return output
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = cifar_resnet18(int(source["classes"]))
    model.load_state_dict(source["model_state"], strict=True)
    _apply_normalization_policy(model, policy)
    payload = {
        "schema_version": "1.0.0",
        "checkpoint_kind": "NORMALIZATION_POLICY_DERIVED",
        "normalization_policy": policy,
        "source_checkpoint_sha256": "sha256:" + hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        "classes": int(source["classes"]),
        "dataset": source.get("dataset"),
        "seed": source.get("seed"),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "normalization_metadata": normalization_metadata(model, policy),
    }
    torch.save(payload, output)
    return output


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


def state_dict_digest(model: torch.nn.Module) -> str:
    return _snapshot_digest({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def _snapshot_digest(snapshot: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(snapshot.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _tensor_map_digest(values: dict[str, torch.Tensor]) -> str:
    """Hash auxiliary method state without putting it into the student model."""
    return _snapshot_digest({name: value.detach().cpu().clone() for name, value in values.items()})


def _optimizer_digest(optimizer: torch.optim.Optimizer | None) -> str | None:
    if optimizer is None:
        return None
    tensors: dict[str, torch.Tensor] = {}
    for index, (parameter, state) in enumerate(optimizer.state.items()):
        for name, value in state.items():
            if torch.is_tensor(value):
                tensors[f"{index}:{name}"] = value
    return _tensor_map_digest(tensors) if tensors else "sha256:" + hashlib.sha256(b"empty-optimizer-state").hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _softmax_entropy_per_sample(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    return -(probability * probability.clamp_min(1e-8).log()).sum(1)


def _soft_target_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_probability = torch.softmax(teacher_logits.detach(), dim=1)
    return -(teacher_probability * torch.log_softmax(student_logits, dim=1)).sum(1).mean()


def _augmentation_batch(x: torch.Tensor, index: int) -> torch.Tensor:
    """Deterministic tensor augmentation used by the independent CoTTA reimplementation."""
    generator = torch.Generator(device=x.device).manual_seed(700000 + index)
    out = x.flip(-1) if index % 2 else x
    noise_scale = 0.005 + 0.002 * (index % 4)
    return (out + noise_scale * torch.randn(out.shape, generator=generator, device=x.device)).clamp(0.0, 1.0)


def _fisher_information(
    model: torch.nn.Module,
    params: list[tuple[str, torch.nn.Parameter]],
    x: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Estimate diagonal Fisher on source predictions without using target labels."""
    fisher = {name: torch.zeros_like(parameter) for name, parameter in params}
    samples = 0
    was_training = model.training
    model.eval()
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        logits = model(xb)
        pseudo = logits.detach().argmax(1)
        losses = torch.nn.functional.cross_entropy(logits, pseudo, reduction="sum")
        grads = torch.autograd.grad(losses, tuple(parameter for _, parameter in params), allow_unused=False)
        for (name, _), grad in zip(params, grads):
            fisher[name].add_(grad.detach().pow(2))
        samples += len(xb)
    if was_training:
        model.train()
    divisor = max(samples, 1)
    fisher = {name: value / divisor for name, value in fisher.items()}
    return fisher, samples


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    return [k for k in sorted(before) if not torch.equal(before[k], after[k])]


def _norm_layers(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm, torch.nn.LayerNorm, ChannelLayerNorm2d)):
            yield name, module


def _adaptable(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, module in _norm_layers(model):
        if getattr(module, "affine", False):
            if module.weight is not None:
                module.weight.requires_grad_(True)
                selected.append((f"{name}.weight", module.weight))
            if module.bias is not None:
                module.bias.requires_grad_(True)
                selected.append((f"{name}.bias", module.bias))
    return selected


def _configure(model: torch.nn.Module, policy: str, adaptive: bool) -> None:
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"unsupported normalization policy: {policy}")
    if not adaptive or policy == "source_eval":
        model.eval()
        return
    model.train()
    if policy != "bn_affine_running_stats":
        for _, module in _norm_layers(model):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def _metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = correct = 0
    entropy_sum = 0.0
    ece_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
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
    if was_training:
        model.train()
    return {"accuracy": correct / max(total, 1), "entropy": entropy_sum / max(total, 1), "ece": ece_sum / max(total, 1)}


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    return -(prob * prob.clamp_min(1e-8).log()).sum(1).mean()


def _stages(x: torch.Tensor, family: str, severity: int, seed: int, trajectory: str) -> list[tuple[str, torch.Tensor]]:
    b = shift(x, family, severity, 10000 + seed)
    c = shift(x, "channel_permutation", severity, 20000 + seed)
    values = {"A-B-A": [("B", b)], "A-B-C-A": [("B", b), ("C", c)], "A-B-A-B-A": [("B", b), ("A", x), ("B", b)], "A-C-B-A": [("C", c), ("B", b)]}
    return values[trajectory]


def _declared_mutations(model: torch.nn.Module, method: str, policy: str) -> list[str]:
    if method == "source" or policy == "source_eval":
        return []
    names: list[str] = []
    for name, module in _norm_layers(model):
        if getattr(module, "affine", False):
            if module.weight is not None:
                names.append(f"{name}.weight")
            if module.bias is not None:
                names.append(f"{name}.bias")
        if policy == "bn_affine_running_stats" and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            names.extend([f"{name}.running_mean", f"{name}.running_var", f"{name}.num_batches_tracked"])
    return sorted(names)


def run_recurrent_episode(source_model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, method: str, trajectory: str, shift_family: str, severity: int, seed: int, batch_size: int, adaptation_passes: int, learning_rate: float = 0.001, normalization_policy: str = "bn_affine_frozen_stats", pass_count: int | None = None) -> dict:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if trajectory not in TRAJECTORIES:
        raise ValueError(f"unsupported trajectory: {trajectory}")
    seed_all(seed)
    model = copy.deepcopy(source_model)
    # Apply structural policies to the private episode model before taking
    # snapshots or discovering adaptable parameters.  The caller's source
    # model remains untouched, while GroupNorm/LayerNorm episodes use the
    # exact architecture declared by the policy.
    model = _apply_normalization_policy(model, normalization_policy)
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    before = _snapshot(model)
    checkpoint = _metrics(model, x, y, batch_size)
    params = _adaptable(model) if method != "source" and normalization_policy != "source_eval" else []
    source_params = {n: p.detach().clone() for n, p in params}
    _configure(model, normalization_policy, adaptive=method != "source")
    configured = _metrics(model, x, y, batch_size)
    stages = _stages(x, shift_family, severity, seed, trajectory)
    optimizer: torch.optim.Optimizer | None = None
    if params:
        if method in {"cotta", "rotta", "eata"}:
            optimizer = torch.optim.Adam([parameter for _, parameter in params], lr=learning_rate)
        else:
            optimizer = torch.optim.SGD([parameter for _, parameter in params], lr=learning_rate)
    optimizer_before_digest = _optimizer_digest(optimizer)
    teacher = copy.deepcopy(model).eval() if method in {"cotta", "rotta"} else None
    teacher_before_digest = state_dict_digest(teacher) if teacher is not None else None
    anchor = copy.deepcopy(model).eval() if method == "cotta" else None
    anchor_state = _snapshot(anchor) if anchor is not None else {}
    fisher: dict[str, torch.Tensor] = {}
    fisher_samples = 0
    current_model_probs: torch.Tensor | None = None
    reliable_samples = 0
    nonredundant_samples = 0
    augmentation_predictions = 0
    stochastic_restore_trials = 0
    sam_first_steps = 0
    sam_second_steps = 0
    sam_perturbation_rollbacks = 0
    entropy_ema: float | None = None
    recovery_resets = 0
    memory_items: list[dict[str, Any]] = []
    memory_capacity = 32
    memory_classes = 0
    memory_max_age = 0
    robust_bn_state: dict[str, torch.Tensor] = {}
    robust_bn_layers = sum(
        1 for _, module in _norm_layers(model)
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ) if method == "rotta" else 0
    if method == "eata" and params:
        fisher, fisher_samples = _fisher_information(model, params, x, batch_size)
    passes = int(pass_count if pass_count is not None else adaptation_passes)
    step = 0
    optimizer_step_count = 0
    target_correct = target_examples = 0
    target_entropy = 0.0
    gradients: list[float] = []
    updates: list[float] = []
    predictor_values: list[float] = []
    for _ in range(passes):
        for _, stage_x in stages:
            for start in range(0, len(stage_x), batch_size):
                xb, yb = stage_x[start:start + batch_size], y[start:start + batch_size]
                _configure(model, normalization_policy, adaptive=method != "source")
                logits = model(xb)
                with torch.no_grad():
                    prob = torch.softmax(logits, dim=1)
                    target_correct += int(prob.argmax(1).eq(yb).sum())
                    target_examples += len(yb)
                    target_entropy += float((-(prob * prob.clamp_min(1e-8).log()).sum(1)).sum())
                predictor_values.append(float(_entropy(logits).detach()))
                if method == "source" or not params:
                    step += 1
                    continue
                old = torch.cat([p.detach().flatten() for _, p in params])
                if method == "eata":
                    entropy = _softmax_entropy_per_sample(logits)
                    margin = max(0.35, 0.85 * math.log(max(logits.shape[1], 2)))
                    reliable = entropy < margin
                    reliable_samples += int(reliable.sum())
                    probabilities = torch.softmax(logits.detach(), dim=1)
                    if current_model_probs is None:
                        nonredundant = reliable
                    else:
                        similarity = torch.nn.functional.cosine_similarity(
                            current_model_probs.unsqueeze(0), probabilities, dim=1
                        )
                        nonredundant = reliable & (similarity.abs() < 0.999)
                    nonredundant_samples += int(nonredundant.sum())
                    selected = entropy[nonredundant]
                    if bool(nonredundant.any()):
                        selected_probs = probabilities[nonredundant].mean(0)
                        current_model_probs = selected_probs if current_model_probs is None else 0.9 * current_model_probs + 0.1 * selected_probs
                    if current_model_probs is None:
                        current_model_probs = probabilities.mean(0)
                    loss = selected.mean() if selected.numel() else entropy.mean() * 0.0
                    ewc = sum(
                        (fisher[name].to(parameter.device) * (parameter - source_params[name].to(parameter.device)).pow(2)).mean()
                        for name, parameter in params
                    )
                    loss = loss + 50.0 * ewc
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    flat = torch.cat([p.grad.detach().flatten() for _, p in params if p.grad is not None])
                    gradients.append(float(flat.norm()) if flat.numel() else 0.0)
                    optimizer.step()
                    optimizer_step_count += 1
                elif method == "sar":
                    entropy = _softmax_entropy_per_sample(logits)
                    margin = max(0.35, 0.85 * math.log(max(logits.shape[1], 2)))
                    first_mask = entropy < margin
                    if not bool(first_mask.any()):
                        first_mask[0] = True
                    first_loss = entropy[first_mask].mean()
                    optimizer.zero_grad(set_to_none=True)
                    first_grads = torch.autograd.grad(first_loss, tuple(parameter for _, parameter in params), allow_unused=False)
                    flat = torch.cat([grad.detach().flatten() for grad in first_grads])
                    gradients.append(float(flat.norm()))
                    sam_first_steps += 1
                    radius = 0.05
                    scale = radius / (flat.norm() + 1e-12)
                    perturbations: list[torch.Tensor] = []
                    with torch.no_grad():
                        for (_, parameter), grad in zip(params, first_grads):
                            perturbation = grad.detach() * scale
                            parameter.add_(perturbation)
                            perturbations.append(perturbation)
                    second_logits = model(xb)
                    second_entropy = _softmax_entropy_per_sample(second_logits)
                    second_mask = second_entropy[first_mask] < margin
                    if not bool(second_mask.any()):
                        second_mask = torch.ones_like(second_entropy[first_mask], dtype=torch.bool)
                    second_loss = second_entropy[first_mask][second_mask].mean()
                    entropy_ema = float(second_loss.detach()) if entropy_ema is None else 0.9 * entropy_ema + 0.1 * float(second_loss.detach())
                    second_grads = torch.autograd.grad(second_loss, tuple(parameter for _, parameter in params), allow_unused=False)
                    with torch.no_grad():
                        for (_, parameter), perturbation in zip(params, perturbations):
                            parameter.sub_(perturbation)
                    sam_perturbation_rollbacks += 1
                    for (_, parameter), grad in zip(params, second_grads):
                        parameter.grad = grad.detach().clone()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    sam_second_steps += 1
                    optimizer_step_count += 1
                    if entropy_ema < 0.2:
                        model.load_state_dict(before, strict=True)
                        recovery_resets += 1
                elif method == "cotta" and teacher is not None and anchor is not None:
                    with torch.no_grad():
                        anchor_probability = torch.softmax(anchor(xb), dim=1).max(1).values.mean()
                        teacher_logits = teacher(xb)
                        augmented = [teacher(_augmentation_batch(xb, index)).detach() for index in range(32)]
                        augmentation_predictions += len(augmented)
                        if float(anchor_probability) < 0.9:
                            teacher_target = torch.stack(augmented).mean(0)
                        else:
                            teacher_target = teacher_logits.detach()
                    loss = _soft_target_loss(logits, teacher_target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    flat = torch.cat([p.grad.detach().flatten() for _, p in params if p.grad is not None])
                    gradients.append(float(flat.norm()) if flat.numel() else 0.0)
                    optimizer.step()
                    optimizer_step_count += 1
                    with torch.no_grad():
                        for teacher_parameter, model_parameter in zip(teacher.parameters(), model.parameters()):
                            teacher_parameter.mul_(0.99).add_(model_parameter, alpha=0.01)
                        for name, parameter in params:
                            restore_mask = (torch.rand(parameter.shape, device=parameter.device) < 0.1)
                            parameter.copy_(torch.where(restore_mask, anchor_state[name].to(parameter.device), parameter))
                            stochastic_restore_trials += 1
                elif method == "rotta" and teacher is not None:
                    with torch.no_grad():
                        teacher.eval()
                        teacher_logits = teacher(xb)
                        probability = torch.softmax(teacher_logits, dim=1)
                        pseudo = probability.argmax(1)
                        uncertainty = _softmax_entropy_per_sample(teacher_logits)
                    memory_classes = int(probability.shape[1])
                    for item_x, item_label, item_uncertainty in zip(xb.detach(), pseudo.tolist(), uncertainty.tolist()):
                        for item in memory_items:
                            item["age"] += 1
                        candidate = {"x": item_x.detach().cpu(), "label": int(item_label), "uncertainty": float(item_uncertainty), "age": 0}
                        same_class = [item for item in memory_items if item["label"] == candidate["label"]]
                        per_class_capacity = max(1, math.ceil(memory_capacity / max(memory_classes, 1)))
                        if len(same_class) >= per_class_capacity:
                            victim = max(same_class, key=lambda item: item["age"] + item["uncertainty"])
                            memory_items.pop(next(index for index, item in enumerate(memory_items) if item is victim))
                        elif len(memory_items) >= memory_capacity:
                            victim = max(memory_items, key=lambda item: item["age"] + item["uncertainty"])
                            memory_items.pop(next(index for index, item in enumerate(memory_items) if item is victim))
                        memory_items.append(candidate)
                    memory_max_age = max((item["age"] for item in memory_items), default=0)
                    sup = memory_items[: min(len(memory_items), 16)]
                    support_x = torch.stack([item["x"] for item in sup]).to(device)
                    support_target = teacher(support_x).detach()
                    support_student = model(support_x)
                    ages = torch.tensor([item["age"] for item in sup], dtype=torch.float32, device=device)
                    weights = torch.exp(-ages) / (1.0 + torch.exp(-ages))
                    losses = -(torch.softmax(support_target, dim=1) * torch.log_softmax(support_student, dim=1)).sum(1)
                    loss = (losses * weights).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    flat = torch.cat([p.grad.detach().flatten() for _, p in params if p.grad is not None])
                    gradients.append(float(flat.norm()) if flat.numel() else 0.0)
                    optimizer.step()
                    optimizer_step_count += 1
                    with torch.no_grad():
                        for name, module in _norm_layers(model):
                            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                                if module.running_mean is not None and module.running_var is not None:
                                    batch_mean = torch.full_like(module.running_mean, float(xb.mean()))
                                    batch_var = torch.full_like(module.running_var, float(xb.var(unbiased=False)))
                                    previous = robust_bn_state.get(name + ".mean", module.running_mean.detach().clone())
                                    previous_var = robust_bn_state.get(name + ".var", module.running_var.detach().clone())
                                    robust_bn_state[name + ".mean"] = 0.9 * previous + 0.1 * batch_mean
                                    robust_bn_state[name + ".var"] = 0.9 * previous_var + 0.1 * batch_var
                        for teacher_parameter, model_parameter in zip(teacher.parameters(), model.parameters()):
                            teacher_parameter.mul_(0.99).add_(model_parameter, alpha=0.01)
                else:
                    loss = _entropy(logits)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    flat = torch.cat([p.grad.detach().flatten() for _, p in params if p.grad is not None])
                    gradients.append(float(flat.norm()) if flat.numel() else 0.0)
                    optimizer.step()
                    optimizer_step_count += 1
                new = torch.cat([p.detach().flatten() for _, p in params])
                updates.append(float((new - old).norm()))
                step += 1
    _configure(model, normalization_policy, adaptive=False)
    return_pre = _metrics(model, x, y, batch_size)
    observed_state = _snapshot(model)
    observed = _changed(before, observed_state)
    declared = _declared_mutations(model, method, normalization_policy)
    unexpected = [name for name in observed if name not in declared]
    drift = float(torch.cat([(p.detach() - source_params[n].to(p.device)).flatten() for n, p in params]).norm()) if params else 0.0
    outcome_step = step + 1
    auxiliary_state: dict[str, torch.Tensor] = {}
    if current_model_probs is not None:
        auxiliary_state["probability_ema"] = current_model_probs
    if robust_bn_state:
        auxiliary_state.update(robust_bn_state)
    memory_histogram = [0] * max(memory_classes, 1)
    for item in memory_items:
        if 0 <= item["label"] < len(memory_histogram):
            memory_histogram[item["label"]] += 1
    teacher_after_digest = state_dict_digest(teacher) if teacher is not None else None
    anchor_digest = _snapshot_digest(anchor_state) if anchor_state else None
    memory_state = {
        f"item_{index}_x": item["x"] for index, item in enumerate(memory_items)
    }
    memory_state.update({
        f"item_{index}_meta": torch.tensor(
            [item["label"], item["uncertainty"], item["age"]], dtype=torch.float64
        ) for index, item in enumerate(memory_items)
    })
    memory_digest = _tensor_map_digest(memory_state) if memory_state else _json_digest([])
    recovery_state = {"entropy_ema": entropy_ema, "recovery_resets": recovery_resets}
    algorithm_semantics: dict[str, Any] = {
        "implementation_status": "INDEPENDENT_REIMPLEMENTATION",
        "optimizer_type": type(optimizer).__name__ if optimizer is not None else "NONE",
        "optimizer_before_digest": optimizer_before_digest,
        "optimizer_after_digest": _optimizer_digest(optimizer),
        "optimizer_step_count": optimizer_step_count,
        "allowed_state": sorted(set(_declared_mutations(model, method, normalization_policy))),
    }
    if method == "eata":
        algorithm_semantics.update({
            "fisher_samples": fisher_samples,
            "fisher_digest": _tensor_map_digest(fisher),
            "reliable_samples": reliable_samples,
            "nonredundant_samples": nonredundant_samples,
            "eata_probability_ema_digest": _tensor_map_digest(auxiliary_state) if current_model_probs is not None else "sha256:" + hashlib.sha256(b"empty-probability-ema").hexdigest(),
            "fisher_regularization": "diagonal_fisher_ewc",
        })
    elif method == "sar":
        algorithm_semantics.update({
            "sam_first_steps": sam_first_steps,
            "sam_second_steps": sam_second_steps,
            "sam_perturbation_rollbacks": sam_perturbation_rollbacks,
            "entropy_ema": entropy_ema,
            "recovery_resets": recovery_resets,
        })
    elif method == "cotta":
        algorithm_semantics.update({
            "teacher_before_digest": teacher_before_digest,
            "teacher_after_digest": teacher_after_digest,
            "anchor_digest": anchor_digest,
            "teacher_updates": optimizer_step_count,
            "augmentation_predictions": augmentation_predictions,
            "stochastic_restore_trials": stochastic_restore_trials,
            "stochastic_restore_probability": 0.1,
        })
    elif method == "rotta":
        algorithm_semantics.update({
            "memory_type": "class_balanced_timeliness_uncertainty",
            "memory_capacity": memory_capacity,
            "memory_occupancy": len(memory_items),
            "memory_class_histogram": memory_histogram,
            "memory_max_age": memory_max_age,
            "memory_age_digest": _tensor_map_digest({f"age_{index}": torch.tensor(item["age"]) for index, item in enumerate(memory_items)}) if memory_items else "sha256:" + hashlib.sha256(b"empty-memory").hexdigest(),
            "robust_bn_layers": robust_bn_layers,
            "robust_bn_state_digest": _tensor_map_digest(robust_bn_state) if robust_bn_state else "sha256:" + hashlib.sha256(b"empty-robust-bn").hexdigest(),
            "teacher_before_digest": teacher_before_digest,
            "teacher_after_digest": teacher_after_digest,
            "teacher_updates": optimizer_step_count,
        })
    components = {
        "student_model": {
            "before_digest": _snapshot_digest(before),
            "after_digest": _snapshot_digest(observed_state),
            "declared_mutations": declared,
            "observed_mutations": observed,
            "unexpected_mutations": unexpected,
        },
        "optimizer": {
            "before_digest": optimizer_before_digest,
            "after_digest": _optimizer_digest(optimizer),
            "step_count": optimizer_step_count,
            "mutated": optimizer_step_count > 0,
        },
        "teacher": {
            "before_digest": teacher_before_digest,
            "after_digest": teacher_after_digest,
            "mutated": teacher_before_digest is not None and teacher_before_digest != teacher_after_digest,
        },
        "anchor": {
            "before_digest": anchor_digest,
            "after_digest": anchor_digest,
            "mutated": False,
        },
        "memory": {
            "digest": memory_digest,
            "occupancy": len(memory_items),
            "capacity": memory_capacity if method == "rotta" else 0,
            "max_age": memory_max_age,
            "mutated": bool(memory_items),
        },
        "robust_bn": {
            "digest": _tensor_map_digest(robust_bn_state) if robust_bn_state else _json_digest([]),
            "layers": robust_bn_layers,
            "mutated": bool(robust_bn_state),
        },
        "recovery": {
            "digest": _json_digest(recovery_state),
            "state": recovery_state,
            "mutated": entropy_ema is not None or recovery_resets > 0,
        },
    }
    return {
        "method": method,
        "normalization_policy": normalization_policy,
        "normalization_modules": normalization_metadata(model, normalization_policy),
        "source_evaluation": configured,
        "checkpoint_source_evaluation": checkpoint,
        "configured_source_evaluation": configured,
        "configuration_gap": checkpoint["accuracy"] - configured["accuracy"],
        "shift_sequence": [name for name, _ in stages],
        "events": {"predictor_step": step, "outcome_step": outcome_step, "predictor_event_id": f"step:{step}", "outcome_event_id": f"step:{outcome_step}"},
        "prospective_mechanism": {"measurement_step": step, "prediction_entropy_before_update": float(np.mean(predictor_values)) if predictor_values else 0.0, "gradient_norm_before_update": float(np.mean(gradients)) if gradients else 0.0, "update_norm": float(np.mean(updates)) if updates else 0.0, "parameter_drift": drift},
        "target_stream_evaluation": {"accuracy": target_correct / max(target_examples, 1), "entropy": target_entropy / max(target_examples, 1), "examples": target_examples, "timing": "pre_update_online_predictions"},
        "return_outcome": {"measurement_step": outcome_step, "accuracy_before_return_adaptation": return_pre["accuracy"], "accuracy_after_return_adaptation": return_pre["accuracy"], "absolute_hysteresis_before_return_adaptation": checkpoint["accuracy"] - return_pre["accuracy"], "update_induced_hysteresis_before_return_adaptation": configured["accuracy"] - return_pre["accuracy"]},
        "state_transition": {"before_digest": _snapshot_digest(before), "after_digest": _snapshot_digest(observed_state), "declared_mutations": declared, "observed_mutations": observed, "unexpected_mutations": unexpected, "optimizer_step_count": optimizer_step_count, "method_state": algorithm_semantics, "components": components},
        "algorithm_semantics": algorithm_semantics,
        "labels_used_during_adaptation": False,
        "adaptation_steps_before_return": step,
        "wall_clock_seconds": 0.0,
    }


def run_job(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.source_checkpoint, map_location=args.device, weights_only=False)
    model = cifar_resnet18(int(checkpoint["classes"])).to(args.device)
    checkpoint_policy = checkpoint.get("normalization_policy") if isinstance(checkpoint, dict) else None
    if checkpoint.get("checkpoint_kind") == "NORMALIZATION_POLICY_DERIVED":
        if checkpoint_policy != args.normalization_policy:
            raise ValueError(
                f"policy checkpoint mismatch: checkpoint={checkpoint_policy!r}, requested={args.normalization_policy!r}"
            )
        model = _apply_normalization_policy(model, args.normalization_policy)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    x, y = load_cifar(args.dataset, ROOT / "data", train=False, download=True)
    if args.max_test_samples:
        x, y = x[:args.max_test_samples], y[:args.max_test_samples]
    episode = run_recurrent_episode(model, x, y, method=args.method, trajectory=args.trajectory, shift_family=args.shift_family, severity=args.severity, seed=args.seed, batch_size=args.batch_size, adaptation_passes=args.adaptation_passes, learning_rate=args.learning_rate, normalization_policy=args.normalization_policy, pass_count=args.pass_count)
    upstream_registry_path = ROOT / "configs" / "modern_baseline_upstreams.json"
    upstream_registry = json.loads(upstream_registry_path.read_text(encoding="utf-8"))
    method_contracts_path = ROOT / "configs" / "tta_method_state_contracts.json"
    provider = upstream_registry.get("providers", {}).get(args.method)
    method_upstream = {
        "binding": "PINNED_UPSTREAM_BEHAVIOR_REFERENCE" if provider else "LOCAL_CONTROL_IMPLEMENTATION",
        "registry_sha256": "sha256:" + hashlib.sha256(upstream_registry_path.read_bytes()).hexdigest(),
        "provider": provider,
    }
    source_checkpoint_sha256 = "sha256:" + hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()
    base_source_checkpoint_sha256 = checkpoint.get("source_checkpoint_sha256", source_checkpoint_sha256)
    return {"schema_version": "1.0.0", "status": "PASS", "evidence_label": "SEMANTIC_CANARY" if args.canary else ("SMOKE" if args.smoke else "FORMAL_PUBLICATION_EXPANSION"), "job_id": args.job_id, "campaign_id": args.campaign_id, "dataset": args.dataset, "method": args.method, "trajectory": args.trajectory, "shift_family": args.shift_family, "severity": args.severity, "seed": args.seed, "normalization_policy": args.normalization_policy, "source_checkpoint_sha256": source_checkpoint_sha256, "base_source_checkpoint_sha256": base_source_checkpoint_sha256, "policy_checkpoint_sha256": source_checkpoint_sha256 if checkpoint.get("checkpoint_kind") == "NORMALIZATION_POLICY_DERIVED" else None, "normalization_metadata": episode.get("normalization_modules"), "runner_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "supervisor_sha256": args.supervisor_sha256, "manifest_sha256": args.manifest_sha256, "dataset_provenance_sha256": args.dataset_provenance_sha256, "method_registry_sha256": "sha256:" + hashlib.sha256(upstream_registry_path.read_bytes()).hexdigest(), "method_contracts_sha256": "sha256:" + hashlib.sha256(method_contracts_path.read_bytes()).hexdigest(), "method_upstream": method_upstream, "protocol_parameters": {"batch_size": args.batch_size, "adaptation_passes": args.adaptation_passes, "adaptation_learning_rate": args.learning_rate, "normalization_policy": args.normalization_policy, "pass_count": args.pass_count}, "episode": episode, "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "platform": platform.platform()}, "created_at": utc()}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--job-id", required=True)
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", choices=list(METHODS), required=True)
    run.add_argument("--trajectory", choices=list(TRAJECTORIES), required=True)
    run.add_argument("--shift-family", choices=["noise", "brightness", "contrast", "blur", "rotation"], required=True)
    run.add_argument("--severity", type=int, required=True)
    run.add_argument("--source-checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--adaptation-passes", type=int, default=1)
    run.add_argument("--pass-count", type=int)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--normalization-policy", choices=list(NORMALIZATION_POLICIES), default="bn_affine_frozen_stats")
    run.add_argument("--max-test-samples", type=int, default=0)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--dataset-provenance-sha256")
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--supervisor-sha256", required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    result = run_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite publication-expansion result: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(args.output)
    result["output_sha256"] = "sha256:" + hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
