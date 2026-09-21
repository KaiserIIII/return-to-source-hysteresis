from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import publication_expansion_experiment as method_engine
from prepare_cifar_c_data import severity_slices


ROOT = Path(__file__).resolve().parent
DATASET_MAP = {"cifar10_c": "cifar10", "cifar100_c": "cifar100"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def verify_expected_digest(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash changed: expected={expected}, observed={observed}")
    return observed


def runtime_environment_identity(device: str) -> dict[str, Any]:
    cuda_requested = str(device).startswith("cuda")
    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_requested else None,
    }


def extract_severity_block(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    severity: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    image_slice, label_slice = severity_slices(
        severity=severity, labels_count=int(labels.shape[0]), block_size=block_size
    )
    selected_images = images[image_slice]
    selected_labels = labels[label_slice]
    if len(selected_images) != block_size or len(selected_labels) != block_size:
        raise ValueError(
            f"severity block misalignment: images={len(selected_images)}, labels={len(selected_labels)}"
        )
    return selected_images, selected_labels, [image_slice.start, image_slice.stop]


def _to_image_tensor(array: np.ndarray, max_samples: int = 0) -> torch.Tensor:
    selected = array[:max_samples] if max_samples else array
    contiguous = np.array(selected, dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(contiguous).permute(0, 3, 1, 2).float().div_(255.0)


def _to_label_tensor(array: np.ndarray, max_samples: int = 0) -> torch.Tensor:
    selected = array[:max_samples] if max_samples else array
    return torch.from_numpy(np.array(selected, dtype=np.int64, copy=True))


@contextmanager
def _external_stages(stage_b: torch.Tensor, stage_c: torch.Tensor) -> Iterator[None]:
    original = method_engine._stages

    def stages(
        x: torch.Tensor,
        _family: str,
        _severity: int,
        _seed: int,
        trajectory: str,
    ) -> list[tuple[str, torch.Tensor]]:
        b = stage_b.to(x.device)
        c = stage_c.to(x.device)
        values = {
            "A-B-A": [("B", b)],
            "A-B-C-A": [("B", b), ("C", c)],
        }
        if trajectory not in values:
            raise ValueError(f"unsupported external trajectory: {trajectory}")
        return values[trajectory]

    method_engine._stages = stages
    try:
        yield
    finally:
        method_engine._stages = original


def run_external_episode(
    source_model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    stage_b: torch.Tensor,
    stage_c: torch.Tensor,
    method: str,
    trajectory: str,
    seed: int,
    batch_size: int,
    adaptation_passes: int,
    learning_rate: float,
    normalization_policy: str = "bn_affine_frozen_stats",
) -> dict[str, Any]:
    if clean.shape != stage_b.shape or clean.shape != stage_c.shape:
        raise ValueError(
            f"external trajectory shape mismatch: clean={tuple(clean.shape)}, "
            f"B={tuple(stage_b.shape)}, C={tuple(stage_c.shape)}"
        )
    if len(labels) != len(clean):
        raise ValueError(f"label count mismatch: labels={len(labels)}, images={len(clean)}")
    source_before = method_engine.state_dict_digest(source_model)
    with _external_stages(stage_b, stage_c):
        episode = method_engine.run_recurrent_episode(
            source_model,
            clean,
            labels,
            method=method,
            trajectory=trajectory,
            shift_family="noise",
            severity=1,
            seed=seed,
            batch_size=batch_size,
            adaptation_passes=adaptation_passes,
            learning_rate=learning_rate,
            normalization_policy=normalization_policy,
            pass_count=adaptation_passes,
        )
    source_after = method_engine.state_dict_digest(source_model)
    episode["external_source_model_immutability"] = {
        "before_digest": source_before,
        "after_digest": source_after,
        "status": "PASS" if source_before == source_after else "FAIL",
    }
    if source_before != source_after:
        raise RuntimeError("source model mutated during external trajectory")
    return episode


def _file_record(dataset_record: dict[str, Any], filename: str) -> dict[str, Any]:
    matches = [
        item for item in dataset_record.get("files", [])
        if str(item.get("path", "")).replace("\\", "/").endswith("/" + filename)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one provenance record for {filename}, found {len(matches)}")
    return matches[0]


def load_external_inputs(
    *,
    dataset: str,
    corruption: str,
    paired_corruption: str,
    severity: int,
    data_root: Path,
    dataset_provenance: dict[str, Any],
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if dataset not in DATASET_MAP:
        raise ValueError(f"unsupported external dataset: {dataset}")
    record = dataset_provenance["datasets"][dataset]
    extracted = data_root / record["source"]["extracted_directory"]
    labels_array = np.load(extracted / "labels.npy", mmap_mode="r", allow_pickle=False)
    primary_array = np.load(extracted / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
    paired_array = np.load(extracted / f"{paired_corruption}.npy", mmap_mode="r", allow_pickle=False)
    block_size = int(dataset_provenance["severity_block_size"])
    primary, labels, sample_range = extract_severity_block(
        primary_array, labels_array, severity=severity, block_size=block_size
    )
    paired, paired_labels, paired_range = extract_severity_block(
        paired_array, labels_array, severity=severity, block_size=block_size
    )
    if sample_range != paired_range or not np.array_equal(labels, paired_labels):
        raise ValueError("paired corruption labels or severity ranges do not align")

    clean, clean_labels = method_engine.load_cifar(
        DATASET_MAP[dataset], ROOT / "data", train=False, download=False
    )
    limit = min(len(clean), block_size)
    if max_samples:
        limit = min(limit, max_samples)
    labels_tensor = _to_label_tensor(labels, limit)
    clean = clean[:limit]
    clean_labels = clean_labels[:limit].to(torch.int64)
    if not torch.equal(clean_labels.cpu(), labels_tensor.cpu()):
        raise ValueError("official CIFAR-C labels do not align with the clean CIFAR test set")
    identity = {
        "protocol": dataset_provenance["protocol"],
        "source_dataset": DATASET_MAP[dataset],
        "external_dataset": dataset,
        "severity": severity,
        "severity_sample_range": sample_range,
        "corruption": corruption,
        "paired_corruption": paired_corruption,
        "archive_sha256": record["archive"]["sha256"],
        "labels_file_sha256": _file_record(record, "labels.npy")["sha256"],
        "corruption_file_sha256": _file_record(record, f"{corruption}.npy")["sha256"],
        "paired_corruption_file_sha256": _file_record(
            record, f"{paired_corruption}.npy"
        )["sha256"],
        "data_source": "REAL_OFFICIAL_CIFAR_C",
    }
    return (
        clean,
        labels_tensor,
        _to_image_tensor(primary, limit),
        _to_image_tensor(paired, limit),
        identity,
    )


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    verify_expected_digest(
        args.dataset_provenance, args.dataset_provenance_sha256, "dataset provenance"
    )
    verify_expected_digest(
        ROOT / "publication_expansion_experiment.py",
        args.method_engine_sha256,
        "method engine",
    )
    source_checkpoint_sha256 = verify_expected_digest(
        args.source_checkpoint,
        args.expected_source_checkpoint_sha256,
        "source checkpoint",
    )
    environment_identity = runtime_environment_identity(args.device)
    if json_digest(environment_identity) != args.environment_sha256:
        raise ValueError("runtime environment changed after campaign freeze")
    dataset_provenance = json.loads(args.dataset_provenance.read_text(encoding="utf-8"))
    clean, labels, stage_b, stage_c, dataset_identity = load_external_inputs(
        dataset=args.dataset,
        corruption=args.corruption,
        paired_corruption=args.paired_corruption,
        severity=args.severity,
        data_root=args.data_root,
        dataset_provenance=dataset_provenance,
        max_samples=args.max_test_samples,
    )
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    expected_source = DATASET_MAP[args.dataset]
    if checkpoint.get("dataset") not in {None, expected_source}:
        raise ValueError(
            f"source checkpoint dataset mismatch: {checkpoint.get('dataset')!r} != {expected_source!r}"
        )
    model = method_engine.cifar_resnet18(int(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(args.device)
    episode = run_external_episode(
        model,
        clean,
        labels,
        stage_b=stage_b,
        stage_c=stage_c,
        method=args.method,
        trajectory=args.trajectory,
        seed=args.seed,
        batch_size=args.batch_size,
        adaptation_passes=args.pass_count,
        learning_rate=args.learning_rate,
        normalization_policy=args.normalization_policy,
    )
    registry_path = ROOT / "configs" / "modern_baseline_upstreams.json"
    contracts_path = ROOT / "configs" / "tta_method_state_contracts.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = registry.get("providers", {}).get(args.method)
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "evidence_label": "SEMANTIC_CANARY" if args.canary else ("SMOKE" if args.smoke else "FORMAL_EXTERNAL_VALIDATION"),
        "job_id": args.job_id,
        "campaign_id": args.campaign_id,
        "dataset": args.dataset,
        "method": args.method,
        "trajectory": args.trajectory,
        "corruption": args.corruption,
        "paired_corruption": args.paired_corruption,
        "severity": args.severity,
        "seed": args.seed,
        "normalization_policy": args.normalization_policy,
        "dataset_identity": dataset_identity,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "runner_sha256": sha256(Path(__file__)),
        "supervisor_sha256": args.supervisor_sha256,
        "manifest_sha256": args.manifest_sha256,
        "dataset_provenance_sha256": args.dataset_provenance_sha256,
        "external_source_registry_sha256": args.external_source_registry_sha256,
        "clean_dataset_provenance_sha256": args.clean_dataset_provenance_sha256,
        "method_engine_sha256": args.method_engine_sha256,
        "checkpoint_inventory_sha256": args.checkpoint_inventory_sha256,
        "environment_sha256": args.environment_sha256,
        "method_registry_sha256": sha256(registry_path),
        "method_contracts_sha256": sha256(contracts_path),
        "method_upstream": {
            "binding": "PINNED_UPSTREAM_BEHAVIOR_REFERENCE" if provider else "LOCAL_CONTROL_IMPLEMENTATION",
            "registry_sha256": sha256(registry_path),
            "provider": provider,
        },
        "protocol_parameters": {
            "batch_size": args.batch_size,
            "pass_count": args.pass_count,
            "adaptation_learning_rate": args.learning_rate,
            "normalization_policy": args.normalization_policy,
            "labels_available_for_evaluation_only": True,
        },
        "episode": episode,
        "environment": environment_identity,
        "created_at": utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_MAP), required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--paired-corruption", required=True)
    parser.add_argument("--severity", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=list(method_engine.METHODS), required=True)
    parser.add_argument("--trajectory", choices=["A-B-A", "A-B-C-A"], required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pass-count", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--normalization-policy", default="bn_affine_frozen_stats")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--supervisor-sha256", required=True)
    parser.add_argument("--dataset-provenance-sha256", required=True)
    parser.add_argument("--external-source-registry-sha256", required=True)
    parser.add_argument("--clean-dataset-provenance-sha256", required=True)
    parser.add_argument("--method-engine-sha256", required=True)
    parser.add_argument("--checkpoint-inventory-sha256", required=True)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    result = run_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite external result: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
