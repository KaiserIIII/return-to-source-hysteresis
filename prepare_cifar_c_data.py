from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def provenance_path(path: Path) -> str:
    """Return a portable project path or an explicit absolute URI."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def severity_slices(*, severity: int, labels_count: int, block_size: int) -> tuple[slice, slice]:
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError(f"severity must be in 1..5, got {severity}")
    if labels_count not in {block_size, 5 * block_size}:
        raise ValueError(
            f"labels count must be {block_size} or {5 * block_size}, got {labels_count}"
        )
    start = (severity - 1) * block_size
    image_slice = slice(start, start + block_size)
    label_slice = slice(0, block_size) if labels_count == block_size else image_slice
    return image_slice, label_slice


def write_manifest_immutably(path: Path, manifest: dict[str, Any]) -> str:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_identity = {key: value for key, value in existing.items() if key != "verified_at"}
        candidate_identity = {key: value for key, value in manifest.items() if key != "verified_at"}
        if existing_identity != candidate_identity:
            raise FileExistsError(
                f"dataset provenance identity changed; refusing to overwrite: {path}"
            )
        return "REUSED"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return "CREATED"


def inspect_dataset(root: Path, dataset: str, source: dict[str, Any], corruptions: list[str]) -> dict[str, Any]:
    archive = root / "archives" / source["archive"]
    extracted = root / source["extracted_directory"]
    if not archive.is_file():
        raise FileNotFoundError(f"missing archive: {archive}")
    if archive.stat().st_size != int(source["bytes"]):
        raise ValueError(f"archive size mismatch: {archive}")
    observed_md5 = digest(archive, "md5")
    if observed_md5 != source["md5"]:
        raise ValueError(f"archive MD5 mismatch: {archive}")
    labels_path = extracted / "labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing labels: {labels_path}")
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 1 or labels.shape[0] not in {10000, 50000}:
        raise ValueError(f"unexpected labels shape for {dataset}: {labels.shape}")
    files: list[dict[str, Any]] = []
    required = [labels_path]
    for corruption in corruptions:
        path = extracted / f"{corruption}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing corruption array: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (50000, 32, 32, 3) or array.dtype != np.uint8:
            raise ValueError(f"unexpected {dataset}/{corruption} array: shape={array.shape}, dtype={array.dtype}")
        required.append(path)
    for path in required:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        files.append(
            {
                "path": provenance_path(path),
                "bytes": path.stat().st_size,
                "sha256": "sha256:" + digest(path, "sha256"),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return {
        "source": source,
        "archive": {
            "path": provenance_path(archive),
            "bytes": archive.stat().st_size,
            "md5": "md5:" + observed_md5,
            "sha256": "sha256:" + digest(archive, "sha256"),
        },
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/external_cifar_c_sources.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/cifar_c_dataset_provenance_v1.json")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    datasets = {
        name: inspect_dataset(args.data_root, name, source, list(sources["corruptions"]))
        for name, source in sources["datasets"].items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "protocol": sources["protocol"],
        "sources_sha256": "sha256:" + hashlib.sha256(args.sources.read_bytes()).hexdigest(),
        "corruptions": sources["corruptions"],
        "severity_block_size": sources["severity_block_size"],
        "datasets": datasets,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    disposition = write_manifest_immutably(args.output, manifest)
    print(json.dumps({"status": "PASS", "output": str(args.output), "datasets": sorted(datasets), "disposition": disposition}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
