from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_cifar_c_data", ROOT / "prepare_cifar_c_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeArray:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


def build_fake_dataset(tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "archives" / "CIFAR-X-C.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"official-archive-fixture")
    extracted = tmp_path / "CIFAR-X-C"
    extracted.mkdir()
    (extracted / "labels.npy").write_bytes(b"labels")
    for corruption in ("gaussian_noise", "motion_blur"):
        (extracted / f"{corruption}.npy").write_bytes(corruption.encode("ascii"))

    def fake_load(path, **_kwargs):
        path = Path(path)
        if path.name == "labels.npy":
            return FakeArray((10000,), np.dtype("int64"))
        return FakeArray((50000, 32, 32, 3), np.dtype("uint8"))

    monkeypatch.setattr(module.np, "load", fake_load)
    source = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        "extracted_directory": extracted.name,
    }
    return source, archive


def test_inspect_dataset_supports_external_data_root(tmp_path, monkeypatch):
    module = load_module()
    source, _archive = build_fake_dataset(tmp_path, module, monkeypatch)
    result = module.inspect_dataset(
        tmp_path, "cifar_x_c", source, ["gaussian_noise", "motion_blur"]
    )
    assert result["archive"]["path"].endswith("CIFAR-X-C.tar")
    assert len(result["files"]) == 3
    assert all(entry["sha256"].startswith("sha256:") for entry in result["files"])


def test_inspect_dataset_rejects_archive_digest_mismatch(tmp_path, monkeypatch):
    module = load_module()
    source, _archive = build_fake_dataset(tmp_path, module, monkeypatch)
    source["md5"] = "0" * 32
    with pytest.raises(ValueError, match="archive MD5 mismatch"):
        module.inspect_dataset(
            tmp_path, "cifar_x_c", source, ["gaussian_noise", "motion_blur"]
        )


@pytest.mark.parametrize("labels_count", [10000, 50000])
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_severity_slice_and_labels_alignment(labels_count, severity):
    module = load_module()
    image_slice, label_slice = module.severity_slices(
        severity=severity, labels_count=labels_count, block_size=10000
    )
    start = (severity - 1) * 10000
    assert image_slice == slice(start, start + 10000)
    expected_labels = slice(0, 10000) if labels_count == 10000 else image_slice
    assert label_slice == expected_labels


@pytest.mark.parametrize("severity", [0, 6])
def test_severity_slice_rejects_invalid_severity(severity):
    module = load_module()
    with pytest.raises(ValueError, match="severity"):
        module.severity_slices(severity=severity, labels_count=10000, block_size=10000)


def test_severity_slice_rejects_unsupported_labels_shape():
    module = load_module()
    with pytest.raises(ValueError, match="labels count"):
        module.severity_slices(severity=1, labels_count=123, block_size=10000)


def test_manifest_write_is_idempotent_but_refuses_identity_change(tmp_path):
    module = load_module()
    output = tmp_path / "provenance.json"
    original = {
        "schema_version": "1.0.0",
        "verified_at": "2026-01-01T00:00:00+00:00",
        "datasets": {"cifar10_c": {"archive": {"sha256": "sha256:a"}}},
    }
    assert module.write_manifest_immutably(output, original) == "CREATED"
    same_identity = dict(original, verified_at="2026-09-18T00:00:00+00:00")
    assert module.write_manifest_immutably(output, same_identity) == "REUSED"
    assert json.loads(output.read_text(encoding="utf-8"))["verified_at"] == original["verified_at"]
    changed = {
        **same_identity,
        "datasets": {"cifar10_c": {"archive": {"sha256": "sha256:changed"}}},
    }
    with pytest.raises(FileExistsError, match="identity changed"):
        module.write_manifest_immutably(output, changed)
