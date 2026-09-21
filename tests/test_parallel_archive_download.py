from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def test_chunk_plan_is_contiguous_and_exhaustive() -> None:
    import download_cifar_c_parallel as downloader

    chunks = downloader.plan_chunks(total_bytes=25, chunk_bytes=8)
    assert [(chunk.start, chunk.end, chunk.size) for chunk in chunks] == [
        (0, 7, 8),
        (8, 15, 8),
        (16, 23, 8),
        (24, 24, 1),
    ]


def test_assemble_requires_every_exact_chunk_and_verifies_md5(tmp_path: Path) -> None:
    import download_cifar_c_parallel as downloader

    content = b"abcdefghijklmnopqrstuvwxyz"
    chunks = downloader.plan_chunks(total_bytes=len(content), chunk_bytes=7)
    chunk_root = tmp_path / "chunks"
    chunk_root.mkdir()
    for chunk in chunks:
        downloader.chunk_path(chunk_root, chunk).write_bytes(
            content[chunk.start : chunk.end + 1]
        )
    output = tmp_path / "archive.tar"
    expected_md5 = hashlib.md5(content).hexdigest()
    result = downloader.assemble_and_verify(
        chunks=chunks,
        chunk_root=chunk_root,
        output=output,
        total_bytes=len(content),
        expected_md5=expected_md5,
    )
    assert output.read_bytes() == content
    assert result["md5"] == expected_md5
    assert result["bytes"] == len(content)

    downloader.chunk_path(chunk_root, chunks[-1]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="chunk size mismatch"):
        downloader.assemble_and_verify(
            chunks=chunks,
            chunk_root=chunk_root,
            output=tmp_path / "bad.tar",
            total_bytes=len(content),
            expected_md5=expected_md5,
        )


def test_existing_archive_must_match_size_and_md5(tmp_path: Path) -> None:
    import download_cifar_c_parallel as downloader

    output = tmp_path / "archive.tar"
    output.write_bytes(b"official")
    assert downloader.valid_archive(
        output,
        total_bytes=8,
        expected_md5=hashlib.md5(b"official").hexdigest(),
    )
    assert not downloader.valid_archive(
        output,
        total_bytes=8,
        expected_md5="0" * 32,
    )
