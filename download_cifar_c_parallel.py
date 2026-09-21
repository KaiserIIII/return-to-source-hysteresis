from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def plan_chunks(*, total_bytes: int, chunk_bytes: int) -> list[Chunk]:
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("total_bytes and chunk_bytes must be positive")
    chunks: list[Chunk] = []
    start = 0
    while start < total_bytes:
        end = min(total_bytes - 1, start + chunk_bytes - 1)
        chunks.append(Chunk(index=len(chunks), start=start, end=end))
        start = end + 1
    return chunks


def chunk_path(root: Path, chunk: Chunk) -> Path:
    return root / f"chunk_{chunk.index:05d}_{chunk.start:012d}_{chunk.end:012d}.complete"


def valid_archive(path: Path, *, total_bytes: int, expected_md5: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == total_bytes
        and digest(path, "md5") == expected_md5.lower()
    )


def assemble_and_verify(
    *,
    chunks: list[Chunk],
    chunk_root: Path,
    output: Path,
    total_bytes: int,
    expected_md5: str,
) -> dict[str, Any]:
    if output.exists():
        if valid_archive(output, total_bytes=total_bytes, expected_md5=expected_md5):
            return {"path": str(output), "bytes": total_bytes, "md5": expected_md5.lower()}
        raise ValueError(f"refusing to overwrite invalid existing archive: {output}")
    for chunk in chunks:
        path = chunk_path(chunk_root, chunk)
        observed = path.stat().st_size if path.is_file() else None
        if observed != chunk.size:
            raise ValueError(
                f"chunk size mismatch: index={chunk.index}, expected={chunk.size}, observed={observed}"
            )
    assembling = output.with_name(output.name + ".assembling")
    output.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    written = 0
    with assembling.open("wb") as target:
        for chunk in chunks:
            with chunk_path(chunk_root, chunk).open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    md5.update(block)
                    written += len(block)
        target.flush()
        os.fsync(target.fileno())
    observed_md5 = md5.hexdigest()
    if written != total_bytes:
        raise ValueError(f"assembled size mismatch: expected={total_bytes}, observed={written}")
    if observed_md5 != expected_md5.lower():
        raise ValueError(
            f"assembled MD5 mismatch: expected={expected_md5.lower()}, observed={observed_md5}"
        )
    os.replace(assembling, output)
    return {"path": str(output), "bytes": written, "md5": observed_md5}


def log_event(log_path: Path, event: str, **values: Any) -> None:
    record = {"timestamp": utc(), "event": event, **values}
    line = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with _LOG_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _download_chunk(
    *,
    chunk: Chunk,
    chunk_root: Path,
    url: str,
    addresses: list[str],
    log_path: Path,
) -> dict[str, Any]:
    final = chunk_path(chunk_root, chunk)
    if final.is_file() and final.stat().st_size == chunk.size:
        return {"index": chunk.index, "status": "REUSED", "bytes": chunk.size}
    chunk_root.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(".tmp")
    last_error = ""
    for attempt, address in enumerate(addresses, start=1):
        command = [
            "curl.exe",
            "-4",
            "-L",
            "--silent",
            "--show-error",
            "--fail",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "20",
            "--max-time",
            "7200",
            "--range",
            f"{chunk.start}-{chunk.end}",
            "--resolve",
            f"zenodo.org:443:{address}",
            "--output",
            str(temporary),
            url,
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        observed = temporary.stat().st_size if temporary.is_file() else None
        if process.returncode == 0 and observed == chunk.size:
            os.replace(temporary, final)
            log_event(
                log_path,
                "CHUNK_READY",
                chunk=chunk.index,
                start=chunk.start,
                end=chunk.end,
                bytes=chunk.size,
                endpoint=address,
                attempt=attempt,
            )
            return {"index": chunk.index, "status": "DOWNLOADED", "bytes": chunk.size}
        last_error = (process.stderr or process.stdout or "unknown curl error")[-1000:]
        log_event(
            log_path,
            "CHUNK_ENDPOINT_FAILED",
            chunk=chunk.index,
            endpoint=address,
            attempt=attempt,
            exit_code=process.returncode,
            expected_bytes=chunk.size,
            observed_bytes=observed,
            error=last_error,
        )
    raise RuntimeError(f"chunk {chunk.index} failed across all endpoints: {last_error}")


def download_archive(
    *,
    name: str,
    url: str,
    output: Path,
    total_bytes: int,
    expected_md5: str,
    addresses: list[str],
    chunk_bytes: int,
    workers: int,
    log_path: Path,
) -> dict[str, Any]:
    if valid_archive(output, total_bytes=total_bytes, expected_md5=expected_md5):
        log_event(log_path, "ARCHIVE_REUSED", archive=name, bytes=total_bytes)
        return {"path": str(output), "bytes": total_bytes, "md5": expected_md5.lower()}
    if output.exists():
        raise ValueError(f"existing archive failed integrity checks: {output}")
    chunks = plan_chunks(total_bytes=total_bytes, chunk_bytes=chunk_bytes)
    chunk_root = output.parent / f".{output.name}.chunks"
    log_event(
        log_path,
        "ARCHIVE_DOWNLOAD_START",
        archive=name,
        total_bytes=total_bytes,
        chunk_bytes=chunk_bytes,
        chunks=len(chunks),
        workers=workers,
        endpoints=addresses,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_chunk,
                chunk=chunk,
                chunk_root=chunk_root,
                url=url,
                addresses=addresses[chunk.index % len(addresses) :] + addresses[: chunk.index % len(addresses)],
                log_path=log_path,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"chunk {chunk.index}: {exc}")
                log_event(log_path, "CHUNK_FAILED", chunk=chunk.index, error=str(exc))
    if failures:
        raise RuntimeError(f"{len(failures)} chunks failed; first failure: {failures[0]}")
    result = assemble_and_verify(
        chunks=chunks,
        chunk_root=chunk_root,
        output=output,
        total_bytes=total_bytes,
        expected_md5=expected_md5,
    )
    log_event(log_path, "ARCHIVE_READY", archive=name, **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/external_cifar_c_sources.json")
    parser.add_argument("--archive-root", type=Path, default=ROOT / "data/external/archives")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/external_data/cifar_c_parallel_download.jsonl")
    parser.add_argument("--resolve-address", action="append", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_mib <= 0:
        parser.error("workers and chunk-mib must be positive")
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    args.archive_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for dataset, source in sources["datasets"].items():
        outputs[dataset] = download_archive(
            name=source["archive"],
            url=source["url"],
            output=args.archive_root / source["archive"],
            total_bytes=int(source["bytes"]),
            expected_md5=source["md5"],
            addresses=list(dict.fromkeys(args.resolve_address)),
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            workers=args.workers,
            log_path=args.log,
        )
    print(json.dumps({"status": "PASS", "archives": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
