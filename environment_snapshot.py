from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def snapshot() -> dict:
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "UNAVAILABLE"
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        git_sha = git.stdout.strip() or "NOT_A_GIT_CHECKOUT"
    except OSError:
        git_sha = "GIT_UNAVAILABLE"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda": __import__("torch").version.cuda or "N/A",
        "cuda_available": bool(__import__("torch").cuda.is_available()),
        "git_sha": git_sha,
        "determinism_notes": ["CPU-first", "torch deterministic algorithms enabled where supported", "thread count fixed to 1 per job"],
    }


if __name__ == "__main__":
    out = ROOT / "results" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot(), indent=2))

