"""
Run provenance: environment, package versions, git state and cache-file hashes, saved alongside outputs for audit and reproduction.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

_PACKAGES = (
    "numpy",
    "pandas",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "huggingface_hub",
    "pillow",
)
_DIFF_LIMIT = 200_000
_CHUNK = 8 * 1024 * 1024


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Call `fn`, returning `default` instead of raising."""
    try:
        return fn()
    except Exception:
        return default


def _cv2_version() -> str:
    import cv2  # import name differs from the distribution name

    return cv2.__version__


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        if name == "opencv-python-headless":
            versions[name] = _safe(_cv2_version)
        else:
            versions[name] = _safe(lambda n=name: metadata.version(n))
    return versions


def _torch_info() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    cuda = bool(_safe(torch.cuda.is_available, False))
    return {
        "version": _safe(lambda: torch.__version__),
        "cuda_available": cuda,
        "cuda_version": _safe(lambda: torch.version.cuda),
        "cudnn_version": _safe(torch.backends.cudnn.version),
        "device_name": _safe(lambda: torch.cuda.get_device_name(0)) if cuda else None,
        "mps_available": bool(_safe(lambda: torch.backends.mps.is_available(), False)),
    }


def _nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return None
    return proc.stdout if proc.returncode == 0 else None


def _git(repo_dir: Path, *args: str) -> str | None:
    """Raw stdout of a git command run in `repo_dir`, or None if it fails or git is unavailable."""
    try:
        proc = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, timeout=60, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def _git_info(repo_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None, "diff": None}
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return info
    info["commit"] = commit.strip()
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    info["branch"] = branch.strip() if branch is not None else None
    status = _git(repo_dir, "status", "--porcelain")
    info["dirty"] = bool(status.strip()) if status is not None else None
    diff = _git(repo_dir, "diff", "HEAD")
    info["diff"] = diff[:_DIFF_LIMIT] if diff is not None else None
    return info


def _source_version(repo_dir: Path) -> str | None:
    # Remote VMs receive the code as a tarball, which has no .git directory but carries this file.
    path = repo_dir / "SOURCE_VERSION"
    if not path.is_file():
        return None
    return path.read_text().strip()


def _cache_entry(p: str | Path) -> dict[str, Any]:
    path = Path(p)
    if not path.is_file():
        return {"path": str(p), "missing": True}
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(p), "bytes": size, "sha256": digest.hexdigest()}


def collect_provenance(repo_dir: str | Path, cache_files: list[str | Path] = (), extra: dict | None = None) -> dict:
    """Everything needed to reproduce or audit a run, as a JSON-serialisable dict."""
    repo = Path(repo_dir)
    git = _git_info(repo)
    git["source_version"] = _safe(lambda: _source_version(repo))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "git": git,
        "cache_files": [_cache_entry(p) for p in cache_files],
        "extra": extra or {},
    }


def write_provenance(path: str | Path, repo_dir: str | Path, cache_files: list[str | Path] = (), extra: dict | None = None) -> dict:
    """collect_provenance(...) written to `path` as indented JSON (parent dirs created). Returns the dict."""
    record = collect_provenance(repo_dir, cache_files, extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return record
