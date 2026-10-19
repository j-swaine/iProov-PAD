"""
Hub upload and verification for training runs.
----------------------------------------------

A run directory (checkpoints, metrics, logs) goes to the HF Hub in one commit, with a MANIFEST.json of every file's byte size and SHA-256. `verify_run` then checks the Hub listing against that manifest, so a partial or truncated upload is caught before the local copy is deleted.

Repos are always created private: checkpoints and logs are not meant to be public.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 8 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def local_manifest(run_dir: str | Path) -> dict[str, dict]:
    """Every file under run_dir (recursive), keyed by POSIX path relative to run_dir: {"bytes": int, "sha256": hex}.

    The root-level MANIFEST.json is skipped so a manifest never lists itself.
    """
    run_dir = Path(run_dir)
    out = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel == MANIFEST_NAME:
            continue
        out[rel] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return out


def _default_api():
    from huggingface_hub import HfApi

    return HfApi()


def upload_run(run_dir: str | Path, repo_id: str, path_in_repo: str, api=None, repo_type: str = "model",
               commit_message: str | None = None) -> dict[str, dict]:
    """Upload the whole run_dir to `repo_id` under `path_in_repo` in one commit, after writing run_dir/MANIFEST.json.

    Returns the manifest (every file except MANIFEST.json itself).
    """
    run_dir = Path(run_dir)
    path_in_repo = path_in_repo.strip("/")
    api = api or _default_api()
    manifest = local_manifest(run_dir)
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=run_dir,
        path_in_repo=path_in_repo,
        commit_message=commit_message or f"Upload run {run_dir.name}",
    )
    return manifest


def verify_run(run_dir: str | Path, repo_id: str, path_in_repo: str, api=None, repo_type: str = "model") -> list[str]:
    """Check every file in run_dir/MANIFEST.json (plus MANIFEST.json itself) exists on the Hub with the same size.

    Returns a list of human-readable problems; an empty list means the upload is verified.
    """
    run_dir = Path(run_dir)
    path_in_repo = path_in_repo.strip("/")
    api = api or _default_api()
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text())

    hub_sizes = {}
    for item in api.list_repo_tree(repo_id, path_in_repo=path_in_repo, recursive=True, repo_type=repo_type):
        if hasattr(item, "size"):
            hub_sizes[item.path] = item.size

    local = {rel: meta["bytes"] for rel, meta in manifest.items()}
    local[MANIFEST_NAME] = (run_dir / MANIFEST_NAME).stat().st_size

    problems = []
    for rel in sorted(local):
        remote_path = f"{path_in_repo}/{rel}"
        if remote_path not in hub_sizes:
            problems.append(f"missing on hub: {rel}")
        elif hub_sizes[remote_path] != local[rel]:
            problems.append(f"size mismatch: {rel} local {local[rel]} hub {hub_sizes[remote_path]}")
    return problems


def ensure_repo(repo_id: str, api=None, repo_type: str = "model") -> None:
    """Create the repo if missing, always private."""
    api = api or _default_api()
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
