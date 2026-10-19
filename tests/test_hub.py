"""Hub upload and verify must round-trip a run dir, and catch missing or truncated files, without touching the network."""

from __future__ import annotations

import hashlib
import json

import pytest

from screenpad.train.hub import ensure_repo, local_manifest, upload_run, verify_run

REPO = "user/pad-run"
PATH = "runs/exp1"


class _File:
    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size


class _Folder:
    def __init__(self, path: str):
        self.path = path


class FakeApi:
    """In-memory stand-in for huggingface_hub.HfApi covering only the methods hub.py uses."""

    def __init__(self):
        self.hub: dict[str, bytes] = {}
        self.created: list[dict] = []

    def upload_folder(self, *, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        for p in sorted(folder_path.rglob("*")):
            if p.is_file():
                rel = p.relative_to(folder_path).as_posix()
                self.hub[f"{path_in_repo}/{rel}"] = p.read_bytes()

    def list_repo_tree(self, repo_id, *, path_in_repo, recursive, repo_type):
        prefix = path_in_repo + "/"
        items = [_Folder(prefix + "sub")]
        items += [_File(path, len(data)) for path, data in sorted(self.hub.items()) if path.startswith(prefix)]
        return items

    def create_repo(self, repo_id, **kwargs):
        self.created.append({"repo_id": repo_id, **kwargs})


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run"
    (root / "ckpt").mkdir(parents=True)
    (root / "ckpt" / "last.pt").write_bytes(b"weights" * 100)
    (root / "metrics.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n")
    return root


@pytest.fixture
def uploaded(run_dir):
    api = FakeApi()
    upload_run(run_dir, REPO, PATH, api=api)
    return api


def test_upload_then_verify_is_clean(run_dir, uploaded):
    assert verify_run(run_dir, REPO, PATH, api=uploaded) == []


def test_deleted_file_reported_missing(run_dir, uploaded):
    del uploaded.hub[f"{PATH}/ckpt/last.pt"]
    assert verify_run(run_dir, REPO, PATH, api=uploaded) == ["missing on hub: ckpt/last.pt"]


def test_size_mismatch_reported(run_dir, uploaded):
    local_len = (run_dir / "metrics.csv").stat().st_size
    uploaded.hub[f"{PATH}/metrics.csv"] = b"x" * (local_len - 2)
    assert verify_run(run_dir, REPO, PATH, api=uploaded) == [
        f"size mismatch: metrics.csv local {local_len} hub {local_len - 2}"
    ]


def test_manifest_written_and_hashes_match(run_dir, uploaded):
    manifest_path = run_dir / "MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "MANIFEST.json" not in manifest
    assert set(manifest) == {"ckpt/last.pt", "metrics.csv"}
    for rel, meta in manifest.items():
        data = (run_dir / rel).read_bytes()
        assert meta["bytes"] == len(data)
        assert meta["sha256"] == hashlib.sha256(data).hexdigest()


def test_ensure_repo_creates_private_and_exist_ok():
    api = FakeApi()
    ensure_repo(REPO, api=api)
    assert api.created == [{"repo_id": REPO, "repo_type": "model", "private": True, "exist_ok": True}]
