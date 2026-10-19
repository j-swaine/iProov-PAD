"""Provenance records must be JSON-safe, hash cache files exactly and fall back to SOURCE_VERSION outside git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from screenpad.utils.provenance import collect_provenance, write_provenance

REPO = Path(__file__).resolve().parents[1]


def test_cache_file_hashed_and_json_safe(tmp_path):
    cache = tmp_path / "scores.npy"
    cache.write_bytes(b"some cached bytes" * 1000)
    out = collect_provenance(REPO, cache_files=[cache])
    entry = out["cache_files"][0]
    assert entry["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert entry["bytes"] == cache.stat().st_size
    json.dumps(out)  # must not need default=str


def test_missing_cache_file_marked(tmp_path):
    missing = tmp_path / "nope.bin"
    entry = collect_provenance(REPO, cache_files=[missing])["cache_files"][0]
    assert entry["missing"] is True
    assert "sha256" not in entry


def test_write_provenance_round_trips(tmp_path):
    cache = tmp_path / "c.bin"
    cache.write_bytes(b"abc")
    target = tmp_path / "nested" / "dir" / "provenance.json"
    returned = write_provenance(target, REPO, cache_files=[cache], extra={"run": "unit"})
    assert target.is_file()
    with open(target) as fh:
        loaded = json.load(fh)
    assert loaded == json.loads(json.dumps(returned))
    assert loaded["extra"] == {"run": "unit"}


def test_source_version_used_outside_git(tmp_path):
    (tmp_path / "SOURCE_VERSION").write_text("abc123\n")
    git = collect_provenance(tmp_path)["git"]
    assert git["source_version"] == "abc123"
    assert git["commit"] is None
