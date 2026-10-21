"""`screenpad infer` finds weights where the frozen ensemble points, or in a previous download, before fetching."""

from __future__ import annotations

import hashlib
import json
import logging

from screenpad.infer import _has_weights, resolve_frozen


def _frozen(root):
    return {"enabled": ["A", "pc"], "roots": {"runs": str(root / "runs"), "oneclass": str(root / "oc")},
            "members": {"A": {"type": "run", "runs": ["r1"]}, "pc": {"type": "oneclass", "path": "p"}}}


def _touch(root):
    for f in ("runs/r1/config.yaml", "runs/r1/ckpt/last.pt", "oc/p/bank.pt"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x")


def test_has_weights(tmp_path):
    f = _frozen(tmp_path)
    assert not _has_weights(f)
    _touch(tmp_path)
    assert _has_weights(f)
    (tmp_path / "oc/p/bank.pt").unlink()
    assert not _has_weights(f)


def test_resolve_uses_previous_download(tmp_path):
    src = tmp_path / "frozen.json"
    src.write_text(json.dumps(_frozen(tmp_path / "missing")))
    models = tmp_path / "models"
    local = _frozen(models)
    _touch(models)
    local["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    (models / "frozen.json").write_text(json.dumps(local))
    assert resolve_frozen(src, models, logging.getLogger("t"))["roots"]["runs"] == str(models / "runs")
