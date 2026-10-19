"""Probe set assembly: no real attacks in fitting/selection, test never loaded, bona fide filter is training-only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screenpad.data.cache import save_cache
from screenpad.probes.data import load_sets, set_masks


def _cache(rows, path):
    n = len(rows)
    df = pd.DataFrame(rows)
    save_cache({"images": np.zeros((n, 4, 4, 3), np.uint8), **{c: np.asarray(df[c].tolist(), dtype=str) for c in ("id", "split", "path")},
                "label": df.label.to_numpy(np.int8), "info_face_found": df.face.to_numpy(bool), "info_shrunk": df.shrunk.to_numpy(bool),
                "info_resample_ratio": np.ones(n), **({"kind": np.asarray(df.kind.tolist(), dtype=str), "variant": df.variant.to_numpy()} if "kind" in df else {})}, path)


def test_sets(tmp_path):
    real = [
        {"id": "1", "split": "train", "path": "a", "label": 0, "face": True, "shrunk": False},
        {"id": "2", "split": "train", "path": "b", "label": 0, "face": True, "shrunk": True},  # filtered from training
        {"id": "3", "split": "val", "path": "c", "label": 0, "face": True, "shrunk": True},  # kept: evaluation is never filtered
        {"id": "4", "split": "dev", "path": "d", "label": 0, "face": True, "shrunk": False},
        {"id": "4", "split": "dev", "path": "e", "label": 1, "face": True, "shrunk": True},
        {"id": "5", "split": "test", "path": "f", "label": 1, "face": True, "shrunk": False},
    ]
    syn = [
        {"id": "1", "split": "train", "path": "g", "label": 1, "face": False, "shrunk": True, "kind": "synthetic_attack", "variant": 0},  # attacks always kept
        {"id": "3", "split": "val", "path": "h", "label": 1, "face": True, "shrunk": False, "kind": "synthetic_attack", "variant": 1},
        {"id": "3", "split": "val", "path": "i", "label": 1, "face": True, "shrunk": False, "kind": "synthetic_attack", "variant": 3},  # outside the fixed draw
    ]
    _cache(real, tmp_path / "crops_tight_4_aligned.npz")
    _cache(syn, tmp_path / "synth_tight_4_train-val.npz")
    images, table = load_sets(tmp_path, "tight", 4)
    assert "test" not in set(table.split) and len(images) == len(table) == 8
    m = set_masks(table)
    assert set(table[m["train"]].path) == {"a", "g"}
    assert set(table[m["val"]].path) == {"c", "h"}
    assert set(table[m["dev"]].path) == {"c", "d", "e"}
