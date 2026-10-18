"""Crop cache: saved arrays must reload identically and match a fresh recompute."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from screenpad.data.cache import build_crops, load_cache, save_cache, verify_cache


@pytest.fixture
def tiny_set(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i, (folder, label) in enumerate((("Bonafide", 0), ("Screens", 1))):
        (tmp_path / folder).mkdir()
        p = f"{folder}/{i:05d}.png"
        Image.fromarray(rng.integers(0, 255, (120, 90, 3), dtype=np.uint8)).save(tmp_path / p)
        rows.append({"id": f"{i:05d}", "label": label, "split": "train", "path": p})
    return tmp_path, pd.DataFrame(rows)


def test_round_trip_and_verify(tiny_set, tmp_path):
    root, splits = tiny_set
    caches = build_crops(splits, root, {"tight": 1.0, "loose": 1.25}, size=32, det_size=128)
    for name, c in caches.items():
        path = tmp_path / f"{name}.npz"
        save_cache(c, path)
        back = load_cache(path)
        assert np.array_equal(back["images"], c["images"]) and back["images"].dtype == np.uint8
        assert list(back["id"]) == list(splits["id"])
        assert verify_cache(path, root, det_size=128) == 2


def test_verify_catches_a_changed_pixel(tiny_set, tmp_path):
    root, splits = tiny_set
    c = build_crops(splits, root, {"tight": 1.0}, size=32, det_size=128)["tight"]
    c["images"][1, 0, 0, 0] ^= 1
    save_cache(c, tmp_path / "t.npz")
    with pytest.raises(ValueError, match="1 crops differ"):
        verify_cache(tmp_path / "t.npz", root, det_size=128)
