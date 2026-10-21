"""Preflight pieces that need no models: the golden inputs are fixed, and the input scan flags problem files."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from screenpad.preflight import _scan, golden_images


def test_golden_images_are_fixed():
    a, b = golden_images(), golden_images()
    assert list(a) == ["seeded_noise", "smooth_gradient"]
    for k in a:
        assert a[k].shape == (512, 512, 3) and a[k].dtype == np.uint8 and np.array_equal(a[k], b[k])


def test_scan_flags_problem_files(tmp_path):
    rgb = np.zeros((200, 200, 3), np.uint8)
    Image.fromarray(rgb).save(tmp_path / "ok.png")
    Image.fromarray(rgb[:64, :64]).save(tmp_path / "small.png")
    Image.fromarray(np.zeros((200, 200), np.uint16) + 1000).save(tmp_path / "deep.png")
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG")
    (tmp_path / "cut.jpg").write_bytes(b.getvalue()[:200])
    (tmp_path / "x.txt").write_bytes(b"x")
    sc = _scan(tmp_path, recursive=False)
    assert sc["images"] == 4 and sc["unsupported"] == ["x.txt"]
    assert [u.split()[0] for u in sc["unreadable"]] == ["cut.jpg"]
    assert sc["small"] == ["small.png (64x64)"] and sc["high_bit_depth"] == ["deep.png"]
