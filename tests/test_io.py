"""The canonical loader must remove every file-level difference between classes."""

from __future__ import annotations


import numpy as np
from PIL import Image, ImageCms

from screenpad.data.io import load_image


def _png_with_metadata(tmp_path, rotate_tag: int):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0112] = rotate_tag  # Orientation
    path = tmp_path / "x.png"
    img.save(path, exif=exif, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    return path


def test_metadata_stripped(tmp_path):
    out = load_image(_png_with_metadata(tmp_path, 1))
    assert out.mode == "RGB"
    assert "icc_profile" not in out.info and "exif" not in out.info
    assert len(out.getexif()) == 0


def test_exif_orientation_applied(tmp_path):
    out = load_image(_png_with_metadata(tmp_path, 6))  # 6 = rotate 90 CW
    assert out.size == (20, 30)
