"""`load_image` turns every input format a hidden test set might contain into 8-bit sRGB with the right orientation; `infer` lists every file it cannot score."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from screenpad.data.io import list_images, load_image
from screenpad.infer import unsupported_files


def _gradient(h=40, w=60) -> Image.Image:
    """RGB image whose top-left is dark and bottom-right bright, so orientation and range can be checked."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), np.full((h, w), 128)], -1).astype(np.uint8)
    return Image.fromarray(a)


@pytest.mark.parametrize("mode,ext", [("RGB", "png"), ("RGBA", "png"), ("L", "jpg"), ("P", "png"), ("CMYK", "jpg"), ("RGB", "webp"), ("RGB", "bmp"), ("RGB", "tif")])
def test_formats_load_as_8bit_rgb(tmp_path, mode, ext):
    p = tmp_path / f"img.{ext}"
    _gradient().convert(mode).save(p)
    a = np.asarray(load_image(p))
    assert a.shape == (40, 60, 3) and a.dtype == np.uint8
    assert a.max() > 200 and a.min() < 60  # range kept, not clipped or blank


def test_16bit_greyscale_is_rescaled_not_clipped(tmp_path):
    grey8 = np.asarray(_gradient().convert("L"))
    p = tmp_path / "sixteen.png"
    Image.fromarray(grey8.astype(np.uint16) * 257).save(p)  # the same picture at full 16-bit range
    a = np.asarray(load_image(p))
    assert a.shape == (40, 60, 3) and a.dtype == np.uint8
    assert np.abs(a[..., 0].astype(int) - grey8).max() <= 1  # matches the 8-bit picture; a plain convert("RGB") gives all-white


def test_exif_orientation_is_applied(tmp_path):
    img = _gradient()  # 40 x 60, landscape
    ex = Image.Exif()
    ex[0x0112] = 6  # "rotate 90 CW to display": pixels stored rotated
    p = tmp_path / "rotated.jpg"
    img.rotate(90, expand=True).save(p, quality=95, exif=ex)
    a = np.asarray(load_image(p))
    assert a.shape == (40, 60, 3)  # displayed orientation restored


def test_metadata_is_dropped(tmp_path):
    ex = Image.Exif()
    ex[0x010F] = "PhoneMaker"  # Make
    p = tmp_path / "meta.jpg"
    _gradient().save(p, exif=ex, icc_profile=b"")
    img = load_image(p)
    assert not img.info and not img.getexif()


def test_truncated_file_raises(tmp_path):
    b = io.BytesIO()
    _gradient(200, 200).save(b, "JPEG")
    p = tmp_path / "cut.jpg"
    p.write_bytes(b.getvalue()[: len(b.getvalue()) // 3])
    with pytest.raises(OSError):
        load_image(p)


def test_listing_and_unsupported(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ("a.png", "B.JPG", "sub/c.jpeg"):
        _gradient().save(tmp_path / n)
    for n in ("notes.txt", "data.csv", ".DS_Store"):
        (tmp_path / n).write_bytes(b"x")
    flat = list_images(tmp_path)
    deep = list_images(tmp_path, recursive=True)
    assert [p.name for p in flat] == ["B.JPG", "a.png"] and len(deep) == 3
    assert [p.name for p in unsupported_files(tmp_path, False, flat)] == ["data.csv", "notes.txt"]  # hidden files ignored
    assert [p.name for p in unsupported_files(tmp_path, True, deep)] == ["data.csv", "notes.txt"]


def test_heic_loads(tmp_path):
    img = _gradient()
    p = tmp_path / "iphone.HEIC"
    img.save(p, format="HEIF", quality=95)
    assert p in list_images(tmp_path)
    a = np.asarray(load_image(p)).astype(int)
    assert a.shape == (40, 60, 3) and np.abs(a - np.asarray(img).astype(int)).mean() < 6  # lossy, but the same picture


def test_heic_is_never_rotated_twice(tmp_path):
    """HEIF stores rotation as a container transform that libheif applies at decode; pillow-heif then reports EXIF orientation 1, so `load_image` (which also applies EXIF orientation) returns the decoded pixels unchanged."""
    ex = Image.Exif()
    ex[0x0112] = 6
    p = tmp_path / "rotated.heic"
    _gradient().rotate(90, expand=True).save(p, format="HEIF", quality=95, exif=ex)
    with Image.open(p) as raw:
        assert raw.getexif().get(0x0112, 1) == 1
        decoded = np.asarray(raw.convert("RGB"))
    assert np.array_equal(np.asarray(load_image(p)), decoded)
