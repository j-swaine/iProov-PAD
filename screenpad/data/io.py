"""
Canonical image loading.
------------------------

Every image the package touches (training, evaluation, inference) goes through `load_image`, so all of them get the same decoding. This matters for PAD: the supplied attacks carry an iPhone EXIF block and a Display P3 colour profile while the bona fide images carry neither, so a loader that ignored those differences would hand the model a free shortcut.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from ..constants import IMAGE_EXTENSIONS

_SRGB = ImageCms.createProfile("sRGB")


def to_srgb(img: Image.Image) -> Image.Image:
    """Convert `img` to 8-bit sRGB using its embedded ICC profile if it has one, and drop the profile afterwards."""
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            img = ImageCms.profileToProfile(img.convert("RGB"), src, _SRGB, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError):
            # A corrupt profile shouldn't kill a batch; fall back to treating the pixels as sRGB.
            img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    img.info.pop("icc_profile", None)
    return img


def load_image(path: str | Path) -> Image.Image:
    """Open `path` as an orientation-corrected sRGB `PIL.Image` with no metadata attached."""
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        img = to_srgb(img)
    # Re-wrap the pixels so no EXIF/info from the source file survives.
    return Image.fromarray(np.asarray(img))


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Return every image file under `folder`, sorted, matched on extension (case-insensitive)."""
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
