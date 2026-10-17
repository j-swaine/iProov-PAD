"""
Dataset manifest.
-----------------

One row per image: its ID, label, path, and the file-level properties (size, colour profile, EXIF) that differ between the two classes in the supplied data. The manifest is what the splits, the EDA and the shortcut audit all read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from ..constants import CLASS_FOLDERS, LABEL_NAMES
from .io import list_images


def _file_props(path: Path) -> dict:
    """File-level properties read from the header only (pixels are not decoded)."""
    with Image.open(path) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        return {
            "width": im.width,
            "height": im.height,
            "mode": im.mode,
            "format": im.format,
            "has_icc": bool(icc),
            "has_exif": len(exif) > 0,
            "exif_make": exif.get(0x010F),  # Make
            "exif_model": exif.get(0x0110),  # Model
            "file_bytes": path.stat().st_size,
        }


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Scan `root/<class folder>/` and return the manifest, sorted by (label, id).

    The image ID is the filename stem. In the supplied data each attack is a recapture of the bona fide image with the same ID, so `id` is also the grouping key for splitting.
    """
    root = Path(root)
    rows = []
    for folder, label in CLASS_FOLDERS.items():
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"Expected class folder {d}")
        for p in list_images(d):
            rows.append(
                {
                    "id": p.stem,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "path": str(p.relative_to(root)),
                    **_file_props(p),
                }
            )
    df = pd.DataFrame(rows).sort_values(["label", "id"]).reset_index(drop=True)
    # Paired = this ID exists in both classes.
    counts = df.groupby("id")["label"].nunique()
    df["paired"] = df["id"].map(counts).eq(2)
    df["aspect"] = df["width"] / df["height"]
    return df
