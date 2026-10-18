"""
Per-image face detection table and crop statistics, shared by the EDA figures and the cache builder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.io import load_image
from ..data.preprocess import FaceCropper


def detect_all(splits: pd.DataFrame, root: str | Path, det_size: int, crops: dict[str, float], size: int) -> pd.DataFrame:
    """One row per image: detection results plus, for each named crop, its side, resample ratio and shift/shrink flags."""
    cropper = FaceCropper(det_size=det_size)
    rows = []
    for _, r in tqdm(splits.iterrows(), total=len(splits), desc="detect", mininterval=10):
        rgb = np.asarray(load_image(Path(root) / r["path"]))
        det = cropper.detect(rgb)
        row = {"id": r["id"], "label": r["label"], "split": r["split"], "src_w": rgb.shape[1], "src_h": rgb.shape[0]}
        for name, k in crops.items():
            _, info = cropper.crop(rgb, det, k, size)
            d = info.to_dict()
            if name == next(iter(crops)):
                row.update({c: d[c] for c in ("face_found", "n_faces", "det_score", "face_w", "roll_deg")})
            row.update({f"{name}_{c}": d[c] for c in ("side", "resample_ratio", "shifted", "shrunk")})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["face_frac"] = df["face_w"] / df["src_w"]
    return df
