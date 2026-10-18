"""
Preprocessed crop cache.
------------------------

One `.npz` per crop variant holding every image already through `load_image` + `FaceCropper`, as uint8 (N, size, size, 3), with its id, label, split, source path and crop info. Training reads this instead of the 1.9 GB of raw PNGs, and every machine sees the same pixels: colour-profile conversion and resampling can differ slightly between Pillow/OpenCV builds, so the cache, not a re-run of preprocessing, is the reference input.

`verify_cache` recomputes every crop from the raw files and requires exact equality (np.array_equal), so the cache is provably what the preprocessing code produces on the machine that built it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io import load_image
from .preprocess import FaceCropper

_INFO_COLS = ("face_found", "n_faces", "det_score", "face_w", "roll_deg", "side", "resample_ratio", "shifted", "shrunk")


def build_crops(splits: pd.DataFrame, root: str | Path, crops: dict[str, float], size: int, det_size: int) -> dict[str, dict]:
    """Crop every image in `splits` once per named crop (one face detection per image). Returns {name: cache dict}."""
    cropper = FaceCropper(det_size=det_size)
    n = len(splits)
    images = {name: np.empty((n, size, size, 3), np.uint8) for name in crops}
    info = {name: {c: [] for c in _INFO_COLS} for name in crops}
    for i, path in enumerate(tqdm(splits["path"], desc="crop", mininterval=10)):
        rgb = np.asarray(load_image(Path(root) / path))
        det = cropper.detect(rgb)
        for name, k in crops.items():
            images[name][i], ci = cropper.crop(rgb, det, k, size)
            for c, v in ci.to_dict().items():
                info[name][c].append(v)
    meta = {c: np.asarray(splits[c].astype(str).tolist(), dtype=str) for c in ("id", "split", "path")}  # fixed-width unicode, loadable without pickle
    meta["label"] = splits["label"].to_numpy(np.int8)
    return {
        name: {"images": images[name], **meta, **{f"info_{c}": np.asarray(v) for c, v in info[name].items()},
               "k": np.float32(k), "size": np.int32(size)}
        for name, k in crops.items()
    }


def save_cache(cache: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def load_cache(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify_cache(path: str | Path, root: str | Path, det_size: int) -> int:
    """Recompute every crop from the raw files and require byte-exact equality with the saved cache. Returns images checked."""
    c = load_cache(path)
    k, size = float(c["k"]), int(c["size"])
    cropper = FaceCropper(det_size=det_size)
    bad = []
    for i, p in enumerate(tqdm(c["path"], desc="verify", mininterval=10)):
        crop, _ = cropper(np.asarray(load_image(Path(root) / p)), k, size)
        if not np.array_equal(crop, c["images"][i]):
            bad.append(str(c["id"][i]))
    if bad:
        raise ValueError(f"{path}: {len(bad)} crops differ from a fresh recompute, e.g. {bad[:5]}")
    return len(c["path"])


def cache_frame(cache: dict) -> pd.DataFrame:
    """Per-image table (everything except the pixels)."""
    return pd.DataFrame({k: v for k, v in cache.items() if np.ndim(v) == 1})
