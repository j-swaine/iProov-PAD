"""
Synthetic training set.
-----------------------

For every bona fide source image in the requested splits, `generate` writes:

- `variants` synthetic attacks: `recapture` of the full source image, then the standard pipeline (`FaceCropper`: detect, align, crop). The crop is detected on the simulated photo, exactly as at inference.
- with probability `camera_only_frac`, one camera-only bona fide control: the same image through the camera stage alone, labelled bona fide.

Synthetic attacks keep their source's ID and split, so each training face appears both as itself and as its recapture, like the real twins. They are only ever made from train and val images (plus dev, kept separate, for calibration plots); test images are never touched.

Every output is a pure function of (seed, image ID, variant), via `np.random.default_rng([seed, STREAM, id, variant])`, so the set can be regenerated exactly and work can be spread over processes in any order.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.cache import _INFO_COLS
from ..data.io import load_image
from ..data.preprocess import FaceCropper
from .recapture import CameraParams, RecaptureParams, camera_only, recapture

STREAM = 7  # separates the simulator's random stream from every other use of the seed
CAMERA_ONLY = -1  # variant index used for the camera-only control

_worker: dict = {}


def _init(det_size: int, align: bool) -> None:
    import cv2

    cv2.setNumThreads(1)
    _worker["cropper"] = FaceCropper(det_size=det_size, align=align)


def simulate(path: str | Path, iid: str, variant: int, seed: int, ranges: dict) -> tuple[np.ndarray, dict, int, str]:
    """Full-resolution simulated photo for one (image, variant): (image, parameters, label, kind). Pure function of its arguments."""
    rng = np.random.default_rng([seed, STREAM, int(iid), variant + 1])
    src = np.asarray(load_image(path))
    if variant == CAMERA_ONLY:
        cam = CameraParams.sample(rng, ranges["camera"])
        cps = float(np.exp(rng.uniform(*np.log(ranges["cam_per_src"]))))
        params = {"cam_per_src": cps, **{f"camera_{k}": v for k, v in cam.__dict__.items()}}
        return camera_only(src, cam, cps, rng), params, 0, "camera_only"
    p = RecaptureParams.sample(rng, ranges)
    return recapture(src, p, rng), p.to_dict(), 1, "synthetic_attack"


def _one(job: tuple) -> dict:
    """Simulate and crop one (image, variant). Returns crops per crop name, labels and parameters."""
    path, iid, split, variant, seed, ranges, crops, size = job
    img, params, label, kind = simulate(path, iid, variant, seed, ranges)
    cropper = _worker["cropper"]
    det = cropper.detect(img)
    out = {"id": iid, "split": split, "variant": variant, "label": label, "kind": kind, "params": params, "crops": {}, "info": {}}
    for name, k in crops.items():
        out["crops"][name], info = cropper.crop(img, det, k, size)
        out["info"][name] = info.to_dict()
    return out


def generate(splits: pd.DataFrame, root: str | Path, ranges: dict, crops: dict[str, float], size: int, det_size: int,
             align: bool, seed: int, variants: int, camera_only_frac: float, workers: int) -> tuple[dict[str, dict], pd.DataFrame]:
    """Synthetic attacks and camera-only controls for the bona fide rows of `splits`. Returns ({crop name: cache dict}, parameter table)."""
    bona = splits[splits["label"] == 0].reset_index(drop=True)
    jobs = []
    for r in bona.itertuples():
        path = str(Path(root) / r.path)
        for v in range(variants):
            jobs.append((path, r.id, r.split, v, seed, ranges, crops, size))
        if np.random.default_rng([seed, STREAM, int(r.id), 0]).random() < camera_only_frac:
            jobs.append((path, r.id, r.split, CAMERA_ONLY, seed, ranges, crops, size))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(det_size, align)) as ex:
        results = list(tqdm(ex.map(_one, jobs, chunksize=4), total=len(jobs), desc="synth", mininterval=10))

    meta = {
        "id": np.asarray([r["id"] for r in results], dtype=str),
        "split": np.asarray([r["split"] for r in results], dtype=str),
        "kind": np.asarray([r["kind"] for r in results], dtype=str),
        "path": np.asarray([f"synth/{r['kind']}/{r['id']}_{r['variant']}" for r in results], dtype=str),
        "variant": np.asarray([r["variant"] for r in results], np.int16),
        "label": np.asarray([r["label"] for r in results], np.int8),
    }
    caches = {}
    for name, k in crops.items():
        info = {f"info_{c}": np.asarray([r["info"][name][c] for r in results]) for c in _INFO_COLS}
        caches[name] = {"images": np.stack([r["crops"][name] for r in results]), **meta, **info,
                        "k": np.float32(k), "size": np.int32(size), "align": np.bool_(align), "seed": np.int64(seed)}
    params = pd.DataFrame([{"id": r["id"], "split": r["split"], "variant": r["variant"], "kind": r["kind"], **r["params"]} for r in results])
    return caches, params

