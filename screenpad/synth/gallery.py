"""`screenpad synth-gallery`: contact sheets for looking at synthetic attacks next to real data.

Written to results/synth/gallery/:
- crops_XX.png: train identities, one per row: bona fide crop | camera-only control | synthetic attack v0 | v1. Crops are 384 px, shown 1:1, exactly what a model sees.
- real_dev.png: the 6 dev twins and their real attacks, same layout and scale, for comparison. Only dev pairs are shown; test attacks are never rendered.
- native.png: 256-px patches from the centre of the face at full camera resolution: real dev attacks (top) and synthetic attacks (bottom).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..data.cache import cache_frame, load_cache
from ..data.io import load_image
from ..data.preprocess import FaceCropper
from ..utils import ensure_dir
from .generate import simulate

ROWS_PER_SHEET = 8


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _save(rows: list[np.ndarray], path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def _centre_patch(rgb: np.ndarray, cropper: FaceCropper, n: int = 256) -> np.ndarray:
    det = cropper.detect(rgb)
    h, w = rgb.shape[:2]
    cx, cy = (det.box[0] + det.box[2] / 2, det.box[1] + det.box[3] * 0.4) if det else (w / 2, h / 2)
    x0, y0 = int(np.clip(cx - n / 2, 0, w - n)), int(np.clip(cy - n / 2, 0, h - n))
    return rgb[y0 : y0 + n, x0 : x0 + n]


def cmd_gallery(args) -> None:
    cfg = _make_cfg(args)
    log = _setup(cfg)
    out = ensure_dir(Path(cfg.output_dir) / "synth" / "gallery")
    size = int(cfg.preprocess.size)
    real = load_cache(Path(cfg.data.cache_dir) / f"crops_tight_{size}_aligned.npz")
    syn = load_cache(Path(cfg.data.cache_dir) / f"synth_tight_{size}_train-val.npz")
    fr, fs = cache_frame(real), cache_frame(syn)
    blank = np.full((size, size, 3), 235, np.uint8)

    def crop_of(frame, cache, mask):
        idx = np.flatnonzero(mask)
        return cache["images"][idx[0]] if len(idx) else blank

    ids = np.random.default_rng(int(cfg.seed)).permutation(sorted(fs.id[fs.split == "train"].unique()))[: args.n]
    rows = []
    for i in ids:
        cells = [_label(crop_of(fr, real, (fr.id == i) & (fr.label == 0)), f"{i} bona fide"),
                 _label(crop_of(fs, syn, (fs.id == i) & (fs.kind == "camera_only")), "camera-only"),
                 _label(crop_of(fs, syn, (fs.id == i) & (fs.kind == "synthetic_attack") & (fs.variant == 0)), "synthetic v0"),
                 _label(crop_of(fs, syn, (fs.id == i) & (fs.kind == "synthetic_attack") & (fs.variant == 1)), "synthetic v1")]
        rows.append(np.hstack(cells))
    for k in range(0, len(rows), ROWS_PER_SHEET):
        _save(rows[k : k + ROWS_PER_SHEET], out / f"crops_{k // ROWS_PER_SHEET:02d}.png")

    dev_ids = sorted(fr.id[fr.split == "dev"].unique())
    rows = [np.hstack([_label(crop_of(fr, real, (fr.id == i) & (fr.label == 0)), f"{i} bona fide"),
                       _label(crop_of(fr, real, (fr.id == i) & (fr.label == 1)), "real attack")]) for i in dev_ids]
    _save(rows, out / "real_dev.png")

    cropper = FaceCropper(det_size=int(cfg.preprocess.det_size))
    root = Path(cfg.data.root)
    real_native = [_label(_centre_patch(np.asarray(load_image(root / f"Screens/{i}.png")), cropper), f"{i} real") for i in dev_ids]
    syn_native = [_label(_centre_patch(simulate(root / f"Bonafide/{i}.png", i, v, int(cfg.seed), cfg.synth.ranges)[0], cropper), f"{i} synth v{v}")
                  for i, v in zip(ids[:6], [0, 1, 0, 1, 0, 1])]
    _save([np.hstack(real_native), np.hstack(syn_native)], out / "native.png")
    log.info("gallery -> %s (%d crop sheets, real_dev.png, native.png)", out, (len(ids) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET)
