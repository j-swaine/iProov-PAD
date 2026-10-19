"""
Training data: which crops each epoch uses, and the fixed evaluation sets.
--------------------------------------------------------------------------

All crops (real aligned crops for every non-test split, plus the synthetic pool) are held in one uint8 array, which training moves to the GPU once.

Per epoch, for every train face:
- bona fide: its original crop + `camera_per_face` camera-only controls drawn from its pool;
- attack: `attacks_per_face` synthetic attacks drawn from its pool, without replacement.

So each face contributes equally to both classes and the classes are 1:1 (with the defaults, 2 + 2 per face). If a face's original crop failed the training quality filter (no face found, or crop shrunk to fit), a second camera-only control takes its place, keeping the balance.

`p_original` is the resample-ratio reweighting. Original crops are downsampled ~2.4x, camera-only controls and synthetic attacks ~3-9x with matching distributions, so originals are the only training images below ~3x, all bona fide. With `p_original` < 1 a face's original is used in an epoch only with that probability, otherwise a second camera-only control takes its slot: the bona fide ratio distribution moves towards the attacks' (identical at 0) while the classes stay 1:1. The default 1.0 draws exactly as before. The draw and the shuffle come from `np.random.default_rng([seed, EPOCH_STREAM, epoch])`, so an epoch's composition and order are reproducible.

Evaluation sets are fixed and never filtered:
- val: val faces, original + camera-only v0 vs attack v0 and v1 (synthetic, 1:1);
- dev: the 6 real dev attacks vs their 6 twins plus the 120 real val bona fide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cache import cache_frame, load_cache

EPOCH_STREAM = 11


def load_pool(cache_dir: str | Path, crop: str, size: int, synth_tag: str = "train-val") -> tuple[np.ndarray, pd.DataFrame]:
    """(images, table) for every non-test real crop and every synthetic crop of one crop variant, plus the shrink-test crops if their cache exists."""
    from ..eval.shortcut import shrink_name

    real = load_cache(Path(cache_dir) / f"crops_{crop}_{size}_aligned.npz")
    syn = load_cache(Path(cache_dir) / f"synth_{crop}_{size}_{synth_tag}.npz")
    fr, fs = cache_frame(real), cache_frame(syn)
    shrink_path = Path(cache_dir) / shrink_name(crop, size)
    if shrink_path.exists():  # split "shrink": never trained on, scored as its own evaluation set
        sh = load_cache(shrink_path)
        syn = {"images": np.concatenate([syn["images"], sh["images"]])}
        fs = pd.concat([fs, cache_frame(sh)], ignore_index=True)
    fr["kind"] = np.where(fr.label == 1, "real_attack", "real_bonafide")
    fr["variant"] = 0
    keep = (fr.split != "test").to_numpy()
    images = np.concatenate([real["images"][keep], syn["images"]])
    cols = ["id", "split", "label", "kind", "variant", "info_face_found", "info_shrunk", "info_resample_ratio"]
    table = pd.concat([fr[keep][cols], fs[cols]], ignore_index=True)
    table["variant"] = table["variant"].astype(int)
    table["train_ok"] = ~((table.label == 0) & (~table.info_face_found.astype(bool) | table.info_shrunk.astype(bool)))
    return images, table


class EpochSampler:
    """Reproducible per-epoch draw of train indices (see module docstring)."""

    def __init__(self, table: pd.DataFrame, seed: int, camera_per_face: int = 1, attacks_per_face: int = 2, p_original: float = 1.0):
        self.seed, self.n_cam, self.n_att, self.p_orig = seed, camera_per_face, attacks_per_face, p_original
        tr = table[table.split == "train"]
        self.faces = []
        for iid, g in tr.groupby("id", sort=True):
            orig = g.index[(g.kind == "real_bonafide") & g.train_ok].tolist()
            cams = g.index[g.kind == "camera_only"].tolist()
            atts = g.index[g.kind == "synthetic_attack"].tolist()
            if len(atts) >= self.n_att and len(cams) >= self.n_cam + (0 if orig and self.p_orig >= 1 else 1):
                self.faces.append((orig, cams, atts))

    def __call__(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, EPOCH_STREAM, epoch])
        idx = []
        for orig, cams, atts in self.faces:
            use = orig if self.p_orig >= 1 or (orig and rng.random() < self.p_orig) else []  # no extra draw at the default, so earlier runs reproduce
            n_cam = self.n_cam + (0 if use else 1)
            idx += use + rng.choice(cams, n_cam, replace=False).tolist() + rng.choice(atts, self.n_att, replace=False).tolist()
        return rng.permutation(np.asarray(idx))


def eval_sets(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """val and dev (as in the module docstring), plus `shrink` (the resampling counterfactual) when present."""
    t = table
    val = (t.split == "val") & (
        (t.kind == "real_bonafide") | ((t.kind == "camera_only") & (t.variant == 0)) | ((t.kind == "synthetic_attack") & (t.variant <= 1))
    )
    dev = (t.split == "dev") | ((t.split == "val") & (t.kind == "real_bonafide"))
    assert not (t.kind[val] == "real_attack").any()
    sets = {"val": np.flatnonzero(val.to_numpy()), "dev": np.flatnonzero(dev.to_numpy())}
    if (t.split == "shrink").any():
        sets["shrink"] = np.flatnonzero((t.split == "shrink").to_numpy())
    return sets
