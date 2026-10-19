"""
Probe datasets: which crops go where.
-------------------------------------

- train: train-split bona fide (real FFHQ crop + camera-only v0 per face) vs synthetic attacks v0 and v1 per face: 1:1, every face in both classes, as one fixed draw of the training composition (screenpad/train/data.py).
- val: the same composition from val images; used only to choose the regularisation strength.
- dev: the 6 real attacks vs their 6 twins plus the 120 real val bona fide, the first measurement on real attacks.

Real attacks never enter train or val. Test rows are never loaded.

Asymmetric quality filter (training only): bona fide training crops are dropped when no face was found or the crop had to be shrunk to fit the image, so the bona fide class is clean; every synthetic attack is kept, so attack variety is preserved. Evaluation sets are never filtered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cache import cache_frame, load_cache


def load_sets(cache_dir: str | Path, crop: str, size: int, synth_tag: str = "train-val") -> tuple[np.ndarray, pd.DataFrame]:
    """All non-test crops for one crop variant: (images, table with split/label/kind/source/keep_train)."""
    real = load_cache(Path(cache_dir) / f"crops_{crop}_{size}_aligned.npz")
    syn = load_cache(Path(cache_dir) / f"synth_{crop}_{size}_{synth_tag}.npz")
    fr, fs = cache_frame(real), cache_frame(syn)
    fr["kind"] = np.where(fr.label == 1, "real_attack", "real_bonafide")
    keep_r = (fr.split != "test").to_numpy()
    images = np.concatenate([real["images"][keep_r], syn["images"]])
    fr["variant"] = 0
    table = pd.concat([fr[keep_r], fs], ignore_index=True)[["id", "split", "path", "label", "kind", "variant", "info_face_found", "info_shrunk", "info_resample_ratio"]]
    bad_bona = (table.label == 0) & (~table.info_face_found.astype(bool) | table.info_shrunk.astype(bool))
    table["keep_train"] = ~bad_bona
    return images, table


def set_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    t = table
    fixed = ((t.kind == "real_bonafide") | ((t.kind == "camera_only") & (t.variant == 0)) | ((t.kind == "synthetic_attack") & (t.variant <= 1)))
    train = ((t.split == "train") & t.keep_train & fixed).to_numpy()
    val = ((t.split == "val") & fixed).to_numpy()
    dev = ((table.split == "dev") | ((table.split == "val") & (table.kind == "real_bonafide"))).to_numpy()
    assert not (table.kind[train | val] == "real_attack").any(), "real attacks must not be used for fitting or selection"
    return {"train": train, "val": val, "dev": dev}
