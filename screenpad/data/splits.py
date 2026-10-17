"""
Train/val/test splits.
----------------------

Splits are made over image IDs, not files, so an ID (a person, and every capture of them) sits in exactly one split.

Test set: every real attack, its bona fide twin, and `test_extra_bonafide` further bona fide IDs drawn at random. Real attacks never appear in train or val: the model is trained on synthetic attacks generated from train bona fide images, and val attacks are synthetic too, generated from val bona fide images. The twins make the test set a controlled comparison: for 23 IDs the same face appears once live and once recaptured, so a model can only separate them on the capture, not the content.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..constants import ATTACK


def make_splits(manifest: pd.DataFrame, cfg: Mapping, seed: int = 42) -> pd.DataFrame:
    """Return `manifest` with a `split` column added, assigned per ID.

    Parameters
    ----------
    manifest : pd.DataFrame
        Output of `build_manifest`; needs `id` and `label`.
    cfg : Mapping
        The `split` config section: `test_extra_bonafide` (int) and `val_frac` (float).
    seed : int
        Seeds the bona fide shuffle, so the split is reproducible.
    """
    rng = np.random.default_rng(seed)
    attack_ids = set(manifest.loc[manifest["label"] == ATTACK, "id"])
    free_ids = np.array(sorted(set(manifest["id"]) - attack_ids))
    rng.shuffle(free_ids)

    n_extra = int(cfg["test_extra_bonafide"])
    if n_extra > len(free_ids):
        raise ValueError(f"test_extra_bonafide={n_extra} but only {len(free_ids)} unpaired IDs exist")
    test_ids = attack_ids | set(free_ids[:n_extra])
    rest = free_ids[n_extra:]
    n_val = int(round(float(cfg["val_frac"]) * len(rest)))
    val_ids = set(rest[:n_val])

    out = manifest.copy()
    out["split"] = np.where(
        out["id"].isin(test_ids), "test", np.where(out["id"].isin(val_ids), "val", "train")
    )
    return out


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Counts of images per split and class, plus the number of paired IDs per split."""
    tab = pd.crosstab(splits["split"], splits["label_name"])
    tab["paired_ids"] = splits[splits["paired"]].groupby("split")["id"].nunique()
    return tab.fillna(0).astype(int)
