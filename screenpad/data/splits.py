"""
Train/val/test splits.
----------------------

Splits are made over image IDs, not files, so an ID (a person, and every capture of them) sits in exactly one split.

Real attacks are split by ID into a small dev set (`dev_attacks` of them, with their twins), used for simulator calibration and model selection, and a test set touched once at the end: the remaining attacks, their twins, and `test_extra_bonafide` further bona fide IDs drawn at random. Real attacks never appear in train or val: the model is trained on synthetic attacks generated from train bona fide images, and val attacks are synthetic too, generated from val bona fide images. The twins make the test set a controlled comparison: for 23 IDs the same face appears once live and once recaptured, so a model can only separate them on the capture, not the content.
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
        The `split` config section: `dev_attacks` (int), `test_extra_bonafide` (int) and `val_frac` (float).
    seed : int
        Seeds the bona fide shuffle, so the split is reproducible.
    """
    rng = np.random.default_rng(seed)
    attack_ids = sorted(set(manifest.loc[manifest["label"] == ATTACK, "id"]))
    free_ids = np.array(sorted(set(manifest["id"]) - set(attack_ids)))
    rng.shuffle(free_ids)

    # Separate stream for the attack draw, so changing dev_attacks never moves bona fide IDs.
    n_dev = int(cfg["dev_attacks"])
    if not 0 <= n_dev < len(attack_ids):
        raise ValueError(f"dev_attacks={n_dev} must be in [0, {len(attack_ids)})")
    dev_ids = set(np.random.default_rng([seed, 1]).permutation(attack_ids)[:n_dev])

    n_extra = int(cfg["test_extra_bonafide"])
    if n_extra > len(free_ids):
        raise ValueError(f"test_extra_bonafide={n_extra} but only {len(free_ids)} unpaired IDs exist")
    test_ids = (set(attack_ids) - dev_ids) | set(free_ids[:n_extra])
    rest = free_ids[n_extra:]
    n_val = int(round(float(cfg["val_frac"]) * len(rest)))
    val_ids = set(rest[:n_val])

    out = manifest.copy()
    out["split"] = np.select(
        [out["id"].isin(dev_ids), out["id"].isin(test_ids), out["id"].isin(val_ids)],
        ["dev", "test", "val"],
        default="train",
    )
    return out


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Counts of images per split and class, plus the number of paired IDs per split."""
    tab = pd.crosstab(splits["split"], splits["label_name"])
    tab["paired_ids"] = splits[splits["paired"]].groupby("split")["id"].nunique()
    return tab.fillna(0).astype(int)
