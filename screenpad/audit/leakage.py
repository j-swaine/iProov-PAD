"""
Leakage checks on a split table (output of `make_splits`).

Each check returns the offending rows, so an empty frame means the check passed. They are run by the test suite and by `screenpad audit`.
"""

from __future__ import annotations

import pandas as pd

from ..constants import ATTACK


def ids_crossing_splits(splits: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ID appears in more than one split."""
    n = splits.groupby("id")["split"].nunique()
    return splits[splits["id"].isin(n[n > 1].index)]


_REAL_ATTACK_SPLITS = ("dev", "test")


def real_attacks_in_training(splits: pd.DataFrame) -> pd.DataFrame:
    """Real attacks assigned to train or val (they may only be dev or test)."""
    return splits[(splits["label"] == ATTACK) & ~splits["split"].isin(_REAL_ATTACK_SPLITS)]


def twins_in_training(splits: pd.DataFrame) -> pd.DataFrame:
    """Bona fide twins of real attacks assigned to train or val."""
    return splits[splits["paired"] & ~splits["split"].isin(_REAL_ATTACK_SPLITS)]


def dev_without_both_classes(splits: pd.DataFrame) -> pd.DataFrame:
    """Dev must hold only twin pairs: any dev ID missing either class is an error."""
    dev = splits[splits["split"] == "dev"]
    return dev[~dev["paired"]]


CHECKS = {
    "ids_crossing_splits": ids_crossing_splits,
    "real_attacks_in_training": real_attacks_in_training,
    "twins_in_training": twins_in_training,
    "dev_without_both_classes": dev_without_both_classes,
}


def run_checks(splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every check; returns {name: offending rows}."""
    return {name: fn(splits) for name, fn in CHECKS.items()}
