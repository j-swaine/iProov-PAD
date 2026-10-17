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


def real_attacks_outside_test(splits: pd.DataFrame) -> pd.DataFrame:
    """Real attacks assigned to train or val (they must only ever be test)."""
    return splits[(splits["label"] == ATTACK) & (splits["split"] != "test")]


def twins_outside_test(splits: pd.DataFrame) -> pd.DataFrame:
    """Bona fide twins of real attacks that are not in test."""
    return splits[splits["paired"] & (splits["split"] != "test")]


CHECKS = {
    "ids_crossing_splits": ids_crossing_splits,
    "real_attacks_outside_test": real_attacks_outside_test,
    "twins_outside_test": twins_outside_test,
}


def run_checks(splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every check; returns {name: offending rows}."""
    return {name: fn(splits) for name, fn in CHECKS.items()}
