"""Shrink-test read-out: percentile of resized-crop scores within the factor-1 scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screenpad.eval.shortcut import FACTORS, shrink_metrics


def _scores(shift):
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, 50)
    rows = [(str(i), v, min(1.0, b + shift * (f != 1.0))) for i, b in enumerate(base) for v, f in enumerate(FACTORS)]
    return pd.DataFrame(rows, columns=["id", "variant", "score"])


def test_no_effect_stays_near_50():
    m = shrink_metrics(_scores(0.0))
    assert set(m) == {f"shrink_pct@{f:g}" for f in FACTORS if f != 1.0}
    assert all(45 <= v <= 55 for v in m.values())


def test_shift_climbs():
    assert all(v > 90 for v in shrink_metrics(_scores(0.6)).values())
