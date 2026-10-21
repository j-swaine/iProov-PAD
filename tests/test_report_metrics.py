from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screenpad.eval.report_metrics import by_group, clopper_pearson, det_points, rates_at_threshold


def test_clopper_pearson_zero_of_17():
    lo, hi = clopper_pearson(0, 17)
    assert lo == 0
    assert hi == pytest.approx(0.195, abs=0.002)


def test_clopper_pearson_all_of_17():
    lo, hi = clopper_pearson(17, 17)
    assert hi == 1
    assert lo == pytest.approx(0.805, abs=0.002)


def test_clopper_pearson_symmetric():
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187, abs=0.002)
    assert hi == pytest.approx(0.813, abs=0.002)


def test_clopper_pearson_empty():
    lo, hi = clopper_pearson(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_rates_at_threshold_counts():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    score = [0.1, 0.2, 0.6, 0.9, 0.3, 0.7, 0.8, 0.95]
    r = rates_at_threshold(y, score, 0.5)
    assert r["n_attack"] == 4
    assert r["n_bonafide"] == 4
    assert r["attacks_missed"] == 1
    assert r["bonafide_rejected"] == 2
    assert r["apcer"] == pytest.approx(0.25)
    assert r["bpcer"] == pytest.approx(0.5)
    assert r["apcer_lo95"] < 0.25 < r["apcer_hi95"]
    assert r["bpcer_lo95"] < 0.5 < r["bpcer_hi95"]


def test_by_group_bonafide_shared():
    df = pd.DataFrame({
        "label": [0, 0, 0, 0, 1, 1, 1, 1, 1],
        "score": [0.1, 0.6, 0.9, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9],
        "phone": ["", "", "", "", "A", "A", "A", "B", "B"],
    })
    out = by_group(df, 0.5, "phone", bonafide_shared=True).set_index("group")
    assert set(out.index) == {"A", "B", "all"}
    assert out.loc["A", "attacks_missed"] == 2
    assert out.loc["A", "n_attack"] == 3
    assert out.loc["B", "attacks_missed"] == 0
    assert out.loc["B", "n_attack"] == 2
    assert out.loc["A", "bpcer"] == out.loc["B", "bpcer"] == out.loc["all", "bpcer"] == pytest.approx(0.5)
    assert out.loc["A", "n_bonafide"] == out.loc["B", "n_bonafide"] == 4


def test_by_group_attack_only_group_has_nan_bonafide():
    df = pd.DataFrame({
        "label": [0, 0, 1, 1],
        "score": [0.1, 0.6, 0.2, 0.9],
        "phone": ["", "", "A", "A"],
    })
    out = by_group(df, 0.5, "phone", bonafide_shared=False).set_index("group")
    assert out.loc["A", "n_bonafide"] == 0
    assert np.isnan(out.loc["A", "bpcer"])
    assert out.loc["all", "n_bonafide"] == 2


def test_det_points_monotonic():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    score = rng.uniform(0, 1, 100) + 0.3 * y
    d = det_points(y, score)
    assert list(d.columns) == ["threshold", "apcer", "bpcer"]
    # rows are ordered by threshold descending
    assert (np.diff(d.threshold[1:]) <= 0).all()
    assert (np.diff(d.apcer.to_numpy()) <= 1e-12).all()
    assert (np.diff(d.bpcer.to_numpy()) >= -1e-12).all()
