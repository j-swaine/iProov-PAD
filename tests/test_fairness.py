"""Skin-tone screening: ITA from pixels, band boundaries, and per-band BPCER with exact intervals."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from screenpad.eval.fairness import ITA_BANDS, bpcer_by_band, ita, ita_band, skin_pixels


def _uniform_crop(L: float, a: float, b: float, size: int = 128) -> np.ndarray:
    """uint8 RGB crop filled with the sRGB colour that corresponds to the given CIELAB value."""
    lab = np.zeros((size, size, 3), np.float32)
    lab[...] = (L, a, b)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def test_uniform_crop_gives_formula_ita():
    L, a, b = 65.0, 12.0, 18.0
    expected = np.degrees(np.arctan((L - 50) / b))
    assert abs(ita(_uniform_crop(L, a, b), 0.8) - expected) < 1.5


def test_lighter_skin_has_higher_ita():
    light = ita(_uniform_crop(75.0, 12.0, 20.0), 0.8)
    dark = ita(_uniform_crop(35.0, 12.0, 20.0), 0.8)
    assert light > dark


def test_skin_pixels_shape_and_range():
    px = skin_pixels(_uniform_crop(60.0, 10.0, 15.0), 0.8)
    assert px.ndim == 2 and px.shape[1] == 3 and len(px) > 0
    assert np.all(px[:, 0] >= 15) and np.all(px[:, 0] <= 95)


def test_ita_band_boundaries():
    assert [name for name, _, _ in ITA_BANDS] == ["very light", "light", "intermediate", "tan", "brown", "dark"]
    assert ita_band(60) == "very light"
    assert ita_band(55) == "light"
    assert ita_band(30) == "intermediate"
    assert ita_band(5) == "brown"
    assert ita_band(-40) == "dark"
    assert ita_band(float("nan")) == "unknown"


def test_bpcer_by_band_counts_are_exact():
    df = pd.DataFrame({
        "ita": [60, 60, 30, 30, -40, -40, 5, np.nan, 60],
        "label": [0, 0, 0, 0, 0, 0, 0, 0, 1],  # last row is an attack and must be ignored
        "score": [0.9, 0.2, 0.6, 0.7, 0.1, 0.4, 0.5, 0.8, 0.99],
    })
    out = bpcer_by_band(df, threshold=0.5).set_index("band")
    assert list(out.index) == ["very light", "light", "intermediate", "tan", "brown", "dark", "unknown", "all"]
    assert (out.loc["unknown", ["n", "rejected"]] == [1, 1]).all()
    assert (out.loc["very light", ["n", "rejected"]] == [2, 1]).all()
    assert (out.loc["light", ["n", "rejected"]] == [0, 0]).all()
    assert (out.loc["intermediate", ["n", "rejected"]] == [2, 2]).all()
    assert (out.loc["brown", ["n", "rejected"]] == [1, 1]).all()
    assert (out.loc["dark", ["n", "rejected"]] == [2, 0]).all()
    assert out.loc["all", "n"] == 8 and out.loc["all", "rejected"] == 5
    assert np.isclose(out.loc["all", "bpcer"], 5 / 8)
    assert out.drop("all").n.sum() == out.loc["all", "n"]
    assert out.loc["intermediate", "ci_high"] == 1.0  # k == n
    assert out.loc["dark", "ci_low"] == 0.0  # k == 0
    assert np.isnan(out.loc["tan", "bpcer"]) and out.loc["tan", "n"] == 0


def test_bpcer_all_row_equals_overall_bpcer():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"ita": rng.uniform(-60, 80, 200), "label": rng.integers(0, 2, 200), "score": rng.uniform(0, 1, 200)})
    out = bpcer_by_band(df, threshold=0.7)
    bona = df[df.label == 0]
    assert np.isclose(out.set_index("band").loc["all", "bpcer"], (bona.score >= 0.7).mean())
    assert out.set_index("band").loc["all", "n"] == len(bona)


def test_near_grey_skin_is_unknown_not_extreme():
    import cv2

    from screenpad.eval.fairness import ita

    lab = np.array([[[70.0, 2.0, 1.0]]], np.float32)  # b* below the reliability floor
    rgb = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).round().astype(np.uint8)
    assert np.isnan(ita(np.tile(rgb, (128, 128, 1)), 0.6))
