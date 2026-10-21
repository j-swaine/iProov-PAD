"""Benign-degradation robustness: shapes and dtypes, determinism, JPEG and blur behaviour, table ordering, and BPCER arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screenpad.eval.robustness import DEGRADATIONS, apply_all, blur, condition_summary, jpeg, noise


def _img(size=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_degradation_preserves_shape_and_dtype(name):
    img = _img()
    out = DEGRADATIONS[name](img, 3)
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_degradation_is_deterministic(name):
    img = _img()
    assert np.array_equal(DEGRADATIONS[name](img, 3), DEGRADATIONS[name](img, 3))


def test_noise_depends_on_seed():
    img = _img()
    assert np.array_equal(noise(img, 5, seed=1), noise(img, 5, seed=1))
    assert not np.array_equal(noise(img, 5, seed=1), noise(img, 5, seed=2))


def test_lower_jpeg_quality_differs_more():
    img = _img()
    mad = {q: np.abs(jpeg(img, q).astype(int) - img.astype(int)).mean() for q in (30, 70, 95)}
    assert mad[30] > mad[70] > mad[95]


def test_blur_reduces_std():
    img = _img()
    assert blur(img, 2.0).std() < img.std()


def test_apply_all_ordering_and_length():
    images = np.stack([_img(seed=s) for s in range(3)])
    out, table = apply_all(images)
    n_cond = len(DEGRADATIONS) + 1
    assert len(table) == 3 * n_cond
    assert out.shape[0] == len(table)
    assert list(table.columns) == ["i", "condition"]
    for i in range(3):
        block = table.iloc[i * n_cond : (i + 1) * n_cond]
        assert (block["i"] == i).all()
        assert block["condition"].tolist() == ["original", *DEGRADATIONS]
    assert np.array_equal(out[0], images[0])
    assert np.array_equal(out[n_cond], images[1])


def test_condition_summary_reads_both_classes():
    df = pd.DataFrame({"condition": ["original"] * 4 + ["blur_2"] * 4, "label": [0, 0, 1, 1] * 2,
                       "score": [0.0, 0.5, 3.0, 4.0, 1.5, 2.0, 0.8, 4.0]})
    out = condition_summary(df, threshold=1.0).set_index("condition")
    assert out.loc["original", "bpcer"] == 0 and out.loc["original", "apcer"] == 0
    assert out.loc["blur_2", "bpcer"] == 1 and out.loc["blur_2", "apcer"] == 0.5
    assert out.loc["blur_2", "delta_apcer"] == 0.5 and out.loc["original", "auc"] == 1.0


def test_condition_summary_bonafide_only():
    df = pd.DataFrame({"condition": ["original", "original", "noise_5"], "label": [0, 0, 0], "score": [0.0, 2.0, 2.0]})
    out = condition_summary(df, threshold=1.0).set_index("condition")
    assert out.loc["noise_5", "delta_bpcer"] == 0.5 and np.isnan(out.loc["original", "apcer"])


def test_edge_band_flattens_border_only():
    from screenpad.eval.robustness import EDGE_BANDS, edge_band

    img = _img(40)
    out = edge_band(img, 0.1)  # 4 px band
    assert np.array_equal(out[4:36, 4:36], img[4:36, 4:36])
    border = np.concatenate([out[:4].reshape(-1, 3), out[36:].reshape(-1, 3), out[4:36, :4].reshape(-1, 3), out[4:36, 36:].reshape(-1, 3)])
    assert (border == border[0]).all()
    stacked, table = apply_all(np.stack([img]), EDGE_BANDS)
    assert table.condition.tolist() == ["original", *EDGE_BANDS] and len(stacked) == 1 + len(EDGE_BANDS)


def test_centre_zoom_keeps_size_and_centre():
    from screenpad.eval.robustness import centre_zoom

    img = np.zeros((40, 40, 3), np.uint8)
    img[16:24, 16:24] = 255  # bright centre square
    out = centre_zoom(img, 0.8)
    assert out.shape == img.shape and out[20, 20].min() > 200 and out[0, 0].max() < 30
