"""
Per-image statistics for the EDA notebook (and reusable as hand-crafted baseline features).

All pixel statistics are computed on a crop as the model would see it (uint8 RGB, square), so they describe what a classifier could actually use, not properties of the raw files.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def _radial_index(n: int) -> np.ndarray:
    yy, xx = np.indices((n, n))
    return np.hypot(yy - n / 2, xx - n / 2)


def spectral_stats(gray: np.ndarray) -> dict:
    """Frequency-domain statistics of a square grayscale crop.

    hf_ratio: fraction of (non-DC) spectral power above 0.25 cycles/pixel.
    peak_score: strongest isolated spectral peak, measured as the largest excess of log-power over the median log-power at the same radius (outside the lowest frequencies). Periodic structure such as moiré or a display's pixel grid shows up as such peaks; natural images have smooth, roughly isotropic spectra.
    """
    n = gray.shape[0]
    g = gray.astype(np.float64) / 255.0
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    r = _radial_index(n)
    freq = r / n
    valid = (freq > 0.02) & (freq <= 0.5)
    hf_ratio = power[freq > 0.25].sum() / power[valid].sum()
    logp = np.log10(power + 1e-12)
    ri = r.astype(int)
    med = np.zeros(ri.max() + 1)
    for k in np.unique(ri[valid]):
        med[k] = np.median(logp[ri == k])
    excess = np.where(valid & (freq > 0.05), logp - med[ri], -np.inf)
    return {"hf_ratio": float(hf_ratio), "peak_score": float(excess.max())}


def pixel_stats(rgb: np.ndarray) -> dict:
    """Colour, contrast, sharpness, noise, highlight and spectral statistics of one RGB uint8 crop."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float64)
    residual = g - cv2.GaussianBlur(g, (0, 0), 1.0)  # float high-pass: a median of integer residuals is quantised
    return {
        "brightness": float(L.mean()),  # mean CIELAB lightness, 0-100
        "contrast": float(L.std()),
        "colour_a": float(a.mean()),  # green(-) / red(+) cast
        "colour_b": float(b.mean()),  # blue(-) / yellow(+) cast
        "saturation": float(hsv[..., 1].mean() / 255),
        "sharpness": float(np.log10(cv2.Laplacian(g, cv2.CV_64F).var() + 1e-9)),  # log10 variance of Laplacian
        "noise_sigma": float(1.4826 * np.median(np.abs(residual - np.median(residual)))),  # robust (MAD) high-pass noise level
        "highlight_frac": float(((L > 95) & (hsv[..., 1] < 40)).mean()),  # bright, desaturated pixels: glare/specular
        "clipped_frac": float(((rgb.min(-1) <= 2) | (rgb.max(-1) >= 253)).mean()),
        **spectral_stats(gray),
    }


def pixel_stats_frame(images: np.ndarray) -> pd.DataFrame:
    """pixel_stats for every image in an (N, H, W, 3) uint8 array."""
    return pd.DataFrame([pixel_stats(im) for im in images])


def separability(values: pd.Series, labels: pd.Series) -> dict:
    """Effect size between classes (label 1 vs 0) for one feature.

    auc: Mann-Whitney AUC, P(attack value > bona fide value); 0.5 = no separation.
    cliffs_delta: 2*auc - 1, in [-1, 1].
    p_value: two-sided Mann-Whitney U test.
    """
    from scipy.stats import mannwhitneyu

    x1, x0 = values[labels == 1].dropna(), values[labels == 0].dropna()
    if x1.nunique() == 1 and x0.nunique() == 1 and x1.iloc[0] == x0.iloc[0]:  # both constant and equal: test undefined
        return {"auc": 0.5, "cliffs_delta": 0.0, "p_value": 1.0}
    u, p = mannwhitneyu(x1, x0, alternative="two-sided")
    auc = u / (len(x1) * len(x0))
    return {"auc": float(auc), "cliffs_delta": float(2 * auc - 1), "p_value": float(p)}
