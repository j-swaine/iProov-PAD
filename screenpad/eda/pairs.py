"""
Twin-pair measurements: what recapture does to an image, measured on (bona fide, attack) pairs.

Used on the dev pairs to calibrate the simulator (notebooks/Recapture.ipynb), and on (source, synthetic) pairs to check it. Never on test pairs.
"""

from __future__ import annotations

import cv2
import numpy as np


def register(bona: np.ndarray, attack: np.ndarray, n_features: int = 8000) -> tuple[np.ndarray, int]:
    """Homography mapping attack pixel coordinates onto bona fide ones (SIFT + RANSAC), and the inlier count."""
    sift = cv2.SIFT_create(n_features)
    ga = cv2.resize(cv2.cvtColor(attack, cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(bona, cv2.COLOR_RGB2GRAY), None)
    m = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[x.queryIdx].pt for x in m]) * 2
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    H, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    return H, int(inliers.sum())


def warp_to_bona(attack: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Attack resampled into the bona fide frame: warped at 2x, then area-downsampled, to limit aliasing."""
    w, h = size
    S = np.diag([2.0, 2.0, 1.0])
    big = cv2.warpPerspective(attack, S @ H, (2 * w, 2 * h), flags=cv2.INTER_LINEAR)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def tone_curves(bona: np.ndarray, attack_reg: np.ndarray, margin: int = 100, bin_width: int = 16, min_count: int = 200) -> np.ndarray:
    """Per-channel median attack value for each bona fide value bin (3, 256 // bin_width); NaN where a bin is too sparse.

    Both images are blurred slightly first, so small registration errors do not dominate.
    """
    b = cv2.GaussianBlur(bona, (0, 0), 2).astype(float)[margin:-margin, margin:-margin]
    a = cv2.GaussianBlur(attack_reg, (0, 0), 2).astype(float)[margin:-margin, margin:-margin]
    out = np.full((3, 256 // bin_width), np.nan)
    for c in range(3):
        x, y = b[..., c].ravel(), a[..., c].ravel()
        for k, lo in enumerate(range(0, 256, bin_width)):
            sel = (x >= lo) & (x < lo + bin_width)
            if sel.sum() >= min_count:
                out[c, k] = np.median(y[sel])
    return out


def grid_peak(rgb: np.ndarray, centre: tuple[int, int], n: int = 512) -> dict:
    """Strongest periodic component in an n x n native-resolution patch: frequency (cycles/px), period (px) and log10 excess over the median power.

    A screen's pixel grid photographed close up appears as a sharp peak; at the dev pairs' scale its period is 3-7 camera pixels.
    """
    cx, cy = centre
    h, w = rgb.shape[:2]
    n = min(n, h, w) // 2 * 2
    x0, y0 = int(np.clip(cx - n // 2, 0, w - n)), int(np.clip(cy - n // 2, 0, h - n))
    g = cv2.cvtColor(rgb[y0 : y0 + n, x0 : x0 + n], cv2.COLOR_RGB2GRAY).astype(float)
    g = (g - cv2.GaussianBlur(g, (0, 0), 3)) * np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices(P.shape) - n // 2
    r = np.hypot(yy, xx)
    P[r < 20] = 0
    k = np.unravel_index(np.argmax(P), P.shape)
    fy, fx = (k[0] - n // 2) / n, (k[1] - n // 2) / n
    f = float(np.hypot(fx, fy))
    return {"fx": fx, "fy": fy, "period_px": 1 / f, "excess_log10": float(np.log10(P[k] / np.median(P[r > 20])))}
