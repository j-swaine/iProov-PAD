"""
Benign-degradation robustness check: do genuine faces get rejected more often after ordinary capture variation?
-----------------------------------------------------------------------------------------------------------------

A security-first threshold is set so that very few attacks pass. That leaves little margin for bona fide captures: real users' photos vary in compression, resolution, lighting, slight blur and sensor noise. If those variations push genuine scores over the threshold, BPCER (the share of bona fide wrongly rejected) rises without any change in who the user is.

The check takes validation bona fide faces, applies each benign degradation in `DEGRADATIONS` to them, re-scores them, and compares BPCER at a fixed threshold against the untouched "original". Only bona fide images are degraded; attacks are not part of this test. Every degradation is deterministic (noise is seeded by the image index), so a change in score comes from the degradation and not from randomness.

Gamma follows the usual `out = in ** g` convention on [0, 1] values, so g < 1 brightens and g > 1 darkens. Brightness is a plain multiply, clipped to the valid range.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd


def _to_u8(x: np.ndarray) -> np.ndarray:
    """[0, 1] float array to uint8, clipped and rounded."""
    return np.clip(np.rint(np.clip(x, 0, 1) * 255), 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode and decode at the given quality (1-100). Input and output are uint8 RGB."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"JPEG encoding failed at quality {quality}")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    """Shrink by `factor` (area), then enlarge back to the original size (bilinear): a low-resolution capture."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness(img: np.ndarray, gain: float) -> np.ndarray:
    """Multiply intensities by `gain` in [0, 1] units, then clip."""
    return _to_u8(img.astype(np.float32) / 255 * gain)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Gamma curve `out = in ** g` on [0, 1] values: g < 1 brightens, g > 1 darkens."""
    return _to_u8((img.astype(np.float32) / 255) ** g)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation `sigma` (pixels)."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive Gaussian noise with standard deviation `sigma` in 0-255 units, from a seeded generator."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# Each entry takes (image, i) where i is the image index; only noise uses i (as its seed).
DEGRADATIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "jpeg_q70": lambda img, i: jpeg(img, 70),
    "jpeg_q50": lambda img, i: jpeg(img, 50),
    "jpeg_q30": lambda img, i: jpeg(img, 30),
    "down_2x": lambda img, i: downscale(img, 2),
    "down_3x": lambda img, i: downscale(img, 3),
    "down_4x": lambda img, i: downscale(img, 4),
    "dark_0.6": lambda img, i: brightness(img, 0.6),
    "bright_1.4": lambda img, i: brightness(img, 1.4),
    "gamma_0.7": lambda img, i: gamma(img, 0.7),
    "gamma_1.4": lambda img, i: gamma(img, 1.4),
    "blur_1": lambda img, i: blur(img, 1.0),
    "blur_2": lambda img, i: blur(img, 2.0),
    "noise_5": lambda img, i: noise(img, 5.0, seed=i),
    "noise_10": lambda img, i: noise(img, 10.0, seed=i),
}


def edge_band(img: np.ndarray, frac: float) -> np.ndarray:
    """Flatten the outer `frac` of the crop's side, on all four edges, to that band's mean colour.

    Tests whether a crop's border carries the decision: in about half the real attacks the loose crop runs a few percent past the displayed photo onto the screen surround (docs/report/review_round1.md, F9), and the simulator renders thin dark border slivers too.
    """
    h, w = img.shape[:2]
    b = max(1, round(frac * min(h, w)))
    mask = np.ones((h, w), bool)
    mask[b : h - b, b : w - b] = False
    out = img.copy()
    out[mask] = np.rint(img[mask].reshape(-1, img.shape[2]).mean(0)).astype(np.uint8)
    return out


def centre_zoom(img: np.ndarray, keep: float) -> np.ndarray:
    """Keep the central `keep` fraction of the side and enlarge it back to full size (cubic): removes the border without inserting anything."""
    h, w = img.shape[:2]
    kh, kw = round(h * keep), round(w * keep)
    y0, x0 = (h - kh) // 2, (w - kw) // 2
    return cv2.resize(img[y0 : y0 + kh, x0 : x0 + kw], (w, h), interpolation=cv2.INTER_CUBIC)


# Border test (not a degradation): kept separate so the robustness summary stays comparable across runs.
# Flattening removes the border's content but also inserts a flat band; the centre zooms remove it without inserting anything.
EDGE_BANDS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "edge_5pct": lambda img, i: edge_band(img, 0.05),
    "edge_10pct": lambda img, i: edge_band(img, 0.10),
    "edge_15pct": lambda img, i: edge_band(img, 0.15),
    "zoom_90pct": lambda img, i: centre_zoom(img, 0.90),
    "zoom_80pct": lambda img, i: centre_zoom(img, 0.80),
}


def apply_all(images: np.ndarray, conditions: dict | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Original plus every degradation for each image, image-major: image 0's conditions in order, then image 1's, and so on.

    Returns the stacked (N * (len(DEGRADATIONS) + 1), S, S, 3) uint8 array and a table with columns `i` and `condition`, row-aligned with it.
    """
    conditions = DEGRADATIONS if conditions is None else conditions
    names = ["original", *conditions]
    out = np.empty((len(images) * len(names), *images.shape[1:]), dtype=np.uint8)  # filled in place: no list-then-stack copy
    for i, img in enumerate(images):
        for j, name in enumerate(names):
            out[i * len(names) + j] = img if name == "original" else conditions[name](img, i)
    rows = [(i, name) for i in range(len(images)) for name in names]
    return out, pd.DataFrame(rows, columns=["i", "condition"])


def condition_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per condition, both classes: counts, BPCER and APCER at `threshold` (exact 95% intervals), AUC, median score per class, and the change versus "original".

    `df` needs columns `condition`, `label` (0 bona fide, 1 attack) and `score`. Either class may be absent (its fields are nan). A degradation that only lowers every score would cut BPCER and raise APCER, so the two are always read together.
    """
    from sklearn.metrics import roc_auc_score

    from .report_metrics import rates_at_threshold

    rows = []
    for cond, g in df.groupby("condition", sort=False):
        y, s = g.label.to_numpy(), g.score.to_numpy()
        r = rates_at_threshold(y, s, threshold)
        rows.append({"condition": cond, **r, "auc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan"),
                     "median_bonafide": float(np.median(s[y == 0])) if (y == 0).any() else float("nan"),
                     "median_attack": float(np.median(s[y == 1])) if (y == 1).any() else float("nan")})
    out = pd.DataFrame(rows)
    if "original" not in set(out.condition):
        raise ValueError("df has no 'original' condition")
    base = out.set_index("condition").loc["original"]
    out["delta_bpcer"] = out.bpcer - base.bpcer
    out["delta_apcer"] = out.apcer - base.apcer
    return out
