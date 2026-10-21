"""
Skin-tone screening of bona fide faces (ITA, Individual Typology Angle): is the false-reject rate similar across skin-tone bands?
---------------------------------------------------------------------------------------------------------------------------------

FFHQ bona fide faces carry no demographic labels, so skin tone is estimated from pixels. ITA = arctan((L* - 50) / b*) in degrees, computed in CIELAB on skin pixels (Chardon et al. 1991). Bands: very light > 55, light 41-55, intermediate 28-41, tan 10-28, brown -30-10, dark < -30.

Read-out (`bpcer_by_band`): BPCER (bona fide rejected, score >= threshold) per ITA band, with exact Clopper-Pearson 95% intervals.

Caveats, which matter for reading the numbers:
- ITA is a proxy for skin tone in these images, not ethnicity or race. It is driven by skin colour as captured, so it also moves with lighting, exposure and white balance.
- Per-band counts are often small, so the intervals are wide. This is a screening check for large gaps, not proof that the system is fair across skin tones.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from ..explain.core import face_mask
from .report_metrics import clopper_pearson

# (name, low, high): a value falls in a band when low < ITA <= high.
ITA_BANDS = [
    ("very light", 55.0, np.inf),
    ("light", 41.0, 55.0),
    ("intermediate", 28.0, 41.0),
    ("tan", 10.0, 28.0),
    ("brown", -30.0, 10.0),
    ("dark", -np.inf, -30.0),
]

CORE_SCALE = 0.6  # semi-axes of the central region, relative to the face ellipse
L_MIN, L_MAX = 15.0, 95.0  # L* limits for skin: below drops eyes/brows/hair, above drops specular highlights
B_RELIABLE = 5.0  # below this median b* (near-grey skin: strong colour cast, greyscale-ish or very desaturated image) ITA swings to +/-90 deg and means nothing


def skin_pixels(img: np.ndarray, face_frac: float) -> np.ndarray:
    """(N, 3) L*, a*, b* of skin pixels in an aligned (S, S, 3) uint8 RGB crop.

    Pixels are kept if they lie inside the face ellipse (`face_mask` > 0.9), inside the central region (both semi-axes scaled by 0.6, which avoids hair and background at the edge), and have 15 <= L* <= 95.
    """
    size = img.shape[0]
    outer = face_mask(size, face_frac)[..., 0] > 0.9
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2
    ax, ay = max(face_frac, 0.4) * size / 2 * CORE_SCALE, 0.8 * size / 2 * CORE_SCALE
    core = ((xx - c) / ax) ** 2 + ((yy - c) / ay) ** 2 <= 1
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    keep = outer & core & (lab[..., 0] >= L_MIN) & (lab[..., 0] <= L_MAX)
    return lab[keep].astype(np.float64)


def ita(img: np.ndarray, face_frac: float) -> float:
    """ITA in degrees from the median L* and median b* of the skin pixels.

    NaN ("unknown" band) if there are no skin pixels or the median b* is below B_RELIABLE: there the angle is dominated by division by a near-zero b* and swings to +/-90 deg, so the face is reported as unreliable rather than forced into an extreme band.
    """
    px = skin_pixels(img, face_frac)
    if len(px) == 0:
        return float("nan")
    L, b = float(np.median(px[:, 0])), float(np.median(px[:, 2]))
    if b < B_RELIABLE:
        return float("nan")
    return float(np.degrees(np.arctan((L - 50.0) / b)))


def ita_band(value: float) -> str:
    """Band name for an ITA value; 'unknown' for NaN."""
    if np.isnan(value):
        return "unknown"
    for name, low, high in ITA_BANDS:
        if low < value <= high:
            return name
    return "unknown"  # unreachable with the open-ended bands above, kept for safety


def _row(name: str, scores: pd.Series, threshold: float) -> dict:
    n = len(scores)
    k = int((scores >= threshold).sum())
    if n == 0:
        return {"band": name, "n": 0, "rejected": 0, "bpcer": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = clopper_pearson(k, n)
    return {"band": name, "n": n, "rejected": k, "bpcer": k / n, "ci_low": lo, "ci_high": hi}


def bpcer_by_band(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """BPCER per ITA band for bona fide rows (label == 0). `df` needs columns ita, label and score.

    One row per band in ITA_BANDS order, an 'unknown' row (no reliable ITA), then 'all'; the band rows and 'unknown' sum to 'all'.
    """
    bona = df[df["label"] == 0]
    bands = bona["ita"].map(ita_band)
    rows = [_row(name, bona.loc[bands == name, "score"], threshold) for name, _, _ in ITA_BANDS + [("unknown", 0, 0)]]
    rows.append(_row("all", bona["score"], threshold))
    return pd.DataFrame(rows, columns=["band", "n", "rejected", "bpcer", "ci_low", "ci_high"])
