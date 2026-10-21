"""
Explainability tests: where in the crop, and at which spatial scale, the evidence for "attack" lives.
-----------------------------------------------------------------------------------------------------

Each test perturbs the input in one controlled way and re-scores it, so the result is a number per condition rather than a picture to interpret.

1. Face vs surroundings (`face_context_variants`): an ellipse covering the detected face (its width from the face box, its height from the crop geometry), with a soft edge. "face only" replaces everything outside it with the image's mean colour; "context only" replaces the face itself. A flat fill removes the evidence without adding texture; the soft edge avoids creating a sharp new boundary.
2. Fine texture vs visible content (`frequency_variants`): Gaussian blur of increasing strength at crop resolution (384 px) removes progressively coarser detail; "texture only" keeps just the fine residual (image minus its sigma-2 blur, added to mid-grey). If attack scores collapse under mild blur while the face is still plainly visible, the model relies on detail the eye barely sees, as expected for screen artefacts.
3. Occlusion maps (`occlusion_grid`): a mean-colour square slides over a grid; each cell's value is the drop in attack logit when that cell is hidden (positive = that region supported "attack").
"""

from __future__ import annotations

import cv2
import numpy as np

BLURS = (0.5, 1.0, 2.0, 4.0)


def face_mask(size: int, face_frac: float, feather: float = 6.0) -> np.ndarray:
    """Soft ellipse (1 inside, 0 outside) centred in the crop: width = face box width, height from the crop's 1.25x-face framing (0.8 of the side)."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2
    ax, ay = max(face_frac, 0.4) * size / 2, 0.8 * size / 2
    inside = (((xx - c) / ax) ** 2 + ((yy - c) / ay) ** 2 <= 1).astype(np.float32)
    return cv2.GaussianBlur(inside, (0, 0), feather)[..., None]


def _fill(img: np.ndarray, keep: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mean = x.reshape(-1, 3).mean(0)
    return np.clip(x * keep + mean * (1 - keep), 0, 255).astype(np.uint8)


def face_context_variants(img: np.ndarray, face_frac: float) -> dict[str, np.ndarray]:
    m = face_mask(img.shape[0], face_frac)
    return {"original": img, "face_only": _fill(img, m), "context_only": _fill(img, 1 - m)}


def frequency_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    x = img.astype(np.float32)
    out = {"original": img}
    for s in BLURS:
        out[f"blur_{s:g}"] = np.clip(cv2.GaussianBlur(x, (0, 0), s), 0, 255).astype(np.uint8)
    out["texture_only"] = np.clip(x - cv2.GaussianBlur(x, (0, 0), 2.0) + 128, 0, 255).astype(np.uint8)
    return out


def occlusion_grid(img: np.ndarray, cells: int = 8) -> np.ndarray:
    """(cells*cells, H, W, 3): the image with one grid cell replaced by its mean colour, row-major."""
    size = img.shape[0]
    step = size // cells
    mean = img.reshape(-1, 3).mean(0).astype(np.uint8)
    out = np.repeat(img[None], cells * cells, 0)
    for k in range(cells * cells):
        r, c = divmod(k, cells)
        out[k, r * step : (r + 1) * step, c * step : (c + 1) * step] = mean
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
