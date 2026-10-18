"""
Training-time augmentation.
---------------------------

`rescale_jitter` shrinks an image by a random factor and enlarges it back, so the amount of resampling an image has been through stops being informative. In the supplied data that amount separates the classes perfectly (attack crops are downsampled 3.8-11x, bona fide 1.7-2.7x), so it is applied to both classes, with the same distribution, at training time only.

Randomness comes from `sample_rng(seed, epoch, index)`, a generator derived from those three numbers alone, so an image gets the same augmentation in a given epoch whatever the DataLoader worker layout or batch order.
"""

from __future__ import annotations

import cv2
import numpy as np

_UP_INTERPOLATIONS = (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)


def sample_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Independent, reproducible generator for one sample in one epoch."""
    return np.random.default_rng([seed, epoch, index])


def rescale_jitter(img: np.ndarray, rng: np.random.Generator, max_factor: float = 4.0, p: float = 0.5) -> np.ndarray:
    """With probability `p`, downsample by a factor drawn log-uniformly from [1, max_factor] (INTER_AREA) and upsample back to the original size with a randomly chosen interpolation. Returns a new uint8 array of the same shape."""
    if rng.random() >= p or max_factor <= 1:
        return img.copy()
    f = float(np.exp(rng.uniform(0, np.log(max_factor))))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    up = _UP_INTERPOLATIONS[rng.integers(len(_UP_INTERPOLATIONS))]
    return cv2.resize(small, (w, h), interpolation=up)
