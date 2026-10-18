"""Rescale jitter must be reproducible from (seed, epoch, index) alone and preserve shape/dtype."""

from __future__ import annotations

import numpy as np

from screenpad.data.augment import rescale_jitter, sample_rng

IMG = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)


def test_same_seed_epoch_index_same_output():
    a = rescale_jitter(IMG, sample_rng(42, 3, 17), p=1.0)
    b = rescale_jitter(IMG, sample_rng(42, 3, 17), p=1.0)
    assert np.array_equal(a, b)


def test_different_epoch_differs():
    outs = {rescale_jitter(IMG, sample_rng(42, e, 17), p=1.0).tobytes() for e in range(5)}
    assert len(outs) > 1


def test_shape_dtype_and_noop():
    out = rescale_jitter(IMG, sample_rng(0, 0, 0), p=1.0)
    assert out.shape == IMG.shape and out.dtype == np.uint8
    assert np.array_equal(rescale_jitter(IMG, sample_rng(0, 0, 0), p=0.0), IMG)
