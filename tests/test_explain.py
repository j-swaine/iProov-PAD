"""Explainability perturbations: shapes, the face mask, and that each perturbation changes only what it claims to."""

from __future__ import annotations

import numpy as np

from screenpad.explain.core import face_context_variants, face_mask, frequency_variants, occlusion_grid


def _img(size=64):
    return np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_face_mask_is_centred_soft_ellipse():
    m = face_mask(96, 0.6)[..., 0]
    assert m[48, 48] > 0.99 and m[0, 0] < 0.01 and 0 < m[48, 48 + 30] < 1


def test_face_context_variants_split_the_image():
    img = _img()
    v = face_context_variants(img, 0.6)
    assert set(v) == {"original", "face_only", "context_only"}
    c = 32
    assert np.abs(v["face_only"][c, c].astype(int) - img[c, c]).max() <= 1  # face kept
    assert np.abs(v["context_only"][0, 0].astype(int) - img[0, 0]).max() <= 1  # corner kept


def test_frequency_variants_and_occlusion():
    img = _img()
    v = frequency_variants(img)
    assert all(x.shape == img.shape and x.dtype == np.uint8 for x in v.values())
    assert v["blur_4"].std() < v["blur_0.5"].std() < img.std()
    occ = occlusion_grid(img, 4)
    assert occ.shape == (16, 64, 64, 3)
    assert (occ[5] != img).any(axis=-1).sum() <= 16 * 16  # only one 16x16 cell changed
