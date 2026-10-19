"""Simulator: reproducible, shape/dtype-correct, and it produces a screen grid at the period its parameters imply."""

from __future__ import annotations

import numpy as np
import yaml

from screenpad.config import CONFIG_DIR
from screenpad.data.preprocess import CropJitter, Detection, FaceCropper
from screenpad.eda.pairs import grid_peak
from screenpad.synth.recapture import RecaptureParams, camera_only, recapture

RANGES = yaml.safe_load(open(CONFIG_DIR / "synth.yaml"))["synth"]["ranges"]
SRC = np.full((96, 96, 3), 128, np.uint8)


def _params(seed: int) -> RecaptureParams:
    return RecaptureParams.sample(np.random.default_rng(seed), RANGES)


def test_same_seed_same_output():
    a = recapture(SRC, _params(1), np.random.default_rng(1))
    b = recapture(SRC, _params(1), np.random.default_rng(1))
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.ndim == 3


def test_output_size_follows_camera_scale():
    p = _params(2)
    out = recapture(SRC, p, np.random.default_rng(2))
    assert abs(out.shape[1] - round(96 * p.cam_per_src)) <= 1


def test_grid_period_matches_parameters():
    p = _params(3)
    p.layout, p.cam_per_src, p.screen_per_src, p.tilt, p.rotation, p.blur, p.glare, p.dust = "rgb_stripe", 4.0, 1.0, 0.0, 0.0, 0.3, 0.0, 0
    out = recapture(SRC, p, np.random.default_rng(3))  # flat grey source: any structure comes from the screen
    peak = grid_peak(out, (out.shape[1] // 2, out.shape[0] // 2), n=256)
    assert abs(peak["period_px"] - 4.0) < 0.3 and peak["excess_log10"] > 5


def test_camera_only_has_no_grid():
    out = camera_only(SRC, _params(4).camera, 4.0, np.random.default_rng(4))
    assert grid_peak(out, (out.shape[1] // 2, out.shape[0] // 2), n=256)["excess_log10"] < 3.5  # noise maxima alone reach ~2.5; a screen grid gives ~7


def test_aligned_crop_levels_the_eyes():
    img = np.zeros((400, 400, 3), np.uint8)
    t = np.radians(15)
    e1, e2 = np.array([200 - 60 * np.cos(t), 200 - 60 * np.sin(t)]), np.array([200 + 60 * np.cos(t), 200 + 60 * np.sin(t)])
    for e in (e1, e2):
        img[int(e[1]) - 3 : int(e[1]) + 4, int(e[0]) - 3 : int(e[0]) + 4] = 255
    lm = np.zeros((5, 2))
    lm[0], lm[1] = e1, e2
    det = Detection(box=np.array([120.0, 120.0, 160.0, 160.0]), landmarks=lm, score=1.0, n_faces=1)
    out, info = FaceCropper(align=True).crop(img, det, 1.0, 160)
    ys, xs = np.nonzero(out[..., 0] > 128)
    left, right = ys[xs < 80].mean(), ys[xs >= 80].mean()
    assert abs(left - right) < 1.0 and abs(info.angle_deg - 15) < 1e-6
    jittered, _ = FaceCropper(align=True).crop(img, det, 1.0, 160, CropJitter(angle=5.0))
    assert not np.array_equal(jittered, out)


def test_shrunk_rotated_crop_keeps_content():
    """A crop shrunk to fit puts its edge exactly on the image border; float error there must not wrap the slice."""
    yy, xx = np.mgrid[0:3048, 0:3048]
    img = np.repeat(((xx + yy) * 255 // 6096).astype(np.uint8)[..., None], 3, axis=2)  # diagonal ramp
    lm = np.zeros((5, 2))
    lm[0], lm[1] = [1089.8947385787965, 1443.0667133331299], [1892.3083328247071, 1382.0108470916748]
    # exact values from a synthetic attack where this happened (its loose crop came out as one smeared row)
    det = Detection(box=np.array([683.8126939773559, 624.4001140594482, 1678.7322704315186, 2287.2534118652343]), landmarks=lm, score=1.0, n_faces=1)
    out, info = FaceCropper(align=True).crop(img, det, 1.25, 384)
    assert info.shrunk and out[:, 0, 0].std() > 20 and out[0, :, 0].std() > 20  # varies down and across, not one smeared row
