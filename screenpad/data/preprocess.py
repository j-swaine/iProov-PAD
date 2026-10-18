"""
Face-centred square crops.
--------------------------

Every image, bona fide or attack, training or inference, goes through the same steps:

1. `load_image` (orientation, sRGB, metadata stripped).
2. Detect faces with YuNet on a copy downscaled to `det_size` on its long side, keep the largest face, and map its box back to full resolution.
3. Take a square of side `k * max(face w, face h)` centred on the face box. If it runs off the image it is shifted back inside; if it is still larger than the image's short side it is shrunk to fit. Shifting or shrinking is recorded, never padded: padding would itself mark the image.
4. Resize to `size` x `size` (INTER_AREA when shrinking, INTER_CUBIC when enlarging).

With `align=True` the square is also rotated about the face-box centre so the eye line is level (roll from YuNet's eye landmarks), so every crop has the face upright in a standard position. The rotated square must lie inside the image, so the shift/shrink test uses its bounding box. Resampling is done once at the output scale: the source region is area-downsampled first when shrinking (warpAffine has no area filter, and linear interpolation would alias screen grids into moiré that was not in the photo), then rotated with cubic interpolation.

`jitter` (training only) perturbs the crop itself - extra rotation, centre shift and scale - so geometric augmentation costs no second resampling of an already-resized crop.

Cropping relative to the face makes the framing identical across classes. It does not equalise the source resolution: attack faces are ~3x more pixels wide than bona fide ones, so their crops are downsampled harder. `CropInfo.resample_ratio` records this per image so it can be analysed and countered.

If no face is found, the largest centred square is used and `face_found` is False, so inference always produces a crop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_PATH = Path(__file__).resolve().parent.parent / "assets" / "face_detection_yunet_2023mar.onnx"


@dataclass
class Detection:
    box: np.ndarray  # x, y, w, h at full resolution
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose tip, right mouth corner, left mouth corner
    score: float
    n_faces: int

    @property
    def roll_deg(self) -> float:
        """In-plane rotation from the eye line (0 = eyes level)."""
        (rx, ry), (lx, ly) = self.landmarks[0], self.landmarks[1]
        return float(np.degrees(np.arctan2(ly - ry, lx - rx)))


@dataclass
class CropInfo:
    face_found: bool
    n_faces: int
    det_score: float
    face_w: float  # detected face width in source pixels
    roll_deg: float
    side: float  # crop side in source pixels
    resample_ratio: float  # side / output size: >1 means downsampled
    shifted: bool
    shrunk: bool
    angle_deg: float = 0.0  # rotation applied to the crop (aligned roll plus any jitter)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CropJitter:
    """Perturbation of one crop: extra rotation (deg), centre shift (fraction of crop side), log-scale of the side."""

    angle: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    log_scale: float = 0.0

    @classmethod
    def sample(cls, rng: np.random.Generator, max_angle: float = 10.0, max_shift: float = 0.05, max_log_scale: float = 0.1) -> "CropJitter":
        return cls(*rng.uniform(-1, 1, 4) * [max_angle, max_shift, max_shift, max_log_scale])


class FaceCropper:
    def __init__(self, model_path: str | Path = YUNET_PATH, det_size: int = 640, score_threshold: float = 0.6, align: bool = False):
        self.det_size = det_size
        self.align = align
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (det_size, det_size), score_threshold)

    def detect(self, rgb: np.ndarray) -> Detection | None:
        """Largest face in an RGB uint8 image, or None."""
        h, w = rgb.shape[:2]
        s = min(1.0, self.det_size / max(h, w))
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(np.ascontiguousarray(small[:, :, ::-1]))  # YuNet expects BGR
        if faces is None or len(faces) == 0:
            return None
        f = faces[np.argmax(faces[:, 2] * faces[:, 3])] / np.r_[np.full(14, s), 1.0]
        return Detection(box=f[:4], landmarks=f[4:14].reshape(5, 2), score=float(f[14]), n_faces=len(faces))

    def crop(self, rgb: np.ndarray, det: Detection | None, k: float, size: int, jitter: CropJitter | None = None) -> tuple[np.ndarray, CropInfo]:
        """Square crop of side k * face size around `det` (or the centre if None), resized to `size`."""
        h, w = rgb.shape[:2]
        if det is not None:
            x, y, bw, bh = det.box
            cx, cy, side = x + bw / 2, y + bh / 2, k * max(bw, bh)
        else:
            cx, cy, side = w / 2, h / 2, float(min(h, w))
        if self.align or jitter is not None:
            return _rotated_crop(rgb, det, cx, cy, side, size, (det.roll_deg if det and self.align else 0.0), jitter or CropJitter())
        shrunk = side > min(h, w)
        side = min(side, float(min(h, w)))
        x0 = float(np.clip(cx - side / 2, 0, w - side))
        y0 = float(np.clip(cy - side / 2, 0, h - side))
        shifted = (abs(x0 - (cx - side / 2)) > 0.5 or abs(y0 - (cy - side / 2)) > 0.5) and not shrunk
        xi, yi, si = int(round(x0)), int(round(y0)), int(round(side))
        patch = rgb[yi : yi + si, xi : xi + si]
        interp = cv2.INTER_AREA if si > size else cv2.INTER_CUBIC
        out = cv2.resize(patch, (size, size), interpolation=interp)
        info = CropInfo(
            face_found=det is not None,
            n_faces=det.n_faces if det else 0,
            det_score=det.score if det else 0.0,
            face_w=float(det.box[2]) if det else 0.0,
            roll_deg=det.roll_deg if det else 0.0,
            side=float(si),
            resample_ratio=si / size,
            shifted=bool(shifted),
            shrunk=bool(shrunk),
        )
        return out, info

    def __call__(self, rgb: np.ndarray, k: float, size: int, jitter: CropJitter | None = None) -> tuple[np.ndarray, CropInfo]:
        return self.crop(rgb, self.detect(rgb), k, size, jitter)


def _rotated_crop(rgb: np.ndarray, det: Detection | None, cx: float, cy: float, side: float, size: int,
                  roll: float, j: CropJitter) -> tuple[np.ndarray, CropInfo]:
    """Square of `side` centred on (cx, cy), rotated by `roll` + jitter so the eye line is level, resampled once to `size`."""
    h, w = rgb.shape[:2]
    angle = roll + j.angle
    side *= float(np.exp(j.log_scale))
    cx, cy = cx + j.dx * side, cy + j.dy * side
    t = np.radians(angle)
    spread = abs(np.cos(t)) + abs(np.sin(t))  # bounding-box side of the rotated square, per unit side
    shrunk = side * spread > min(h, w)
    side = min(side, min(h, w) / spread)
    half = side * spread / 2
    nx, ny = float(np.clip(cx, half, w - half)), float(np.clip(cy, half, h - half))
    shifted = (abs(nx - cx) > 0.5 or abs(ny - cy) > 0.5) and not shrunk
    cx, cy = nx, ny
    # area-downsample the bounding region first, so the rotation below works at (about) the output scale
    ratio = side / size
    x0, y0 = int(np.floor(cx - half)), int(np.floor(cy - half))
    x1, y1 = min(w, int(np.ceil(cx + half)) + 1), min(h, int(np.ceil(cy + half)) + 1)
    region = rgb[y0:y1, x0:x1]
    f = 1.0
    if ratio > 1:
        f = 1 / ratio
        region = cv2.resize(region, (max(1, round(region.shape[1] * f)), max(1, round(region.shape[0] * f))), interpolation=cv2.INTER_AREA)
        f = region.shape[1] / (x1 - x0)
    # map output pixel centres onto the (scaled) region: rotate by `angle` about the crop centre
    s_out = side * f / size
    c, s_ = np.cos(t), np.sin(t)
    rcx, rcy = (cx - x0) * f - 0.5 + 0.5 * f, (cy - y0) * f - 0.5 + 0.5 * f
    o = (size - 1) / 2
    M = np.array([[s_out * c, -s_out * s_, rcx - s_out * (c * o - s_ * o)],
                  [s_out * s_, s_out * c, rcy - s_out * (s_ * o + c * o)]])
    out = cv2.warpAffine(region, M, (size, size), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    info = CropInfo(
        face_found=det is not None,
        n_faces=det.n_faces if det else 0,
        det_score=det.score if det else 0.0,
        face_w=float(det.box[2]) if det else 0.0,
        roll_deg=det.roll_deg if det else 0.0,
        side=float(side),
        resample_ratio=float(ratio),
        shifted=bool(shifted),
        shrunk=bool(shrunk),
        angle_deg=float(angle),
    )
    return out, info
