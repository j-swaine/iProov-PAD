"""
Face-centred square crops.
--------------------------

Every image, bona fide or attack, training or inference, goes through the same steps:

1. `load_image` (orientation, sRGB, metadata stripped).
2. Detect faces with YuNet on a copy downscaled to `det_size` on its long side, keep the largest face, and map its box back to full resolution.
3. Take a square of side `k * max(face w, face h)` centred on the face box. If it runs off the image it is shifted back inside; if it is still larger than the image's short side it is shrunk to fit. Shifting or shrinking is recorded, never padded: padding would itself mark the image.
4. Resize to `size` x `size` (INTER_AREA when shrinking, INTER_CUBIC when enlarging).

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

    def to_dict(self) -> dict:
        return asdict(self)


class FaceCropper:
    def __init__(self, model_path: str | Path = YUNET_PATH, det_size: int = 640, score_threshold: float = 0.6):
        self.det_size = det_size
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

    @staticmethod
    def crop(rgb: np.ndarray, det: Detection | None, k: float, size: int) -> tuple[np.ndarray, CropInfo]:
        """Square crop of side k * face size around `det` (or the centre if None), resized to `size`."""
        h, w = rgb.shape[:2]
        if det is not None:
            x, y, bw, bh = det.box
            cx, cy, side = x + bw / 2, y + bh / 2, k * max(bw, bh)
        else:
            cx, cy, side = w / 2, h / 2, float(min(h, w))
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

    def __call__(self, rgb: np.ndarray, k: float, size: int) -> tuple[np.ndarray, CropInfo]:
        return self.crop(rgb, self.detect(rgb), k, size)
