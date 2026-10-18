"""
Off-the-shelf baseline: Silent-Face-Anti-Spoofing (Minivision, Apache-2.0), used exactly as published.

Pipeline, as in the upstream `test.py`: their Caffe face detector on the BGR image -> two crops around the detected box at 2.7x and 4.0x context, each resized to 80x80 -> MiniFASNetV2 and MiniFASNetV1SE -> softmax summed over the two models. Class 1 is "real"; the attack score is 1 - (summed real probability / 2). Inputs are raw 0-255 floats (upstream's ToTensor does not rescale).

Context confound: upstream clamps the context scale so the crop fits inside the image. The bona fide FFHQ images only fit ~1.4x the face, so they never get the full 2.7x / 4.0x view, whereas attack photos usually do, and that extra context often contains the screen bezel. `context="matched"` caps each attack's scale at the scale its bona fide twin received, so both members of a pair are seen with the same context; comparing the two modes shows how much of the score comes from context alone.

The upstream detector is a Caffe model, and OpenCV 5 dropped Caffe support, so this baseline runs with OpenCV 4 layered over the project environment (`uv run --with 'opencv-python-headless>=4.10,<5' ...`). Only this evaluation-only baseline uses it; the main pipeline and the crop caches stay on the project's OpenCV.

Weights and detector are not bundled: they are downloaded from the upstream repository at a pinned commit and checked against SHA-256 hashes.
"""

from __future__ import annotations

import math
import urllib.request
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"
BASE_URL = f"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/{COMMIT}/resources"
FILES = {
    "deploy.prototxt": ("detection_model", "9fe2f141b4baee039ed9442da2833e216af40a6ff3e639e7b39258812bcda808"),
    "Widerface-RetinaFace.caffemodel": ("detection_model", "d08338a2c207df16a9c566f767fea67fb43ba6fff76ce11e938fe3fabefb9402"),
    "2.7_80x80_MiniFASNetV2.pth": ("anti_spoof_models", "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"),
    "4_0_0_80x80_MiniFASNetV1SE.pth": ("anti_spoof_models", "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990"),
}
MODELS = (("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7), ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0))
CACHE_DIR = Path.home() / ".cache" / "screenpad" / "minifasnet"


def fetch_weights(cache_dir: Path = CACHE_DIR) -> Path:
    """Download (once) and hash-check every upstream file. Returns the directory holding them."""
    from ..data.archive import sha256_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, (sub, sha) in FILES.items():
        path = cache_dir / name
        if not path.exists() or sha256_file(path) != sha:
            urllib.request.urlretrieve(f"{BASE_URL}/{sub}/{name}", path)
            if sha256_file(path) != sha:
                path.unlink()
                raise ValueError(f"{name}: SHA-256 mismatch after download")
    return cache_dir


def _effective_box(src_w: int, src_h: int, bbox, scale: float) -> tuple[int, int, int, int, float]:
    """Upstream CropImage._get_new_box, also returning the scale actually used after clamping."""
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    x0, y0, x1, y1 = cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > src_w - 1:
        x0 -= x1 - src_w + 1
        x1 = src_w - 1
    if y1 > src_h - 1:
        y0 -= y1 - src_h + 1
        y1 = src_h - 1
    return int(x0), int(y0), int(x1), int(y1), scale


class MiniFASNetBaseline:
    def __init__(self, device: str = "cpu"):
        import torch

        from .minifasnet_arch import MiniFASNetV1SE, MiniFASNetV2

        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no Caffe importer (removed in 5.0), which the upstream face detector needs. "
                "Run this baseline with OpenCV 4 layered over the project env, leaving the project's OpenCV untouched:\n"
                "  uv run --with 'opencv-python-headless>=4.10,<5' screenpad baseline-minifasnet"
            )
        d = fetch_weights()
        self.torch = torch
        self.device = torch.device(device)
        self.detector = cv2.dnn.readNetFromCaffe(str(d / "deploy.prototxt"), str(d / "Widerface-RetinaFace.caffemodel"))
        arch = {"MiniFASNetV2": MiniFASNetV2, "MiniFASNetV1SE": MiniFASNetV1SE}
        self.models = []
        for fname, kind, scale in MODELS:
            net = arch[kind](conv6_kernel=(5, 5)).to(self.device)  # upstream get_kernel(80, 80)
            state = torch.load(d / fname, map_location=self.device)
            state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
            net.load_state_dict(state)
            self.models.append((net.eval(), scale))

    def detect(self, bgr: np.ndarray) -> list[int]:
        """Upstream Detection.get_bbox: most confident face, [x, y, w, h]."""
        h, w = bgr.shape[:2]
        img = bgr
        if w * h >= 192 * 192:
            ar = w / h
            img = cv2.resize(bgr, (int(192 * math.sqrt(ar)), int(192 / math.sqrt(ar))), interpolation=cv2.INTER_LINEAR)
        self.detector.setInput(cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123)), "data")
        out = self.detector.forward("detection_out").squeeze()
        i = np.argmax(out[:, 2])
        left, top, right, bottom = out[i, 3] * w, out[i, 4] * h, out[i, 5] * w, out[i, 6] * h
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]

    def score(self, bgr: np.ndarray, max_scales: tuple[float, float] | None = None) -> tuple[float, tuple[float, float]]:
        """Attack score for one BGR uint8 image, and the context scale each model actually used.

        `max_scales` optionally caps each model's context scale (used for context-matched evaluation).
        """
        bbox = self.detect(bgr)
        h, w = bgr.shape[:2]
        real, used = 0.0, []
        for j, (net, scale) in enumerate(self.models):
            s = scale if max_scales is None else min(scale, max_scales[j])
            x0, y0, x1, y1, eff = _effective_box(w, h, bbox, s)
            patch = cv2.resize(bgr[y0 : y1 + 1, x0 : x1 + 1], (80, 80))
            t = self.torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(net(t), dim=1)[0].cpu().numpy()
            real += float(p[1])
            used.append(eff)
        return 1.0 - real / len(self.models), tuple(used)
