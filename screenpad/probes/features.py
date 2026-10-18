"""
Frozen feature extractors for linear probes.
--------------------------------------------

Each backbone maps a batch of uint8 RGB crops to one vector per image. Nothing is trained here.

- DINOv2 (Meta, Apache-2.0): self-supervised ViT; features keep local texture. Run at 224 (the 384 crop shrunk 1.7x) and at 392, the patch-14 multiple nearest the stored 384, so almost no resampling hides fine screen texture.
- CLIP (OpenAI, MIT): image-text ViT; features are more semantic.
- handcrafted: the EDA's pixel statistics plus a 32-bin radial log power spectrum; a transparent reference point.

ViT features are the CLS token concatenated with the mean of the patch tokens: CLS summarises the image, the patch mean keeps more of the evenly spread local texture a screen leaves.

A face-recognition backbone is planned as a contrast (it is trained to ignore capture conditions, so should probe badly), pending a model with a permissive licence that installs with the project's torch.
"""

from __future__ import annotations

import cv2
import numpy as np

BACKBONES = {
    "dinov2_s224": ("vit_small_patch14_dinov2.lvd142m", 224),
    "dinov2_b224": ("vit_base_patch14_dinov2.lvd142m", 224),
    "dinov2_b392": ("vit_base_patch14_dinov2.lvd142m", 392),
    "clip_b16": ("vit_base_patch16_clip_224.openai", 224),
}


def radial_spectrum(gray: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Log mean power in `n_bins` radial frequency bands (0 to Nyquist) of a windowed grayscale image."""
    n = gray.shape[0]
    g = gray.astype(np.float64) / 255.0
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - n / 2, xx - n / 2) / n
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    return np.array([np.log10(power[(r >= lo) & (r < hi)].mean() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])])


def handcrafted(images: np.ndarray) -> np.ndarray:
    from ..eda.stats import pixel_stats

    rows = []
    for im in images:
        s = pixel_stats(im)
        spec = radial_spectrum(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
        rows.append(np.r_[list(s.values()), spec])
    return np.asarray(rows, np.float32)


class TimmBackbone:
    def __init__(self, name: str, device):
        import timm
        import torch

        model_id, self.size = BACKBONES[name]
        self.torch = torch
        self.device = device
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=self.size).eval().to(device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)

    def __call__(self, images: np.ndarray, batch: int = 32) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(images), batch):
            b = images[i : i + batch]
            if b.shape[1] != self.size:
                interp = cv2.INTER_AREA if b.shape[1] > self.size else cv2.INTER_CUBIC
                b = np.stack([cv2.resize(im, (self.size, self.size), interpolation=interp) for im in b])
            x = torch.from_numpy(b).to(self.device).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                tokens = self.model.forward_features((x - self.mean) / self.std)
            feats = torch.cat([tokens[:, 0], tokens[:, self.n_prefix :].mean(1)], dim=1)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out)


def extract(name: str, images: np.ndarray, device) -> np.ndarray:
    if name == "handcrafted":
        return handcrafted(images)
    return TimmBackbone(name, device)(images)
