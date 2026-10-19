"""
GPU augmentation for training batches.
--------------------------------------

Colab VMs have few CPU cores, so per-image OpenCV augmentation would starve the GPU. Every operation here runs batched on the GPU on float images in [0, 1], shape (B, 3, H, W). All of it is applied to both classes with the same probabilities: augmentation must never be a cue.

Order: flip -> small affine (rotation, shift, scale) -> rescale jitter -> colour -> greyscale.

- Rescale jitter: shrink by a log-uniform factor in [1, max_rescale] with an anti-aliased (low-pass filtered) resize, enlarge back with a randomly chosen interpolation, as `screenpad.data.augment.rescale_jitter`, so how much resampling a crop has been through stops being informative.

Every shrink uses antialiased bilinear rather than area averaging: both low-pass before subsampling (so screen grids are not aliased into new moiré), and only the former runs at arbitrary ratios on Apple MPS as well as CUDA, so local and Colab runs behave the same.
- Colour: brightness, contrast, saturation and per-channel gains (white balance), mild.
- Greyscale: removes colour entirely for a fraction of images, so skin tone and colour casts cannot carry the decision.

Randomness: `sample_params` draws every per-image parameter from a numpy Generator, so a batch's augmentation is fixed by (seed, epoch, batch position) and reproducible whatever the device.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_UP_MODES = ("bilinear", "bicubic")


def sample_params(rng: np.random.Generator, n: int, a: dict) -> dict[str, np.ndarray]:
    """Per-image augmentation parameters for a batch of `n`, from the `train.augment` config."""
    u = lambda: rng.uniform(-1, 1, n)
    return {
        "flip": rng.random(n) < a["p_flip"],
        "affine": rng.random(n) < a["p_affine"],
        "angle": u() * a["max_rotate"],
        "dx": u() * a["max_shift"],
        "dy": u() * a["max_shift"],
        "log_scale": u() * a["max_log_scale"],
        "rescale": rng.random(n) < a["p_rescale"],
        "rescale_factor": np.exp(rng.uniform(0, math.log(a["max_rescale"]), n)),
        "up_mode": rng.integers(len(_UP_MODES), size=n),
        "colour": rng.random(n) < a["p_colour"],
        "brightness": 1 + u() * a["brightness"],
        "contrast": 1 + u() * a["contrast"],
        "saturation": 1 + u() * a["saturation"],
        "gain": 1 + rng.uniform(-1, 1, (n, 3)) * a["channel_gain"],
        "grey": rng.random(n) < a["p_grey"],
    }


def _grey(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def apply(x: torch.Tensor, p: dict[str, np.ndarray]) -> torch.Tensor:
    """Augment a float batch in [0, 1] with parameters from `sample_params`. Returns a new tensor."""
    dev, dt = x.device, x.dtype
    t = lambda v: torch.as_tensor(v, device=dev, dtype=dt)
    b, _, h, w = x.shape
    x = torch.where(t(p["flip"]).bool().view(b, 1, 1, 1), x.flip(-1), x)

    if p["affine"].any():
        ang = np.radians(np.where(p["affine"], p["angle"], 0.0))
        s = np.exp(np.where(p["affine"], -p["log_scale"], 0.0))  # grid scale < 1 zooms in
        dx, dy = np.where(p["affine"], p["dx"], 0.0) * 2, np.where(p["affine"], p["dy"], 0.0) * 2
        theta = np.stack([np.stack([s * np.cos(ang), -s * np.sin(ang), dx], 1), np.stack([s * np.sin(ang), s * np.cos(ang), dy], 1)], 1)
        grid = F.affine_grid(t(theta), (b, 3, h, w), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    if p["rescale"].any():
        out = []
        for i in range(b):
            xi = x[i : i + 1]
            if p["rescale"][i]:
                f = float(p["rescale_factor"][i])
                small = F.interpolate(xi, size=(max(1, round(h / f)), max(1, round(w / f))), mode="bilinear", antialias=True, align_corners=False)
                xi = F.interpolate(small, size=(h, w), mode=_UP_MODES[int(p["up_mode"][i])], align_corners=False)
            out.append(xi)
        x = torch.cat(out).clamp(0, 1)

    c = t(p["colour"]).view(b, 1, 1, 1)
    y = x * t(p["brightness"]).view(b, 1, 1, 1)
    mean = _grey(y).mean((2, 3), keepdim=True)
    y = (y - mean) * t(p["contrast"]).view(b, 1, 1, 1) + mean
    g = _grey(y)
    y = (y - g) * t(p["saturation"]).view(b, 1, 1, 1) + g
    y = y * t(p["gain"]).view(b, 3, 1, 1)
    x = torch.where(c.bool(), y, x).clamp(0, 1)

    return torch.where(t(p["grey"]).bool().view(b, 1, 1, 1), _grey(x).expand(-1, 3, -1, -1), x)


def to_input(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a float batch to the model's input size: antialiased bilinear when shrinking, bicubic when enlarging."""
    if x.shape[-1] == size:
        return x
    if x.shape[-1] > size:
        return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
