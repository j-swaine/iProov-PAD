"""
GPU augmentation for training batches.
--------------------------------------

Colab VMs have few CPU cores, so per-image OpenCV augmentation would starve the GPU. Every operation here runs batched on the GPU on float images in [0, 1], shape (B, 3, H, W). All of it is applied to both classes with the same probabilities: augmentation must never be a cue.

Order: flip -> small affine (rotation, shift, scale) -> rescale jitter -> blur -> colour (incl. gamma) -> noise -> greyscale.

- Rescale jitter: shrink by a log-uniform factor in [1, max_rescale] with an anti-aliased (low-pass filtered) resize, enlarge back with a randomly chosen interpolation, so how much resampling a crop has been through stops being informative.

Every shrink uses antialiased bilinear rather than area averaging: both low-pass before subsampling (so screen grids are not aliased into new moiré), and only the former runs at arbitrary ratios on Apple MPS as well as CUDA, so local and Colab runs behave the same.
- Colour: brightness, contrast, saturation and per-channel gains (white balance), mild.
- Greyscale: removes colour entirely for a fraction of images, so skin tone and colour casts cannot carry the decision.
- Photometric robustness (optional, off unless `p_blur`, `p_gamma` or `p_noise` > 0; added 2026-10-20 after the robustness test showed genuine faces rejected under sensor noise, blur and brightening): Gaussian blur (sigma uniform in [0, max_blur] px at crop resolution), gamma (log-uniform in [1/max_gamma, max_gamma]) and additive Gaussian noise (sigma uniform in [0, max_noise], [0, 1] units). Applied to both classes alike, like everything here, so noise cannot become a "genuine" cue an attacker could add.

Randomness: `sample_params` draws every per-image parameter from a numpy Generator, so a batch's augmentation is fixed by (seed, epoch, batch position) and reproducible whatever the device.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_UP_MODES = ("bilinear", "bicubic")


def sample_params(rng: np.random.Generator, n: int, a: dict) -> dict[str, np.ndarray]:
    """Per-image augmentation parameters for a batch of `n`, from the `train.augment` config.

    The optional photometric draws happen only when their probability is > 0, so configs without them consume the generator exactly as before (earlier runs stay reproducible).
    """
    u = lambda: rng.uniform(-1, 1, n)
    p = {
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
    if a.get("p_blur", 0) > 0:
        p["blur"] = rng.random(n) < a["p_blur"]
        p["blur_sigma"] = rng.uniform(0, a["max_blur"], n)
    if a.get("p_gamma", 0) > 0:
        p["gamma"] = rng.random(n) < a["p_gamma"]
        p["gamma_value"] = np.exp(u() * math.log(a["max_gamma"]))
    if a.get("p_noise", 0) > 0:
        p["noise"] = rng.random(n) < a["p_noise"]
        p["noise_sigma"] = rng.uniform(0, a["max_noise"], n)
        p["noise_seed"] = rng.integers(2**31, size=n)
    return p


def _blur(x: torch.Tensor, sigma: np.ndarray) -> torch.Tensor:
    """Separable Gaussian blur with a per-image sigma (px); sigma 0 leaves the image unchanged."""
    b, c, h, w = x.shape
    r = max(1, int(math.ceil(3 * float(sigma.max()))))
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    s = torch.as_tensor(np.maximum(sigma, 1e-6), device=x.device, dtype=torch.float32).view(b, 1)
    g = torch.exp(-0.5 * (k / s) ** 2)
    g = (g / g.sum(1, keepdim=True)).to(x.dtype).repeat_interleave(c, 0)  # (b * c, 2r + 1)
    y = F.pad(x.reshape(1, b * c, h, w), (r, r, r, r), mode="reflect")
    y = F.conv2d(y, g.view(b * c, 1, 1, -1), groups=b * c)
    y = F.conv2d(y, g.view(b * c, 1, -1, 1), groups=b * c)
    return y.view(b, c, h, w)


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

    if "blur" in p and p["blur"].any():
        x = torch.where(t(p["blur"]).bool().view(b, 1, 1, 1), _blur(x, np.where(p["blur"], p["blur_sigma"], 0.0)), x)

    c = t(p["colour"]).view(b, 1, 1, 1)
    y = x * t(p["brightness"]).view(b, 1, 1, 1)
    mean = _grey(y).mean((2, 3), keepdim=True)
    y = (y - mean) * t(p["contrast"]).view(b, 1, 1, 1) + mean
    g = _grey(y)
    y = (y - g) * t(p["saturation"]).view(b, 1, 1, 1) + g
    y = y * t(p["gain"]).view(b, 3, 1, 1)
    x = torch.where(c.bool(), y, x).clamp(0, 1)

    if "gamma" in p and p["gamma"].any():
        x = torch.where(t(p["gamma"]).bool().view(b, 1, 1, 1), x.clamp_min(1e-6) ** t(p["gamma_value"]).view(b, 1, 1, 1), x)

    if "noise" in p and p["noise"].any():
        # one generator per batch on the batch's device, seeded from the sampled parameters (reproducible per device)
        gen = torch.Generator(device=dev).manual_seed(int(p["noise_seed"][0]))
        z = torch.randn((b, 3, h, w), generator=gen, device=dev).to(dt)
        x = (x + z * t(np.where(p["noise"], p["noise_sigma"], 0.0)).view(b, 1, 1, 1)).clamp(0, 1)

    return torch.where(t(p["grey"]).bool().view(b, 1, 1, 1), _grey(x).expand(-1, 3, -1, -1), x)


def to_input(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a float batch to the model's input size: antialiased bilinear when shrinking, bicubic when enlarging."""
    if x.shape[-1] == size:
        return x
    if x.shape[-1] > size:
        return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
