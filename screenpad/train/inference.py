"""
Scoring with a trained run: load a run folder's weights and score uint8 crops, outside the training loop.

Uses the same input path as `Trainer.batch` without augmentation (resize to the model input, normalise), so scores match the run's own scores_*.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def run_config(run_dir: str | Path) -> dict:
    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text())["train"]


def load_run(run_dir: str | Path, device, ckpt: str = "last.pt"):
    """(model in eval mode on `device`, the run's train config)."""
    import torch

    from .model import build_model

    c = run_config(run_dir)
    model = build_model(c["backbone"], int(c["input_size"]), pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / "ckpt" / ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval(), c


def score_images(model, images: np.ndarray, input_size: int, device, batch_size: int = 64) -> np.ndarray:
    """P(attack) for uint8 (N, H, W, 3) crops."""
    import torch

    from . import augment
    from .model import normalisation

    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in normalisation(model))
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i : i + batch_size]).to(device).permute(0, 3, 1, 2).float().div_(255)
            x = (augment.to_input(x, input_size) - mean) / std
            out.append(torch.sigmoid(model(x).float().squeeze(1)).cpu().numpy())
    return np.concatenate(out)
