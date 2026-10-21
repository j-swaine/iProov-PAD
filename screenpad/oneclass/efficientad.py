"""
EfficientAD one-class member, through the published implementation (anomalib, Apache-2.0).
------------------------------------------------------------------------------------------

EfficientAD (Batzner et al., WACV 2024): a frozen teacher network, pretrained on ImageNet (the authors' released weights, downloaded by anomalib), and a small student plus an autoencoder trained only on normal images (here: train bona fide, the same fit set as PatchCore) to reproduce the teacher's features. Where an image differs from the bona fide domain, the student and autoencoder fail to match the teacher; the anomaly map is that discrepancy and the image score is its maximum. The teacher is the transfer-learning component; training the student is the method itself.

Everything methodological is anomalib's: model, losses (including the ImageNet penalty on Imagenette, which anomalib downloads), optimiser, schedule, the bona-fide-only map normalisation (quantiles of val bona fide maps). This module only:
- writes our crops as PNG folders in the layout anomalib's `Folder` datamodule expects (train: train bona fide; validation/normalisation: val bona fide originals + camera-only);
- trains with anomalib's `Engine` for a fixed number of steps (batch size 1, as the method requires);
- scores our evaluation sets with the trained torch model and writes scores.csv / summary.json in the same format as the other one-class members.

Needs the `anomaly` extra (anomalib pinned), installed only where this runs (Colab).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def export_folder(images: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(str(folder / f"{i:05d}.png"), np.ascontiguousarray(img[:, :, ::-1]))


def make_imagenette_stub(path: Path, n: int = 8) -> None:
    """Tiny stand-in for Imagenette so a local smoke test does not download 1.5 GB (never used for a real run)."""
    d = path / "imagenette2" / "train" / "n00000000"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(d / f"{i}.JPEG"), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


try:  # anomalib (and Lightning) are only installed with the `anomaly` extra
    from lightning.pytorch.callbacks import Callback as _Callback
except ImportError:  # pragma: no cover
    _Callback = object


class Progress(_Callback):
    """Writes {"state": "running", "epoch": step, "epochs": max_steps, ...} to the run's STATUS every `every` steps, so the orchestrator's poll shows progress (its "epoch k/n" line is the step count here). Module-level so Lightning can pickle it into its checkpoint."""

    def __init__(self, status_path: Path | None, max_steps: int, every: int):
        self.status_path, self.max_steps, self.every = status_path, max_steps, every

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import json
        import logging

        step = trainer.global_step
        if step % self.every == 0 or step == self.max_steps:
            el = time.time() - self.t0
            st = {"state": "running", "epoch": step, "epochs": self.max_steps, "steps_per_s": round(step / max(el, 1e-9), 2),
                  "elapsed_s": round(el), "eta_s": round((self.max_steps - step) * el / max(step, 1)), "loss": float(outputs["loss"])}
            if self.status_path is not None:
                self.status_path.write_text(json.dumps(st))
            logging.getLogger("screenpad").info("EfficientAD step %d/%d, %.1f steps/s, loss %.4f, eta %d min",
                                                step, self.max_steps, st["steps_per_s"], st["loss"], st["eta_s"] // 60)

    def __getstate__(self):  # keep checkpoints free of timing state
        return {"status_path": self.status_path, "max_steps": self.max_steps, "every": self.every}


def train(work: Path, fit_images: np.ndarray, norm_images: np.ndarray, image_size: int, max_steps: int, seed: int,
          imagenet_dir: Path, accelerator: str = "auto", num_workers: int = 4, log=None, status_path: Path | None = None,
          progress_every: int = 500):
    """Train EfficientAD-S on `fit_images` (bona fide); `norm_images` (bona fide) set its map normalisation. Returns (lightning model, timing)."""
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd
    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=True)
    export_folder(fit_images, work / "data" / "train_good")
    export_folder(norm_images, work / "data" / "val_good")
    dm = Folder(name="screenpad_bonafide", root=work / "data", normal_dir="train_good", normal_test_dir="val_good",
                train_batch_size=1, eval_batch_size=16, num_workers=num_workers, val_split_mode="same_as_test", seed=seed)
    model = EfficientAd(imagenet_dir=imagenet_dir, model_size="small", pre_processor=EfficientAd.configure_pre_processor((image_size, image_size)))
    # no progress bar: anomalib's rich bar cannot be pickled into Lightning's checkpoint; progress goes to STATUS and the log instead
    engine = Engine(default_root_dir=work / "anomalib", max_steps=max_steps, accelerator=accelerator, devices=1, logger=False,
                    enable_progress_bar=False, callbacks=[Progress(status_path, max_steps, progress_every)])
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    return model, {"train_s": time.time() - t0, "max_steps": max_steps, "steps_per_s": max_steps / (time.time() - t0)}


def score(model, images: np.ndarray, image_size: int, device, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(image scores, anomaly maps) from the trained torch model; inputs prepared exactly as anomalib's pre-processor does (resize, no normalisation)."""
    import torch
    import torch.nn.functional as F

    net = model.model.to(device).eval()
    scores, maps = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i : i + batch]).to(device).permute(0, 3, 1, 2).float().div_(255)
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", antialias=True, align_corners=False)
            out = net(x)
            scores.append(out.pred_score.flatten().float().cpu().numpy())
            maps.append(F.adaptive_avg_pool2d(out.anomaly_map.float(), 32).squeeze(1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(maps)
