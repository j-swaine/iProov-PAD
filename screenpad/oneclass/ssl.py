"""
Domain-adapted self-supervised features: continue DINOv2's self-distillation on bona fide faces only.
-----------------------------------------------------------------------------------------------------

DINO-style self-distillation (Caron et al. 2021; the image-level objective DINOv2 trains with): a student network sees several crops of a face (2 global at 224 px, `n_local` local at 98 px) and is trained to give the same output distribution as a teacher, which sees only the global crops. The teacher is an exponential moving average of the student. Both start from pretrained DINOv2-base; the projection head is new, so it is trained alone for `head_warmup` epochs before the backbone is unfrozen (a random head's gradients would otherwise scramble pretrained features). Teacher outputs are centred and sharpened to prevent collapse.

No labels, no attacks, no simulator: the features adapt to what bona fide faces look like at this crop and scale. PatchCore is then fitted on these features (screenpad oneclass --weights-run <this run>).

Augmentation, a deliberate departure from DINO's recipe. DINO trains invariance to Gaussian blur, strong colour jitter, greyscale and solarisation. Recapture artefacts are blur, tone and colour shifts, so invariance to them would blind a one-class detector to exactly what it must notice. Only random resized crops, horizontal flips and mild brightness/contrast jitter (±10%) are used.

Batched on the GPU: all crops are cut in one call with torchvision's `roi_align` (bilinear sampling, several samples per output pixel, so downscaled crops are averaged rather than aliased).

Memory: the student runs 2 + n_local crops per image through ViT-B with gradients; at batch 64 that exceeds an L4's 22 GB (smoke test, 2026-10-20), so `grad_ckpt` (activation checkpointing) is available.

Run folder (same contract as training runs): config.yaml, provenance.json, metrics.csv (per epoch: loss, teacher entropy, a collapse indicator, throughput), bench.json, ckpt/last.pt (backbone weights only, loadable by `oneclass.models.Dino(weights=...)`), STATUS (progress every few steps).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

DINO = "vit_base_patch14_dinov2.lvd142m"


def _head(in_dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256):
    import torch.nn as nn
    from torch.nn.utils.parametrizations import weight_norm

    mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
    last = weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
    last.parametrizations.weight.original0.data.fill_(1)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp, self.last = mlp, last

        def forward(self, x):
            return self.last(nn.functional.normalize(self.mlp(x), dim=-1))

    return Head()


def multicrop(x, rng: np.random.Generator, n_local: int, global_size: int = 224, local_size: int = 98,
              global_scale=(0.4, 1.0), local_scale=(0.08, 0.4), jitter: float = 0.1):
    """x: float (B, 3, H, W) in [0, 1] on the device. Returns [global1, global2, local1..localN], each (B, 3, s, s)."""
    import torch
    from torchvision.ops import roi_align

    b, _, h, w = x.shape
    out = []
    for k in range(2 + n_local):
        size, (lo, hi) = (global_size, global_scale) if k < 2 else (local_size, local_scale)
        area = rng.uniform(lo, hi, b) * h * w
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3), b))
        cw = np.clip(np.sqrt(area * ratio), 8, w)
        ch = np.clip(np.sqrt(area / ratio), 8, h)
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        boxes = torch.tensor(np.stack([np.arange(b), x0, y0, x0 + cw, y0 + ch], 1), dtype=x.dtype, device=x.device)
        c = roi_align(x, boxes, (size, size), spatial_scale=1.0, sampling_ratio=4, aligned=True)
        flip = torch.as_tensor(rng.random(b) < 0.5, device=x.device).view(b, 1, 1, 1)
        c = torch.where(flip, c.flip(-1), c)
        g = torch.as_tensor(1 + rng.uniform(-jitter, jitter, (b, 1, 1, 1)), dtype=x.dtype, device=x.device)
        m = c.mean((1, 2, 3), keepdim=True)
        cst = torch.as_tensor(1 + rng.uniform(-jitter, jitter, (b, 1, 1, 1)), dtype=x.dtype, device=x.device)
        out.append(((c * g - m) * cst + m).clamp(0, 1))
    return out


def train(images: np.ndarray, run_dir: Path, device, epochs: int, batch_size: int, seed: int, n_local: int = 6, out_dim: int = 8192,
          lr: float = 5e-5, head_lr: float = 5e-4, layer_decay: float = 0.75, head_warmup: int = 1, tau_s: float = 0.1, tau_t: float = 0.04,
          ema=(0.996, 1.0), center_m: float = 0.9, save: bool = True, grad_ckpt: bool = False, log=None) -> dict:
    import copy

    import pandas as pd
    import timm
    import torch
    import torch.nn.functional as F
    from timm.optim import create_optimizer_v2
    from timm.scheduler import CosineLRScheduler

    torch.manual_seed(seed)
    cuda = device.type == "cuda"
    backbone = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=224, dynamic_img_size=True)  # local crops resample the 224 grid
    cfg = backbone.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    if grad_ckpt:  # recompute block activations in the backward pass: several-fold less memory for ~30% more compute
        backbone.set_grad_checkpointing(True)
    head = _head(backbone.num_features, out_dim)
    student = torch.nn.ModuleDict({"backbone": backbone, "head": head}).to(device)
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    center = torch.zeros(1, out_dim, device=device)
    data = torch.from_numpy(images).to(device)  # uint8 (N, H, W, 3), moved once
    steps_per_epoch = len(images) // batch_size
    total = epochs * steps_per_epoch
    opt = create_optimizer_v2(student.backbone, opt="adamw", lr=lr, weight_decay=0.04, layer_decay=layer_decay)
    opt.add_param_group({"params": list(student.head.parameters()), "lr": head_lr, "lr_scale": head_lr / lr, "weight_decay": 0.04})
    sched = CosineLRScheduler(opt, t_initial=total, lr_min=lr * 0.01, warmup_t=steps_per_epoch, warmup_lr_init=lr * 0.01, t_in_epochs=False)
    status = run_dir / "STATUS"
    hist, step, t_start = [], 0, time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def fwd(net, crops):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            return [net["head"](net["backbone"]((c - mean) / std)).float() for c in crops]

    for epoch in range(epochs):
        frozen = epoch < head_warmup
        for p in student.backbone.parameters():
            p.requires_grad_(not frozen)
        rng = np.random.default_rng([seed, 17, epoch])
        order = rng.permutation(len(images))
        t0, losses, ents = time.time(), [], []
        student.train()
        for s in range(steps_per_epoch):
            idx = torch.as_tensor(order[s * batch_size : (s + 1) * batch_size], device=device)
            x = data[idx].permute(0, 3, 1, 2).float().div_(255)
            crops = multicrop(x, rng, n_local)
            with torch.no_grad():
                t_out = fwd(teacher, crops[:2])
            s_out = fwd(student, crops)
            t_prob = [F.softmax((t - center) / tau_t, dim=-1) for t in t_out]
            loss, n = 0.0, 0
            for i, tp in enumerate(t_prob):
                for j, so in enumerate(s_out):
                    if i != j:
                        loss = loss + torch.sum(-tp * F.log_softmax(so / tau_s, dim=-1), dim=-1).mean()
                        n += 1
            loss = loss / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
            step += 1
            sched.step_update(step)
            m = ema[1] - (ema[1] - ema[0]) * (math.cos(math.pi * step / total) + 1) / 2
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                center.mul_(center_m).add_(torch.cat(t_out).mean(0, keepdim=True), alpha=1 - center_m)
                p_mean = torch.cat(t_prob).mean(0)
                ents.append(float(-(p_mean * p_mean.clamp_min(1e-12).log()).sum()))  # entropy of the mean teacher output: near 0 = collapse
            losses.append(loss.item())
            if step % 10 == 0 or step == total:
                el = time.time() - t_start
                status.write_text(json.dumps({"state": "running", "epoch": step, "epochs": total, "loss": round(loss.item(), 4),
                                              "elapsed_s": round(el), "eta_s": round((total - step) * el / step)}))
        dt = time.time() - t0
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "teacher_mean_entropy": float(np.mean(ents)),
               "max_entropy": math.log(out_dim), "backbone_frozen": frozen, "images_per_s": steps_per_epoch * batch_size / dt, "epoch_s": dt,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None}
        hist.append(row)
        pd.DataFrame(hist).to_csv(run_dir / "metrics.csv", index=False)
        if log is not None:
            log.info("ssl epoch %d/%d: loss %.4f, teacher entropy %.2f / %.2f, %.0f img/s", epoch + 1, epochs, row["loss"],
                     row["teacher_mean_entropy"], row["max_entropy"], row["images_per_s"])
    if save:
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save(teacher["backbone"].state_dict(), run_dir / "ckpt" / "last.pt")  # the teacher (EMA) backbone, as in DINO
    return {"history": hist, "steps": total, "train_s": time.time() - t_start}
