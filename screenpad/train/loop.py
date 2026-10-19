"""
Fine-tuning loop, evaluation and the batch-size probe.
------------------------------------------------------

One run = one backbone, one crop variant, a fixed number of epochs. Per epoch: the `EpochSampler` draw (1:1 classes, every face in both), GPU augmentation, AdamW with layer-wise learning-rate decay, linear warm-up then cosine decay (per step), gradient clipping, bf16 autocast on CUDA. After every epoch the model is scored on the fixed val (synthetic) and dev (real) sets.

Checkpoints: `last.pt` (the default choice: fixed budget) and `best.pt` (highest synthetic val AUC), weights only. A separate resume file holds optimizer/scheduler state so a restarted VM can continue; it is kept out of the uploaded run folder.

Shortcut checks run on every evaluation (see `shortcut_checks`), and the resampling counterfactual (`eval.shortcut`) when its crops are in the pool.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..eval.metrics import bpcer_at_apcer, eer
from ..eval.shortcut import shrink_metrics
from . import augment

AUG_STREAM = 13


class Trainer:
    def __init__(self, tcfg: dict, images: np.ndarray, table: pd.DataFrame, device, seed: int, log):
        from .data import EpochSampler, eval_sets
        from .model import build_model, normalisation

        self.c, self.device, self.seed, self.log, self.table = tcfg, device, seed, log, table
        torch.manual_seed(seed)
        self.cuda = device.type == "cuda"
        self.amp = bool(tcfg["amp"]) and self.cuda
        self.images = torch.from_numpy(images).to(device)  # uint8 (N, H, W, 3), moved once
        self.labels = torch.as_tensor(table.label.to_numpy(), dtype=torch.float32, device=device)
        self.sampler = EpochSampler(table, seed, int(tcfg["camera_per_face"]), int(tcfg["attacks_per_face"]), float(tcfg.get("p_original", 1.0)))
        self.sets = eval_sets(table)
        self.model = build_model(tcfg["backbone"], int(tcfg["input_size"])).to(device)
        if tcfg.get("channels_last") and tcfg["backbone"].startswith("convnext"):
            self.model = self.model.to(memory_format=torch.channels_last)
        mean, std = normalisation(self.model)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.forward = torch.compile(self.model) if tcfg.get("compile") else self.model

    def batch(self, idx: np.ndarray, aug_rng: np.random.Generator | None) -> torch.Tensor:
        x = self.images[torch.as_tensor(idx, device=self.device)].permute(0, 3, 1, 2).float().div_(255)
        if aug_rng is not None:
            x = augment.apply(x, augment.sample_params(aug_rng, len(idx), self.c["augment"]))
        x = augment.to_input(x, int(self.c["input_size"]))
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if self.c.get("channels_last") and self.c["backbone"].startswith("convnext") else x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            return self.forward(x).float().squeeze(1)

    @torch.no_grad()
    def score(self, idx: np.ndarray, batch_size: int) -> np.ndarray:
        self.model.eval()
        out = [torch.sigmoid(self.logits(self.batch(idx[i : i + batch_size], None))).cpu().numpy() for i in range(0, len(idx), batch_size)]
        return np.concatenate(out)

    def fit(self, run_dir: Path, resume_path: Path, batch_size: int) -> dict:
        from timm.scheduler import CosineLRScheduler

        from .model import build_optimizer

        c = self.c
        epochs = int(c["epochs"])
        steps_per_epoch = len(self.sampler(0)) // batch_size
        lr = float(c["lr"]) * batch_size / 64
        opt = build_optimizer(self.model, lr, float(c["weight_decay"]), float(c["layer_decay"]))
        sched = CosineLRScheduler(opt, t_initial=epochs * steps_per_epoch, lr_min=lr * float(c["min_lr_ratio"]),
                                  warmup_t=int(float(c["warmup_epochs"]) * steps_per_epoch), warmup_lr_init=lr * 0.01, t_in_epochs=False)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        start, best_auc, history = 0, -1.0, []
        if resume_path.exists():
            st = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            start, best_auc, history = st["epoch"] + 1, st["best_auc"], st["history"]
            self.log.info("resumed from epoch %d", start)
        ckpt = run_dir / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        smooth = float(c.get("label_smoothing", 0.0))

        for epoch in range(start, epochs):
            self.model.train()
            order = self.sampler(epoch)
            t0, losses, seen = time.time(), [], 0
            if self.cuda:
                torch.cuda.reset_peak_memory_stats()
            for b in range(steps_per_epoch):
                idx = order[b * batch_size : (b + 1) * batch_size]
                x = self.batch(idx, np.random.default_rng([self.seed, AUG_STREAM, epoch, b]))
                y = self.labels[torch.as_tensor(idx, device=self.device)]
                y = y * (1 - smooth) + 0.5 * smooth
                loss = loss_fn(self.logits(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(c["grad_clip"]))
                opt.step()
                sched.step_update(epoch * steps_per_epoch + b + 1)
                losses.append(loss.item())
                seen += len(idx)
            train_time = time.time() - t0
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr_end": opt.param_groups[-1]["lr"],
                   "images_per_s": seen / train_time, "epoch_s": train_time,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if self.cuda else None}
            if (epoch + 1) % int(c["eval_every"]) == 0 or epoch == epochs - 1:
                ev = self.evaluate(batch_size * 2)
                row.update(ev["metrics"])
                if row["val_auc"] > best_auc:
                    best_auc = row["val_auc"]
                    torch.save(self.model.state_dict(), ckpt / "best.pt")
                    ev["scores"].to_csv(run_dir / "scores_best.csv", index=False)
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
            torch.save(self.model.state_dict(), ckpt / "last.pt")
            torch.save({"model": self.model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_auc": best_auc, "history": history}, resume_path)
            (run_dir / "STATUS").write_text(json.dumps({"state": "running", "epoch": epoch + 1, "epochs": epochs}))
            self.log.info("epoch %d/%d loss %.4f | val AUC %.3f | dev AUC %.3f | %.0f img/s, %.0f s",
                          epoch + 1, epochs, row["train_loss"], row.get("val_auc", float("nan")), row.get("dev_auc", float("nan")),
                          row["images_per_s"], train_time)
        ev = self.evaluate(batch_size * 2)
        ev["scores"].to_csv(run_dir / "scores_last.csv", index=False)
        return {"final": ev["metrics"], "best_val_auc": best_auc, "history": history}

    def evaluate(self, batch_size: int) -> dict:
        from sklearn.metrics import roc_auc_score

        t = self.table
        frames, m = [], {}
        for name, idx in self.sets.items():
            s = self.score(idx, batch_size)
            frames.append(t.iloc[idx][["id", "split", "label", "kind", "variant", "info_resample_ratio"]].assign(set=name, score=s))
            if name == "shrink":
                m.update(shrink_metrics(frames[-1]))
                continue
            y = t.label.to_numpy()[idx]
            m[f"{name}_auc"] = float(roc_auc_score(y, s))
            m[f"{name}_eer"] = eer(y, s)
            m[f"{name}_bpcer@apcer0"] = bpcer_at_apcer(y, s, 0.0)
        scores = pd.concat(frames, ignore_index=True)
        m.update(shortcut_checks(scores[scores.set == "val"]))
        m.update(bootstrap_auc(scores[scores.set == "dev"], self.seed))
        return {"metrics": m, "scores": scores}


def shortcut_checks(val: pd.DataFrame) -> dict:
    """Is the model using resampling history instead of the screen?

    ratio_spearman_bonafide / _attack: rank correlation between score and crop downsample factor within each class (near 0 is good).
    ratio_spearman_camera: the same within camera-only controls alone. The bona fide figure mixes originals (~2.4x) with camera-only (~3-9x), so it partly repeats camera_vs_original; this one does not.
    camera_vs_original_auc: AUC separating camera-only bona fide from original bona fide by score (0.5 is good: phone processing alone does not look like an attack).
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    out = {}
    for cls, name in ((0, "bonafide"), (1, "attack")):
        v = val[val.label == cls]
        out[f"ratio_spearman_{name}"] = float(spearmanr(v.score, v.info_resample_ratio).statistic) if len(v) > 2 else float("nan")
    cam = val[val.kind == "camera_only"]
    out["ratio_spearman_camera"] = float(spearmanr(cam.score, cam.info_resample_ratio).statistic) if len(cam) > 2 else float("nan")
    b = val[val.label == 0]
    is_cam = (b.kind == "camera_only").to_numpy()
    out["camera_vs_original_auc"] = float(roc_auc_score(is_cam, b.score)) if 0 < is_cam.sum() < len(b) else float("nan")
    out["camera_median_score"] = float(b.score[is_cam].median())
    out["original_median_score"] = float(b.score[~is_cam].median())
    return out


def bootstrap_auc(dev: pd.DataFrame, seed: int, n: int = 2000) -> dict:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y, s = dev.label.to_numpy(), dev.score.to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[s[rng.choice(pos, len(pos))], s[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return {"dev_auc_lo95": float(np.percentile(aucs, 2.5)), "dev_auc_hi95": float(np.percentile(aucs, 97.5))}


def probe_batch_size(trainer: Trainer, candidates=(16, 32, 48, 64, 96, 128, 192, 256, 384, 512), steps: int = 6, headroom: float = 0.9, log=None) -> dict:
    """Time a few real training steps at each batch size until memory runs out; pick the fastest within `headroom` of GPU memory."""
    from .model import build_optimizer

    dev = trainer.device
    total = torch.cuda.get_device_properties(dev).total_memory if trainer.cuda else None
    order = trainer.sampler(0)
    results = []
    state = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    for bs in candidates:
        if bs > len(order):
            break
        try:
            opt = build_optimizer(trainer.model, 1e-6, 0.0, 0.75)
            trainer.model.train()
            if trainer.cuda:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            for k in range(steps):
                if k == 2:  # first steps include allocator / kernel warm-up
                    if trainer.cuda:
                        torch.cuda.synchronize()
                    t0 = time.time()
                idx = order[:bs]
                x = trainer.batch(idx, np.random.default_rng([0, k]))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(trainer.logits(x), trainer.labels[torch.as_tensor(idx, device=dev)])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if trainer.cuda:
                torch.cuda.synchronize()
            ips = bs * (steps - 2) / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() if trainer.cuda else 0
            results.append({"batch_size": bs, "images_per_s": ips, "peak_mem_gb": peak / 1e9,
                            "mem_frac": peak / total if total else None})
            if log:
                log.info("batch %d: %.0f img/s, peak %.1f GB", bs, ips, peak / 1e9)
            del opt
        except torch.OutOfMemoryError:
            if log:
                log.info("batch %d: out of memory", bs)
            break
        finally:
            trainer.model.zero_grad(set_to_none=True)
            if trainer.cuda:
                torch.cuda.empty_cache()
    trainer.model.load_state_dict(state)
    ok = [r for r in results if r["mem_frac"] is None or r["mem_frac"] <= headroom]
    best = max(ok, key=lambda r: r["images_per_s"]) if ok else None
    return {"total_mem_gb": total / 1e9 if total else None, "headroom": headroom, "results": results,
            "chosen": best["batch_size"] if best else None}


def overfit_check(trainer: Trainer, n_faces: int = 16, steps: int = 200, lr: float = 1e-4, log=None) -> dict:
    """Can the loop learn at all? Train on a fixed tiny set (n_faces train faces, one epoch draw, no augmentation) until it is memorised.

    Passing means train accuracy reaches 1.0 and the loss falls well below its starting value; a broken loop (labels misaligned with images, frozen weights, a wrong loss) cannot do this. Every face is in both classes and only texture separates them, so it is not instant: on an L4 (2026-10-19) DINOv2-base sat near chance for ~90 steps, then reached 100% by step 130. At 3x this learning rate it did not converge.
    """
    from .model import build_optimizer

    order = trainer.sampler(0)
    faces = sorted(set(trainer.table.id.to_numpy()[order]))[:n_faces]
    idx = order[np.isin(trainer.table.id.to_numpy()[order], faces)]
    y = trainer.labels[torch.as_tensor(idx, device=trainer.device)]
    state = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    opt = build_optimizer(trainer.model, lr, 0.0, 0.75)
    hist = []
    for step in range(steps):
        trainer.model.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(trainer.logits(trainer.batch(idx, None)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            acc = float(((trainer.score(idx, 64) > 0.5) == y.cpu().numpy()).mean())
            hist.append({"step": step, "loss": float(loss.item()), "train_acc": acc})
            if log:
                log.info("overfit step %d: loss %.4f, train acc %.3f", step, loss.item(), acc)
    trainer.model.load_state_dict(state)
    passed = hist[-1]["train_acc"] == 1.0 and hist[-1]["loss"] < 0.2 * hist[0]["loss"]
    return {"n_images": int(len(idx)), "n_attack": int(y.sum().item()), "history": hist, "passed": bool(passed)}
