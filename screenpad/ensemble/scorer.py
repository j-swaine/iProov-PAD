"""
Score new face crops with the frozen ensemble (results/ensemble/frozen.json, written by `screenpad ensemble`).
--------------------------------------------------------------------------------------------------------------

For each enabled member: run its model(s) on the crops it was built for (its crop variant and input size), map the raw score to the calibrated scale with the frozen bona fide statistics (logit or log, then (t - median) / scale, as in `core.robust_z`), average seeds within the member, then combine members with the frozen rule. An image is flagged as an attack when the ensemble z is at or above the frozen threshold.

Models are loaded one at a time and released, so peak memory is one model plus the crops.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .core import _transform, combine

log = logging.getLogger("screenpad")


class EnsembleScorer:
    def __init__(self, frozen: str | Path | dict, device, batch_size: int = 64):
        """`batch_size`: images per forward pass; lower it to cut peak memory (scores do not depend on it)."""
        self.f = json.loads(Path(frozen).read_text()) if not isinstance(frozen, dict) else frozen
        self.device = device
        self.batch_size = batch_size
        self.roots = self.f["roots"]

    @property
    def crops(self) -> set[str]:
        """Crop variants the members need (keys of the dict passed to `score`)."""
        from ..train.inference import run_config

        out = set()
        for spec in self.f["members"].values():  # all members, so `all_members` scoring has its crops too
            if spec["type"] == "run":
                out |= {run_config(Path(self.roots["runs"]) / r)["crop"] for r in spec["runs"]}
            elif self.scorable(spec):
                out.add(self._bank(spec)[1]["crop"])
        return out

    def scorable(self, spec: dict) -> bool:
        """Can this member score new images? (A one-class member needs its saved memory bank; Mahalanobis keeps none.)"""
        return spec["type"] == "run" or (Path(spec.get("root") or self.roots["oneclass"]) / spec["path"] / "bank.pt").exists()

    def _bank(self, spec):
        import torch

        path = Path(spec.get("root") or self.roots["oneclass"]) / spec["path"] / "bank.pt"  # `root`: per-member override written by fetch-models
        meta = {k: v for k, v in torch.load(path, map_location="cpu", weights_only=True).items() if k != "bank"}
        return path, meta

    def _release(self):
        import torch

        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def member_t(self, name: str, crops: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per source (seed run or one-class model): raw scores on the calibrated scale before standardising (logit or log distance)."""
        spec = self.f["members"][name]
        out = {}
        if spec["type"] == "run":
            from ..train.inference import load_run, score_images

            for j, r in enumerate(spec["runs"], 1):
                t0 = time.perf_counter()
                model, c = load_run(Path(self.roots["runs"]) / r, self.device)
                s = score_images(model, crops[c["crop"]], int(c["input_size"]), self.device, self.batch_size, desc=f"{name} {j}/{len(spec['runs'])}")
                del model
                self._release()
                out[r] = _transform(s, "run")
                log.info("scored %s (%s, %d/%d): %d images in %.0f s", name, r, j, len(spec["runs"]), len(s), time.perf_counter() - t0)
        else:
            from ..oneclass.models import Dino, PatchCore

            path, meta = self._bank(spec)
            pc, _ = PatchCore.load(path, self.device)
            w = Path(self.roots["runs"]) / meta["weights_run"] / "ckpt" / "last.pt" if meta.get("weights_run") else None
            dino = Dino(self.device, int(meta["input_size"]), int(meta["block"]), str(w) if w else None, meta.get("arch", "dinov2_b"))
            from tqdm import tqdm

            t0 = time.perf_counter()
            imgs = crops[meta["crop"]]
            b = self.batch_size
            s = np.concatenate([pc.score(dino.patches(imgs[i : i + b], batch=b))[0]
                                for i in tqdm(range(0, len(imgs), b), desc=name, unit="batch", mininterval=30)])
            del pc, dino
            self._release()
            out[spec["path"]] = _transform(s, "oneclass")
            log.info("scored %s: %d images in %.0f s", name, len(s), time.perf_counter() - t0)
        return out

    def member_z(self, name: str, crops: dict[str, np.ndarray]) -> np.ndarray:
        cal = {c["source"]: c for c in self.f["members"][name]["calibration"]}
        return np.mean([(t - cal[src]["median"]) / cal[src]["scale"] for src, t in self.member_t(name, crops).items()], axis=0)

    def score(self, crops: dict[str, np.ndarray], all_members: bool = False) -> pd.DataFrame:
        """One row per image: each member's z, the ensemble z and the decision (attack = z >= threshold).

        The ensemble uses the enabled members only; `all_members` also scores the disabled ones (for a descriptive ablation).
        """
        names = list(self.f["enabled"])
        pool = self.f.get("ablation_members", list(self.f["members"]))
        todo = [n for n, sp in self.f["members"].items() if n in names or (n in pool and self.scorable(sp))] if all_members else names
        df = pd.DataFrame({n: self.member_z(n, crops) for n in todo})
        families = {n: self.f["members"][n]["family"] for n in names}
        df["ensemble_z"] = combine(df, names, families, self.f["combine"])
        df["attack"] = df.ensemble_z >= self.f["threshold_z"]
        return df
