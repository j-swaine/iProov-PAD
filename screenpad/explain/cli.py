"""`screenpad explain`: face-vs-surroundings, fine-texture and occlusion tests on the dev set for the enabled ensemble members (see explain/core.py).

Scores per member: classifiers as the mean attack logit over their seeds; PatchCore as log(worst-patch distance), from its saved memory bank. Writes results/explain/: face_context.csv and frequency.csv (per image, condition and member), summary.csv (per condition and member: AUC, median score per class), occlusion.npz (maps for the 6 dev attacks and their twins), and figures/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir


def _scorers(cfg, device, log):
    """{member: fn(uint8 images) -> attack score}, loading one model at a time."""
    import torch

    from ..oneclass.models import Dino, PatchCore
    from ..train.inference import load_run, score_images
    from .core import logit

    e = cfg.ensemble
    out = {}
    for name, spec in dict(e.members).items():
        if not spec["enabled"]:
            continue
        if spec["type"] == "run":
            def fn(images, runs=list(spec["runs"])):
                acc = 0
                for r in runs:
                    model, c = load_run(Path(e.roots["runs"]) / r, device)
                    acc = acc + logit(score_images(model, images, int(c["input_size"]), device))
                    del model
                    if device.type == "mps":
                        torch.mps.empty_cache()
                return acc / len(runs)
            out[name] = fn
        else:
            bank = Path(e.roots["oneclass"]) / spec["path"] / "bank.pt"
            if not bank.exists():
                log.warning("%s: no memory bank at %s; skipped", name, bank)
                continue

            def fn(images, bank=bank):
                pc, meta = PatchCore.load(bank, device)
                w = Path(e.roots["runs"]) / meta["weights_run"] / "ckpt" / "last.pt" if meta.get("weights_run") else None
                dino = Dino(device, int(meta["input_size"]), int(meta["block"]), str(w) if w else None, meta.get("arch", "dinov2_b"))
                s = np.concatenate([pc.score(dino.patches(images[i : i + 64]))[0] for i in range(0, len(images), 64)])
                return np.log(np.maximum(s, 1e-12))
            out[name] = fn
    return out


def cmd_explain(args) -> None:
    from sklearn.metrics import roc_auc_score

    from ..data.cache import cache_frame, load_cache
    from ..utils.device import resolve_device
    from .core import face_context_variants, frequency_variants, occlusion_grid

    cfg = _make_cfg(args)
    log = _setup(cfg)
    device = resolve_device(cfg.device)
    size = int(cfg.preprocess.size)
    c = load_cache(Path(cfg.data.cache_dir) / f"crops_{args.crop}_{size}_aligned.npz")
    f = cache_frame(c)
    dev = ((f.split == "dev") | ((f.split == "val") & (f.label == 0))).to_numpy()
    imgs, t = c["images"][dev], f[dev].reset_index(drop=True)
    face_frac = (t.info_face_w / t.info_side).to_numpy()
    log.info("dev set: %d attacks, %d bona fide", (t.label == 1).sum(), (t.label == 0).sum())

    rows, batch = [], []
    for i, img in enumerate(imgs):
        for test, variants in (("face_context", face_context_variants(img, face_frac[i])), ("frequency", frequency_variants(img))):
            for cond, v in variants.items():
                rows.append({"i": i, "id": t.id[i], "label": int(t.label[i]), "test": test, "condition": cond})
                batch.append(v)
    var = pd.DataFrame(rows)
    batch = np.stack(batch)
    pairs = np.flatnonzero(t.split.to_numpy() == "dev")  # the 6 attacks and their 6 twins
    occ = np.concatenate([occlusion_grid(imgs[i], args.cells) for i in pairs])

    scorers = _scorers(cfg, device, log)
    maps = {}
    for name, fn in scorers.items():
        log.info("scoring %s on %d variants + %d occlusions", name, len(batch), len(occ))
        s = fn(np.concatenate([batch, occ]))
        var[name] = s[: len(batch)]
        base = var[(var.test == "frequency") & (var.condition == "original")].set_index("i")[name]
        o = s[len(batch) :].reshape(len(pairs), args.cells, args.cells)
        maps[name] = base.loc[pairs].to_numpy()[:, None, None] - o  # drop in score when a cell is hidden
    out = ensure_dir(Path(cfg.output_dir) / "explain")
    for test in ("face_context", "frequency"):
        var[var.test == test].to_csv(out / f"{test}.csv", index=False)
    summ = []
    for (test, cond), g in var.groupby(["test", "condition"], sort=False):
        for name in scorers:
            summ.append({"test": test, "condition": cond, "member": name, "dev_auc": float(roc_auc_score(g.label, g[name])),
                         "median_attack": float(g[g.label == 1][name].median()), "median_bonafide": float(g[g.label == 0][name].median())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    np.savez_compressed(out / "occlusion.npz", images=imgs[pairs], id=t.id.to_numpy()[pairs].astype(str), label=t.label.to_numpy()[pairs],
                        **{f"map_{k}": v.astype(np.float32) for k, v in maps.items()})
    log.info("summary:\n%s", summ.round(3).to_string(index=False))
    from .figures import make_figures

    make_figures(out, var, summ, list(scorers), Path(cfg.ensemble.roots["oneclass"]))
