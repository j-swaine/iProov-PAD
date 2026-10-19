"""`screenpad oneclass`: fit PatchCore and Mahalanobis on train bona fide and score the val, dev and shrink-test sets.

With --run-name (used on Colab via scripts/orchestrate.py), outputs go to <runs_dir>/<run-name>/<member>_<tag>/ with a STATUS file, and --push uploads and verifies that folder on the HF runs repo like a training run.

Writes results/oneclass/<member>/: scores.csv (same columns as a training run's scores_last.csv, so the ensemble reads both alike), summary.json (dev metrics, shortcut checks and the shrink test), and for PatchCore heatmaps.npz (per-patch distances for the dev set).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir, save_json


def cmd_oneclass(args) -> None:
    import json
    import traceback

    if not args.run_name:
        return _oneclass(args, None)
    from ..config import load_config

    run_dir = Path(load_config().train.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / "STATUS"
    status.write_text(json.dumps({"state": "running"}))
    try:
        _oneclass(args, run_dir)
        if args.push:
            from ..train.hub import ensure_repo, upload_run, verify_run

            cfg = load_config()
            ensure_repo(cfg.train.hf_runs_repo)
            upload_run(run_dir, cfg.train.hf_runs_repo, f"runs/{args.run_name}")
            problems = verify_run(run_dir, cfg.train.hf_runs_repo, f"runs/{args.run_name}")
            if problems:
                raise RuntimeError(f"upload verification failed: {problems[:5]}")
        status.write_text(json.dumps({"state": "done", "pushed": bool(args.push)}))
    except Exception:
        status.write_text(json.dumps({"state": "failed", "error": traceback.format_exc()[-4000:]}))
        raise


def _oneclass(args, run_dir: Path | None) -> None:
    from sklearn.metrics import roc_auc_score

    from ..eval.metrics import bpcer_at_apcer, eer
    from ..eval.shortcut import shrink_metrics
    from ..train.data import eval_sets, load_pool
    from ..train.loop import bootstrap_auc, shortcut_checks
    from ..utils.device import resolve_device
    from .models import Dino, Mahalanobis, PatchCore

    cfg = _make_cfg(args)
    log = _setup(cfg)
    device = resolve_device(cfg.device)
    size = int(cfg.preprocess.size)
    images, table = load_pool(cfg.data.cache_dir, args.crop, size)
    fit = ((table.split == "train") & (((table.kind == "real_bonafide") & table.train_ok) | (table.kind == "camera_only"))).to_numpy()
    sets = eval_sets(table)
    log.info("fit on %d train bona fide (%s); sets %s", fit.sum(), table[fit].kind.value_counts().to_dict(), {k: len(v) for k, v in sets.items()})
    # keep only the images used here (fit set + evaluation sets) and free the rest of the ~3 GB pool
    keep = np.unique(np.r_[np.flatnonzero(fit), np.concatenate(list(sets.values()))])
    remap = np.full(len(table), -1)
    remap[keep] = np.arange(len(keep))
    images = images[keep]
    fit_idx = remap[np.flatnonzero(fit)]
    img = lambda idx: images[remap[idx]]
    dino = Dino(device, args.input_size, args.block)
    tag = f"{args.crop}{args.input_size}"

    def finish(name: str, score_fn) -> None:
        out = ensure_dir((run_dir or Path(cfg.output_dir) / "oneclass") / f"{name}_{tag}")
        frames, m = [], {}
        for s_name, idx in sets.items():
            s = score_fn(idx)
            frames.append(table.iloc[idx][["id", "split", "label", "kind", "variant", "info_resample_ratio"]].assign(set=s_name, score=s))
            if s_name == "shrink":
                m.update(shrink_metrics(frames[-1]))
                continue
            y = table.label.to_numpy()[idx]
            m.update({f"{s_name}_auc": float(roc_auc_score(y, s)), f"{s_name}_eer": eer(y, s), f"{s_name}_bpcer@apcer0": bpcer_at_apcer(y, s, 0.0)})
        scores = pd.concat(frames, ignore_index=True)
        m.update(shortcut_checks(scores[scores.set == "val"]))
        m.update(bootstrap_auc(scores[scores.set == "dev"], int(cfg.seed)))
        scores.to_csv(out / "scores.csv", index=False)
        save_json({"member": name, "crop": args.crop, "input_size": args.input_size, "block": args.block, "coreset": args.coreset,
                   "n_fit": int(fit.sum()), "metrics": m}, out / "summary.json")
        log.info("%s: %s", name, {k: round(v, 3) for k, v in m.items()})

    if "mahalanobis" in args.members:
        g = dino.global_features(images)
        maha = Mahalanobis().fit(g[fit_idx])
        finish("mahalanobis", lambda idx: maha.score(g[remap[idx]]))
        del g
    if "patchcore" in args.members:
        log.info("patch features for %d fit images", fit.sum())
        pc = PatchCore(device, args.coreset, int(cfg.seed)).fit(dino.patches(images[fit_idx]))
        log.info("memory bank %s", tuple(pc.bank.shape))
        bank_dir = ensure_dir((run_dir or Path(cfg.output_dir) / "oneclass") / f"patchcore_{tag}")
        pc.save(bank_dir / "bank.pt", {"block": args.block, "input_size": args.input_size, "crop": args.crop, "coreset": args.coreset})
        maps = {}

        def pc_score(idx):
            s, mp = pc.score(dino.patches(img(idx)))
            maps[len(maps)] = (idx, mp)
            return s

        finish("patchcore", pc_score)
        dev_idx, dev_maps = next(v for v in maps.values() if np.array_equal(v[0], sets["dev"]))
        np.savez_compressed((run_dir or Path(cfg.output_dir) / "oneclass") / f"patchcore_{tag}" / "heatmaps.npz",
                            maps=dev_maps.astype(np.float16), id=table.id.to_numpy()[dev_idx].astype(str),
                            label=table.label.to_numpy()[dev_idx], kind=table.kind.to_numpy()[dev_idx].astype(str))
