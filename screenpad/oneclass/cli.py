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
    weights = None
    if args.weights_run:  # features from a fine-tuned classifier's backbone (fetched from the HF runs repo if not local)
        local = Path(cfg.train.runs_dir) / args.weights_run / "ckpt" / "last.pt"
        if local.exists():
            weights = str(local)
        else:
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(cfg.train.hf_runs_repo, f"runs/{args.weights_run}/ckpt/last.pt", repo_type="model")
    dino = Dino(device, args.input_size, args.block, weights, args.arch)
    tag = (f"{args.crop}{args.input_size}" + (f"_{args.arch}" if args.arch != "dinov2_b" else "")
           + (f"_b{args.block}" if args.block != 8 and args.arch != "wrn50" else "") + (f"_ft{args.weights_run.split('_')[0]}" if args.weights_run else ""))

    def finish(name: str, score_fn, member_tag: str | None = None) -> None:
        out = ensure_dir((run_dir or Path(cfg.output_dir) / "oneclass") / f"{name}_{member_tag or tag}")
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
        save_json({"member": name, "crop": args.crop, "input_size": args.input_size, "arch": args.arch, "block": args.block, "coreset": args.coreset, "weights_run": args.weights_run,
                   "n_fit": int(fit.sum()), "metrics": m}, out / "summary.json")
        log.info("%s: %s", name, {k: round(v, 3) for k, v in m.items()})

    if "efficientad" in args.members:
        import torch

        from . import efficientad as ead

        tag_e = f"{args.crop}{args.ead_size}"
        work = Path(cfg.output_dir) / "oneclass" / "_efficientad_work" / tag_e
        imagenet = Path(args.imagenet_dir)
        if args.smoke:
            ead.make_imagenette_stub(imagenet)
        norm = ((table.split == "val") & (table.label == 0) & table.kind.isin(["real_bonafide", "camera_only"])).to_numpy()
        workers = args.workers
        if args.probe_workers:  # short trial runs: pick the data-loading setting with the best throughput (batch size is fixed at 1 by the method)
            import json
            import shutil

            bench = []
            for w in [int(x) for x in args.probe_workers.split(",")]:
                _, t = ead.train(work / f"_probe_w{w}", images[fit_idx], images[remap[np.flatnonzero(norm)]], args.ead_size, 300,
                                 int(cfg.seed), imagenet, num_workers=w, log=log)
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
                bench.append({"workers": w, "steps_per_s": t["steps_per_s"], "peak_mem_gb": mem})
                shutil.rmtree(work / f"_probe_w{w}", ignore_errors=True)
                log.info("probe: %s", bench[-1])
            workers = max(bench, key=lambda b: b["steps_per_s"])["workers"]
            gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else str(device)
            (run_dir or work).joinpath("bench.json").write_text(json.dumps({"gpu": gpu, "probe": bench, "chosen_workers": workers,
                                                                           "eta_full_min": round(args.steps / max(b["steps_per_s"] for b in bench) / 60)}, indent=2))
        model, timing = ead.train(work, images[fit_idx], images[remap[np.flatnonzero(norm)]], args.ead_size, args.steps,
                                  int(cfg.seed), imagenet, num_workers=workers, log=log,
                                  status_path=run_dir / "STATUS" if run_dir else None, progress_every=max(1, min(500, args.steps // 10)))
        log.info("EfficientAD trained: %s", timing)
        maps = {}

        def ead_score(idx):
            s, mp = ead.score(model, img(idx), args.ead_size, device)
            maps[len(maps)] = (idx, mp)
            return s

        name = f"efficientad_{tag_e}"
        finish("efficientad", ead_score, tag_e)
        edir = (run_dir or Path(cfg.output_dir) / "oneclass") / name
        torch.save({"state_dict": model.model.state_dict(), "image_size": args.ead_size, "crop": args.crop, "model_size": "small",
                    "timing": timing, "anomalib": __import__("anomalib").__version__}, edir / "efficientad.pt")
        dev_idx, dev_maps = next(v for v in maps.values() if np.array_equal(v[0], sets["dev"]))
        np.savez_compressed(edir / "heatmaps.npz", maps=dev_maps.astype(np.float16), id=table.id.to_numpy()[dev_idx].astype(str),
                            label=table.label.to_numpy()[dev_idx], kind=table.kind.to_numpy()[dev_idx].astype(str))
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
        pc.save(bank_dir / "bank.pt", {"block": args.block, "input_size": args.input_size, "crop": args.crop, "coreset": args.coreset,
                                       "weights_run": args.weights_run or "", "arch": args.arch})
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


def cmd_ssl_adapt(args) -> None:
    """`screenpad ssl-adapt`: self-supervised domain adaptation of DINOv2-base on train bona fide (oneclass/ssl.py), as a run folder."""
    import json
    import traceback

    from ..config import REPO_ROOT, save_config
    from ..train.data import load_pool
    from ..utils.device import resolve_device
    from ..utils.provenance import write_provenance
    from .ssl import train

    cfg = _make_cfg(args)
    run_dir = Path(cfg.train.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup(cfg, logfile=run_dir / "train.log")
    status = run_dir / "STATUS"
    status.write_text(json.dumps({"state": "starting"}))
    try:
        save_config(cfg, run_dir / "config.yaml")
        write_provenance(run_dir / "provenance.json", REPO_ROOT, extra={"run": args.run_name, "kind": "ssl-adapt", **vars(args) | {"func": None}})
        images, table = load_pool(cfg.data.cache_dir, args.crop, int(cfg.preprocess.size))
        fit = ((table.split == "train") & (((table.kind == "real_bonafide") & table.train_ok) | (table.kind == "camera_only"))).to_numpy()
        imgs = images[fit]
        del images
        log.info("ssl-adapt %s: %d train bona fide, %d epochs, batch %d, %d local crops", args.run_name, len(imgs), args.epochs, args.batch_size, args.n_local)
        res = train(imgs, run_dir, resolve_device(cfg.device), args.epochs, args.batch_size, int(cfg.seed), n_local=args.n_local,
                    lr=args.lr, head_warmup=args.head_warmup, save=not args.no_ckpt, grad_ckpt=args.grad_ckpt, log=log)
        h = res["history"]
        (run_dir / "bench.json").write_text(json.dumps({"images_per_s": h[-1]["images_per_s"], "peak_mem_gb": h[-1]["peak_mem_gb"],
                                                        "epoch_s": h[-1]["epoch_s"], "batch_size": args.batch_size, "n_local": args.n_local, "grad_ckpt": args.grad_ckpt,
                                                        "train_s": res["train_s"]}, indent=2))
        if args.push:
            from ..train.hub import ensure_repo, upload_run, verify_run

            ensure_repo(cfg.train.hf_runs_repo)
            upload_run(run_dir, cfg.train.hf_runs_repo, f"runs/{args.run_name}")
            problems = verify_run(run_dir, cfg.train.hf_runs_repo, f"runs/{args.run_name}")
            if problems:
                raise RuntimeError(f"upload verification failed: {problems[:5]}")
        status.write_text(json.dumps({"state": "done", "pushed": bool(args.push)}))
    except Exception:
        status.write_text(json.dumps({"state": "failed", "error": traceback.format_exc()[-4000:]}))
        log.exception("ssl-adapt failed")
        raise
