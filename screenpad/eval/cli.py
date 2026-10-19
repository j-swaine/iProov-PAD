"""`screenpad shortcut-test`: build the resampling-counterfactual crops and/or score trained runs on them (see eval/shortcut.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir, save_json


def cmd_shortcut_test(args) -> None:
    from ..data.cache import load_cache, save_cache
    from .shortcut import FACTORS, build_shrink_set, shrink_metrics, shrink_name

    cfg = _make_cfg(args)
    log = _setup(cfg)
    size = int(cfg.preprocess.size)
    cache_dir = Path(cfg.data.cache_dir)
    if args.build:
        splits = pd.read_csv(Path(cfg.output_dir) / "splits.csv", dtype={"id": str})
        for crop, k in dict(cfg.preprocess.crops).items():
            path = cache_dir / shrink_name(crop, size)
            c = build_shrink_set(splits, cfg.data.root, float(k), size, int(cfg.preprocess.det_size), bool(cfg.preprocess.align))
            ref = load_cache(cache_dir / f"crops_{crop}_{size}_aligned.npz")  # factor 1 must reproduce the cached crops exactly
            pos = {i: n for n, i in enumerate(ref["id"]) if ref["label"][n] == 0}
            same = [(ref["images"][pos[i]] == img).all() for i, img, f in zip(c["id"], c["images"], c["factor"]) if f == 1]
            if not all(same):
                raise ValueError(f"{crop}: {len(same) - sum(same)} factor-1 crops differ from the cache")
            save_cache(c, path)
            log.info("%s: %d crops (%d faces x %d factors), factor 1 identical to cache -> %s", crop, len(c["id"]), len(same), len(FACTORS), path)
            if args.push:
                from huggingface_hub import HfApi

                HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"{cfg.data.hf_cache_dir}/{path.name}", repo_id=cfg.data.hf_repo, repo_type="dataset")
                log.info("pushed -> %s/%s", cfg.data.hf_repo, cfg.data.hf_cache_dir)
    if args.control:
        from ..probes.features import extract
        from ..utils.device import resolve_device
        from .shortcut import detectability

        rows = []
        for crop in dict(cfg.preprocess.crops):
            sc = load_cache(cache_dir / shrink_name(crop, size))
            for bb in ("dinov2_b224", "handcrafted"):
                X = extract(bb, sc["images"], resolve_device(cfg.device) if bb != "handcrafted" else None)
                r = {"crop": crop, "features": bb, **detectability(X, sc["id"], sc["factor"])}
                rows.append(r)
                log.info("positive control %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
        out = ensure_dir(Path(cfg.output_dir) / "shortcut")
        pd.DataFrame(rows).to_csv(out / "shrink_detectability.csv", index=False)
    if args.runs:
        import torch

        from ..train.inference import load_run, run_config, score_images
        from ..utils.device import resolve_device

        device = resolve_device(cfg.device)
        out = ensure_dir(Path(cfg.output_dir) / "shortcut")
        frames, rows = [], []
        for run_dir in args.runs:
            name = Path(run_dir).name
            crop = run_config(run_dir)["crop"]
            sc = load_cache(cache_dir / shrink_name(crop, size))
            model, c = load_run(run_dir, device)
            s = score_images(model, sc["images"], int(c["input_size"]), device)
            df = pd.DataFrame({"run": name, "id": sc["id"], "variant": sc["variant"], "factor": sc["factor"],
                               "resample_ratio": sc["info_resample_ratio"], "score": s})
            frames.append(df)
            m = shrink_metrics(df)
            rows.append({"run": name, **m})
            log.info("%s: %s", name, {k: round(v, 1) for k, v in m.items()})
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        pd.concat(frames).to_csv(out / "shrink_scores.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "shrink_summary.csv", index=False)
        save_json({"factors": list(FACTORS), "runs": rows}, out / "shrink_summary.json")
