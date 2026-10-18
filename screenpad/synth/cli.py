"""`screenpad synth`: synthetic attacks and camera-only controls, as crop caches like `screenpad cache`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup


def synth_name(name: str, size: int, splits: str) -> str:
    return f"synth_{name}_{size}_{splits.replace(',', '-')}.npz"


def cmd_synth(args) -> None:
    from ..data.cache import save_cache
    from .generate import generate

    cfg = _make_cfg(args)
    log = _setup(cfg)
    wanted = args.splits or ",".join(cfg.synth.splits)
    if "test" in wanted.split(","):
        raise SystemExit("synthetic data is never generated from test images")
    splits = pd.read_csv(cfg.data.splits, dtype={"id": str})
    splits = splits[splits["split"].isin(wanted.split(","))]
    size = int(cfg.preprocess.size)
    caches, params = generate(
        splits, cfg.data.root, cfg.synth.ranges, dict(cfg.preprocess.crops), size, int(cfg.preprocess.det_size),
        bool(cfg.preprocess.align), int(cfg.seed), int(args.variants or cfg.synth.variants), float(cfg.synth.camera_only_frac),
        int(cfg.num_threads),
    )
    out = Path(cfg.data.cache_dir)
    params_path = out / synth_name("params", size, wanted).replace(".npz", ".csv")
    params.to_csv(params_path, index=False)
    for name, c in caches.items():
        path = out / synth_name(name, size, wanted)
        save_cache(c, path)
        log.info("%s: %d crops (%s), %.0f MB", path, len(c["id"]), pd.Series(c["kind"]).value_counts().to_dict(), path.stat().st_size / 1e6)
        if args.push:
            from huggingface_hub import HfApi

            HfApi().upload_file(repo_id=cfg.data.hf_repo, repo_type="dataset", path_or_fileobj=path, path_in_repo=f"{cfg.data.hf_cache_dir}/{path.name}")
    log.info("parameters -> %s", params_path)
