"""`screenpad pull-data | manifest | splits`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup
from .manifest import build_manifest
from .splits import make_splits, split_summary


def cache_name(name: str, size: int) -> str:
    return f"crops_{name}_{size}.npz"


def cmd_pull(args) -> None:
    """Download the raw-data tar from the HF Hub, extract it into `data.root`, and verify every file's SHA-256.

    With --cache, download the preprocessed crop caches into `data.cache_dir` instead.
    """
    from huggingface_hub import hf_hub_download

    from .archive import extract_archive, read_checksums

    cfg = _make_cfg(args)
    log = _setup(cfg)
    get = lambda f, **kw: hf_hub_download(repo_id=cfg.data.hf_repo, repo_type="dataset", filename=f, **kw)
    if args.cache:
        for name in cfg.preprocess.crops:
            f = cache_name(name, int(cfg.preprocess.size))
            get(f"{cfg.data.hf_cache_dir}/{f}", local_dir=Path(cfg.data.cache_dir).parent)
            log.info("pulled %s -> %s", f, cfg.data.cache_dir)
        return
    tar_path = get(cfg.data.hf_archive)
    checksums = read_checksums(get(cfg.data.hf_checksums))
    n = extract_archive(tar_path, cfg.data.root, checksums)
    log.info("pulled %s -> %s (%d files, all SHA-256 verified)", cfg.data.hf_repo, cfg.data.root, n)


def load_or_build_manifest(cfg, log) -> pd.DataFrame:
    """Read the manifest CSV if it exists, otherwise build and save it."""
    path = Path(cfg.data.manifest)
    if path.exists():
        return pd.read_csv(path, dtype={"id": str})
    df = build_manifest(cfg.data.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote manifest (%d images) -> %s", len(df), path)
    return df


def cmd_manifest(args) -> None:
    cfg = _make_cfg(args)
    log = _setup(cfg)
    df = build_manifest(cfg.data.root)
    path = Path(cfg.data.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote manifest (%d images) -> %s", len(df), path)
    log.info("\n%s", df.groupby("label_name")[["width", "height", "has_icc", "has_exif", "paired"]].agg(["min", "max"]))


def cmd_splits(args) -> None:
    cfg = _make_cfg(args)
    log = _setup(cfg)
    splits = make_splits(load_or_build_manifest(cfg, log), cfg.split, seed=int(cfg.seed))
    path = Path(cfg.data.splits)
    splits.to_csv(path, index=False)
    log.info("wrote splits -> %s\n%s", path, split_summary(splits))


def cmd_cache(args) -> None:
    """Build one crop cache per `preprocess.crops` entry, verify each against a fresh recompute, optionally push to the HF Hub."""
    from .cache import build_crops, cache_frame, save_cache, verify_cache

    cfg = _make_cfg(args)
    log = _setup(cfg)
    splits = pd.read_csv(cfg.data.splits, dtype={"id": str})
    size, det_size = int(cfg.preprocess.size), int(cfg.preprocess.det_size)
    caches = build_crops(splits, cfg.data.root, dict(cfg.preprocess.crops), size, det_size)
    out = Path(cfg.data.cache_dir)
    for name, c in caches.items():
        path = out / cache_name(name, size)
        save_cache(c, path)
        n = verify_cache(path, cfg.data.root, det_size)
        f = cache_frame(c)
        log.info("%s: %d crops, %.0f MB, verified exact against a fresh recompute; shifted %d, shrunk %d, no face %d",
                 path, n, path.stat().st_size / 1e6, f.info_shifted.sum(), f.info_shrunk.sum(), (~f.info_face_found).sum())
        if args.push:
            from huggingface_hub import HfApi

            HfApi().upload_file(repo_id=cfg.data.hf_repo, repo_type="dataset", path_or_fileobj=path,
                                path_in_repo=f"{cfg.data.hf_cache_dir}/{path.name}")
            log.info("pushed -> %s/%s", cfg.data.hf_repo, cfg.data.hf_cache_dir)
