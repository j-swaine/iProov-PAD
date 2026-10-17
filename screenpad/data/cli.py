"""`screenpad pull-data | manifest | splits`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup
from .manifest import build_manifest
from .splits import make_splits, split_summary


def cmd_pull(args) -> None:
    """Download the dataset repo from the HF Hub into `data.root`'s parent (the repo holds `data/Bonafide`, `data/Screens`)."""
    from huggingface_hub import snapshot_download

    cfg = _make_cfg(args)
    log = _setup(cfg)
    dest = Path(cfg.data.root).parent
    snapshot_download(repo_id=cfg.data.hf_repo, repo_type="dataset", local_dir=dest)
    log.info("pulled %s -> %s", cfg.data.hf_repo, dest)


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
