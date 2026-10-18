"""`screenpad eda`: detection table, attack/twin grid and spectra under results/eda/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir


def cmd_eda(args) -> None:
    from .detections import detect_all
    from .figures import pairs_grid, spectra

    cfg = _make_cfg(args)
    log = _setup(cfg)
    out = ensure_dir(Path(cfg.output_dir) / "eda")
    splits = pd.read_csv(cfg.data.splits, dtype={"id": str})
    crops = dict(cfg.preprocess.crops)

    det = detect_all(splits, cfg.data.root, cfg.preprocess.det_size, crops, cfg.preprocess.size)
    det.to_csv(out / "detections.csv", index=False)
    log.info("detections -> %s (no face: %d images)", out / "detections.csv", (~det["face_found"]).sum())

    pairs_grid(splits, cfg.data.root, crops, out / "pairs_grid.png")
    log.info("attack/twin grid -> %s", out / "pairs_grid.png")

    hf = spectra(splits, cfg.data.root, crops["tight"], cfg.preprocess.size, out, seed=int(cfg.seed))
    hf.to_csv(out / "spectra_hf_energy.csv", index=False)
    log.info("spectra -> %s; high-minus-low frequency energy by group:\n%s", out,
             hf.groupby("group")["hf_minus_lf"].describe()[["count", "mean", "min", "max"]].round(3).to_string())
