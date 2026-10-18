"""`screenpad baseline-minifasnet`: score images with the off-the-shelf MiniFASNet model, native and context-matched."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir, save_json


def cmd_minifasnet(args) -> None:
    from ..data.io import load_image
    from ..eval.metrics import summary
    from .minifasnet import MiniFASNetBaseline

    cfg = _make_cfg(args)
    log = _setup(cfg)
    splits = pd.read_csv(cfg.data.splits, dtype={"id": str})
    use = splits[splits["split"].isin(args.splits.split(","))].copy()
    model = MiniFASNetBaseline()

    rows = []
    # Bona fide first, so each attack can be capped at the context its twin received.
    twin_scales = {}
    for _, r in use.sort_values("label").iterrows():
        bgr = np.asarray(load_image(Path(cfg.data.root) / r["path"]))[:, :, ::-1].copy()  # same decode as everything else
        native, used = model.score(bgr)
        row = {"id": r["id"], "label": r["label"], "split": r["split"], "native": native,
               "scale_2.7": used[0], "scale_4.0": used[1]}
        if r["label"] == 0:
            twin_scales[r["id"]] = used
            row["matched"] = native
        else:
            cap = twin_scales.get(r["id"])
            row["matched"] = model.score(bgr, max_scales=cap)[0] if cap else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)

    out = ensure_dir(Path(cfg.output_dir) / "baselines")
    df.to_csv(out / "minifasnet_scores.csv", index=False)
    report = {}
    for mode in ("native", "matched"):
        d = df.dropna(subset=[mode])
        report[mode] = summary(d["label"].to_numpy(), d[mode].to_numpy(), threshold=0.5)
    save_json({"splits": args.splits, **report}, out / "minifasnet_metrics.json")
    log.info("context scale actually used (median), by class:\n%s",
             df.groupby("label")[["scale_2.7", "scale_4.0"]].median().round(2).to_string())
    log.info("metrics (threshold 0.5 = upstream decision rule):\n%s", pd.DataFrame(report).round(3).to_string())
    log.info("-> %s", out)
