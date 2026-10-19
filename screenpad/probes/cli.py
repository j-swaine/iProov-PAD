"""`screenpad probe`: frozen features + logistic regression, trained on synthetic attacks, measured on real dev attacks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir, save_json

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, seed: int, n: int = 2000) -> tuple[float, float]:
    """95% percentile interval of AUC, resampling attacks and bona fide separately (stratified)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[s[rng.choice(pos, len(pos))], s[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cmd_probe(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..eval.metrics import bpcer_at_apcer, summary
    from ..utils.device import resolve_device
    from .data import load_sets, set_masks
    from .features import extract

    cfg = _make_cfg(args)
    log = _setup(cfg)
    device = resolve_device(cfg.device) if args.backbones != "handcrafted" else None
    size = int(cfg.preprocess.size)
    out = ensure_dir(Path(cfg.output_dir) / "probes")
    feat_dir = ensure_dir(Path(cfg.data.cache_dir) / "features")
    rows, scores = [], []
    for crop in args.crops.split(","):
        images, table = load_sets(cfg.data.cache_dir, crop, size)
        m = set_masks(table)
        digest = hashlib.sha1(images.tobytes()).hexdigest()  # features are reused only for byte-identical inputs
        log.info("%s: train %s, val %s, dev %s", crop, table[m["train"]].kind.value_counts().to_dict(),
                 table[m["val"]].kind.value_counts().to_dict(), table[m["dev"]].kind.value_counts().to_dict())
        for bb in args.backbones.split(","):
            fpath = feat_dir / f"{bb}_{crop}_{size}.npz"
            if fpath.exists() and str(np.load(fpath)["digest"]) == digest:
                X = np.load(fpath)["X"]
            else:
                log.info("extracting %s on %s crops (%d images, device %s)", bb, crop, len(images), device)
                X = extract(bb, images, device)
                np.savez(fpath, X=X, digest=digest)
            y = table.label.to_numpy()
            best = None
            for C in C_GRID:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
                clf.fit(X[m["train"]], y[m["train"]])
                val_auc = roc_auc_score(y[m["val"]], clf.predict_proba(X[m["val"]])[:, 1])
                if best is None or val_auc > best[0]:
                    best = (val_auc, C, clf)
            val_auc, C, clf = best
            s = clf.predict_proba(X)[:, 1]
            yd, sd = y[m["dev"]], s[m["dev"]]
            lo, hi = _bootstrap_auc(yd, sd, int(cfg.seed))
            dev = summary(yd, sd, threshold=0.5)
            rows.append({"backbone": bb, "crop": crop, "C": C, "val_auc_synthetic": val_auc,
                         "dev_auc": dev["auc"], "dev_auc_ci95": f"[{lo:.2f}, {hi:.2f}]", "dev_eer": dev["eer"],
                         "dev_apcer@0.5": dev["apcer@0.5"], "dev_bpcer@0.5": dev["bpcer@0.5"],
                         "dev_bpcer@apcer0": bpcer_at_apcer(yd, sd, 0.0)})
            log.info("%s/%s: C=%g val AUC (synthetic) %.3f | dev AUC (real) %.3f [%.2f, %.2f], BPCER at APCER 0: %.3f",
                     bb, crop, C, val_auc, dev["auc"], lo, hi, rows[-1]["dev_bpcer@apcer0"])
            scores.append(table.assign(backbone=bb, crop=crop, score=s)[["id", "split", "label", "kind", "info_resample_ratio", "backbone", "crop", "score"]])
    res = pd.DataFrame(rows)
    res.to_csv(out / "probe_results.csv", index=False)
    pd.concat(scores).to_csv(out / "probe_scores.csv", index=False)
    save_json({"c_grid": list(C_GRID), "note": "dev = 6 real attacks vs 6 twins + 120 real val bona fide; val bona fide also chose C"}, out / "probe_meta.json")
    log.info("\n%s", res.round(3).to_string(index=False))
