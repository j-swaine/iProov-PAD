"""`screenpad evaluate`: score a split with the frozen ensemble and write the full report-level evaluation. `screenpad robustness`: benign degradations of val bona fide.

evaluate --split dev is the rehearsal: it re-scores the dev set from the images through the frozen ensemble (so its numbers must match `screenpad ensemble`), exercising every step the one-off test run will use.

evaluate --split test is the one-off final evaluation. It needs --confirm-test (so it cannot run by accident); everything else is recorded, not enforced, in results/evaluation/test/LOCK: time, commit, frozen-ensemble hash, whether the tree had uncommitted changes (with a fingerprint and UNCOMMITTED.patch), and whether this was a re-run (the previous record is kept as LOCK.prev).

Outputs in results/evaluation/<split>/: scores.csv (per image: every member's z, ensemble z, decision, phone, ITA), metrics.json (AUC with bootstrap CI, EER, BPCER@APCER0, APCER/BPCER at the frozen threshold with exact intervals), by_phone.csv, by_skin_tone.csv, per_member.csv, ablation.csv (every coalition of the scorable members, both averaging rules, descriptive), det.csv.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..cli_common import _make_cfg, _setup
from ..config import REPO_ROOT
from ..utils import ensure_dir, save_json


def _rows(f: pd.DataFrame, split: str) -> np.ndarray:
    if split == "dev":  # the dev set used throughout: 6 dev attacks + their twins + val bona fide originals
        return ((f.split == "dev") | ((f.split == "val") & (f.label == 0))).to_numpy()
    return (f.split == "test").to_numpy()


def _guard_test(args, frozen_path: Path, out: Path, log) -> dict:
    if not args.confirm_test:
        raise SystemExit("--split test is the one-off final evaluation: pass --confirm-test")
    lock = out / "LOCK"
    rerun = lock.exists()
    if rerun:  # advisory: keep the previous record and note the re-run, so the report can say so
        lock.rename(out / "LOCK.prev")
        log.warning("test was evaluated before (kept as LOCK.prev); this is a re-run")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    tree = {"dirty": bool(dirty)}
    if dirty:
        # record exactly which uncommitted code produced the result: a hash of the tracked diff plus every untracked file
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=REPO_ROOT, capture_output=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
        h = hashlib.sha256(diff)
        for f in sorted(untracked):
            h.update(f.encode()); h.update((REPO_ROOT / f).read_bytes())
        tree.update({"uncommitted_sha256": h.hexdigest(), "n_changed": len(dirty.splitlines())})
        (out / "UNCOMMITTED.patch").write_bytes(diff)
        log.warning("evaluating an uncommitted tree: fingerprint %s recorded in LOCK", tree["uncommitted_sha256"][:12])
    sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    info = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit, "frozen_sha256": sha, "rerun": rerun, **tree}
    lock.write_text(json.dumps(info))
    log.info("test evaluation locked: %s", info)
    return info


def cmd_evaluate(args) -> None:
    from sklearn.metrics import roc_auc_score

    from ..data.cache import cache_frame, load_cache
    from ..ensemble.core import ablation
    from ..ensemble.scorer import EnsembleScorer
    from ..train.loop import bootstrap_auc
    from ..utils.device import resolve_device
    from .fairness import bpcer_by_band, ita
    from .metrics import bpcer_at_apcer, eer
    from .report_metrics import by_group, det_points, phone_of, rates_at_threshold

    cfg = _make_cfg(args)
    log = _setup(cfg)
    frozen_path = Path(args.frozen)
    # the frozen system keeps results/evaluation/<split>; any other system (--tag) gets results/evaluation/<tag>/<split>
    out = ensure_dir(Path(cfg.output_dir) / "evaluation" / (args.split if args.tag == "frozen" else f"{args.tag}/{args.split}"))
    device = resolve_device(cfg.device)
    cal_dev = json.loads(frozen_path.read_text()).get("calibrated_on", {}).get("device")
    if args.split == "test" and cal_dev and cal_dev.split(":")[0] != device.type:  # checked before the lock is taken
        raise SystemExit(f"frozen ensemble was calibrated on {cal_dev} but this run would score on {device}: score on the calibration device")
    lock = _guard_test(args, frozen_path, out, log) if args.split == "test" else None
    scorer = EnsembleScorer(frozen_path, device)
    size = int(cfg.preprocess.size)
    crops, table = {}, None
    for crop in sorted(scorer.crops):
        c = load_cache(Path(cfg.data.cache_dir) / f"crops_{crop}_{size}_aligned.npz")
        f = cache_frame(c)
        rows = _rows(f, args.split)
        crops[crop] = c["images"][rows]
        t = f[rows].reset_index(drop=True)
        if table is None:
            table = t
        elif not (t.id.equals(table.id) and t.label.equals(table.label)):
            raise ValueError("crop caches disagree on image order")
    log.info("%s: %d attacks, %d bona fide", args.split, (table.label == 1).sum(), (table.label == 0).sum())

    s = scorer.score(crops, all_members=True)
    df = pd.concat([table[["id", "split", "label"]], s], axis=1)
    df["phone"] = np.where(df.label == 1, phone_of(df.id, cfg.data.manifest), "")
    face = crops.get("loose", next(iter(crops.values())))  # ITA uses the face ellipse of whichever crop is loaded
    df["ita"] = [ita(img, fr) if lab == 0 else np.nan for img, fr, lab in zip(face, (table.info_face_w / table.info_side).to_numpy(), table.label)]
    df.to_csv(out / "scores.csv", index=False)

    y, z, thr = df.label.to_numpy(), df.ensemble_z.to_numpy(), float(scorer.f["threshold_z"])
    metrics = {"split": args.split, "enabled": scorer.f["enabled"], "combine": scorer.f["combine"], "threshold_z": thr,
               "bpcer_target": scorer.f["bpcer_target"], "auc": float(roc_auc_score(y, z)), "eer": eer(y, z),
               "bpcer@apcer0": bpcer_at_apcer(y, z, 0.0), **bootstrap_auc(df.assign(score=z), int(cfg.seed)),
               **rates_at_threshold(y, z, thr), "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(), "lock": lock}
    save_json(metrics, out / "metrics.json")
    by_group(df.assign(score=z), thr, "phone").to_csv(out / "by_phone.csv", index=False)
    bpcer_by_band(df.assign(score=z), thr).to_csv(out / "by_skin_tone.csv", index=False)
    det_points(y, z).to_csv(out / "det.csv", index=False)
    members = [m for m in scorer.f.get("ablation_members", scorer.f["members"]) if m in df]  # every scored member is still in per_member.csv
    pd.DataFrame([{"member": m, "enabled": m in scorer.f["enabled"], "auc": float(roc_auc_score(y, df[m])),
                   "bpcer@apcer0": bpcer_at_apcer(y, df[m].to_numpy(), 0.0)} for m in scorer.f["members"] if m in df]).to_csv(out / "per_member.csv", index=False)
    # descriptive ablation on this split: thresholds still come from the frozen calibration set, not from this split
    families = {m: scorer.f["members"][m]["family"] for m in members}
    cal = pd.read_csv(Path(cfg.output_dir) / str(cfg.ensemble.get("dir", "ensemble")) / "member_scores.csv", dtype={"id": str})
    abl = []
    for mode in ("family", "mean"):
        a = ablation(pd.concat([cal.assign(set=cal.set.where(cal.set == "val", "other")), df.assign(set="dev")], ignore_index=True),
                     members, families, float(scorer.f["bpcer_target"]), int(cfg.seed), mode)
        abl.append(a)
    pd.concat(abl).rename(columns=lambda c: c.replace("dev_", f"{args.split}_")).to_csv(out / "ablation.csv", index=False)
    log.info("%s: %s", args.split, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if k not in ("lock",)})
    log.info("by phone:\n%s", pd.read_csv(out / "by_phone.csv").round(3).to_string(index=False))


def cmd_robustness(args) -> None:
    """Benign degradations of real images, re-scored by the frozen ensemble: BPCER and APCER per condition at the frozen threshold.

    --split val: val bona fide originals only (the first run). dev: the dev set as used throughout (6 attacks + twins + val bona fide). test: the test split (after the one-off evaluation; descriptive).
    Images are degraded and scored in chunks of --chunk images, so memory stays bounded whatever the split size.
    --suite edge replaces the degradations with the border test (outer 5/10/15% of the crop flattened), written to results/evaluation/edge_band/<tag>/<split>/.
    Outputs in results/evaluation/robustness/<tag>/<split>/: scores.csv (per image and condition: every member's z, ensemble z), summary.csv (per condition, both classes), run.json.
    """
    from ..data.cache import cache_frame, load_cache
    from ..ensemble.scorer import EnsembleScorer
    from ..utils.device import resolve_device
    from .robustness import DEGRADATIONS, EDGE_BANDS, apply_all, condition_summary

    cfg = _make_cfg(args)
    log = _setup(cfg)
    conditions = EDGE_BANDS if args.suite == "edge" else DEGRADATIONS
    frozen_path = Path(args.frozen)
    scorer = EnsembleScorer(frozen_path, resolve_device(cfg.device), args.batch_size)
    size = int(cfg.preprocess.size)
    images, table = {}, None
    for crop in sorted(scorer.crops):
        c = load_cache(Path(cfg.data.cache_dir) / f"crops_{crop}_{size}_aligned.npz")
        f = cache_frame(c)
        rows = ((f.split == "val") & (f.label == 0)).to_numpy() if args.split == "val" else _rows(f, args.split)
        images[crop] = c["images"][rows]
        t = f[rows].reset_index(drop=True)[["id", "split", "label"]]
        del c  # the full cache is ~0.5 GB per crop variant
        if table is None:
            table = t
        elif not t.equals(table):
            raise ValueError("crop caches disagree on image order")
    n = len(table)
    log.info("robustness on %s: %d bona fide + %d attacks x %d conditions, chunks of %d images",
             args.split, (table.label == 0).sum(), (table.label == 1).sum(), len(conditions) + 1, args.chunk)
    parts = []
    for k, a in enumerate(range(0, n, args.chunk), 1):
        idx = np.arange(a, min(a + args.chunk, n))
        crops, tab = {}, None
        for crop, imgs in images.items():
            crops[crop], tab = apply_all(imgs[idx], conditions)
        tab["i"] = idx[tab.i.to_numpy()]
        log.info("chunk %d/%d: images %d-%d", k, -(-n // args.chunk), idx[0], idx[-1])
        parts.append(pd.concat([tab.reset_index(drop=True), scorer.score(crops).reset_index(drop=True)], axis=1))
        del crops
    s = pd.concat(parts, ignore_index=True)
    s = pd.concat([table.iloc[s.i].reset_index(drop=True), s], axis=1)
    out = ensure_dir(Path(cfg.output_dir) / "evaluation" / ("robustness" if args.suite == "degradations" else "edge_band") / args.tag / args.split)
    s.to_csv(out / "scores.csv", index=False)
    thr = float(scorer.f["threshold_z"])
    res = condition_summary(s.assign(score=s.ensemble_z), thr)
    for m in scorer.f["enabled"]:  # per member, so each member's sensitivity is visible
        for lab, name in ((0, "bonafide"), (1, "attack")):
            g = s[s.label == lab]
            if len(g):
                res[f"median_z_{m}_{name}"] = res.condition.map(g.groupby("condition")[m].median())
    res.to_csv(out / "summary.csv", index=False)
    save_json({"frozen": str(frozen_path), "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(), "split": args.split,
               "device": str(scorer.device), "suite": args.suite, "threshold_z": thr, "enabled": scorer.f["enabled"], "n_images": n}, out / "run.json")
    log.info("robustness (%s, %s):\n%s", args.tag, args.split,
             res[["condition", "bpcer", "apcer", "attacks_missed", "auc", "median_bonafide", "median_attack"]].round(3).to_string(index=False))


def cmd_bench(args) -> None:
    """Inference cost of the frozen ensemble: seconds per image for each enabled member and in total, on this device (model loading included: the scorer loads each member per call)."""
    import platform
    import time

    from ..data.cache import cache_frame, load_cache
    from ..ensemble.scorer import EnsembleScorer
    from ..utils.device import resolve_device

    cfg = _make_cfg(args)
    log = _setup(cfg)
    device = resolve_device(cfg.device)
    scorer = EnsembleScorer(Path(args.frozen), device, args.batch_size)
    size = int(cfg.preprocess.size)
    crops = {}
    for crop in sorted(scorer.crops):
        c = load_cache(Path(cfg.data.cache_dir) / f"crops_{crop}_{size}_aligned.npz")
        f = cache_frame(c)
        crops[crop] = c["images"][((f.split == "val") & (f.label == 0)).to_numpy()][: args.n]
    rows = []
    for j, name in enumerate(scorer.f["enabled"], 1):
        log.info("latency: member %s (%d/%d), %d images", name, j, len(scorer.f["enabled"]), args.n)
        scorer.member_t(name, {k: v[:2] for k, v in crops.items()})  # warm-up (kernels, weights cached by the OS)
        t0 = time.perf_counter()
        scorer.member_t(name, crops)
        dt = time.perf_counter() - t0
        spec = scorer.f["members"][name]
        rows.append({"member": name, "models": len(spec.get("runs", [1])), "s_per_image": dt / args.n})
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"member": "ensemble", "models": int(res.models.sum()), "s_per_image": float(res.s_per_image.sum())}
    out = ensure_dir(Path(cfg.output_dir) / "evaluation" / "bench")
    res.to_csv(out / f"latency_{device.type}.csv", index=False)
    save_json({"device": str(device), "platform": platform.platform(), "n_images": args.n, "includes_model_loading": True,
               "note": "each call loads its member's weights, as the scorer does; per-image cost at batch scale is lower in a resident service"},
              out / f"latency_{device.type}.json")
    log.info("latency on %s (%d images, model loading included):\n%s", device, args.n, res.round(4).to_string(index=False))
