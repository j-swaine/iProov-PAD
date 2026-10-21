"""`screenpad ensemble`: calibrate members on bona fide, score the enabled ensemble, and run the coalition ablation on dev.

Writes results/ensemble/: frozen_csv.json (enabled members, calibration from the training-time score files, combination rule, threshold: the template for `screenpad freeze`), member_scores.csv (calibrated z per member, val + dev), ablation.csv (every coalition), shapley.csv (per-member contribution), summary.json (the enabled ensemble's dev metrics and threshold).
"""

from __future__ import annotations

from pathlib import Path

from ..cli_common import _make_cfg, _setup
from ..utils import ensure_dir, save_json


def cmd_ensemble(args) -> None:
    import pandas as pd

    from .core import FAMILY, ablation, member_table, shapley

    cfg = _make_cfg(args)
    log = _setup(cfg)
    e = cfg.ensemble
    members = {k: dict(v) for k, v in dict(e.members).items()}
    roots = dict(e.roots)
    available = [k for k, v in members.items() if (Path(roots["runs"]) / v["runs"][0] / "scores_last.csv" if v["type"] == "run"
                                                 else Path(roots["oneclass"]) / v["path"] / "scores.csv").exists()]
    missing = sorted(set(members) - set(available))
    if missing:
        log.warning("no scores for %s; left out", missing)
    tab = member_table({k: members[k] for k in available}, roots)
    abl_members = [k for k in e.get("ablation_members", available) if k in available]
    if not set(k for k in available if members[k]["enabled"]) <= set(abl_members):
        raise ValueError("every enabled member must be in ensemble.ablation_members")
    out = ensure_dir(Path(cfg.output_dir) / str(e.get("dir", "ensemble")))
    tab.to_csv(out / "member_scores.csv", index=False)
    target = float(e.bpcer_target)
    families = {k: FAMILY[members[k]["type"]] for k in available}
    mode = str(e.get("combine", "family"))
    abl = ablation(tab, abl_members, families, target, int(cfg.seed), mode)
    abl.sort_values(["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).to_csv(out / "ablation.csv", index=False)
    # every coalition under each averaging rule, all reported: by family (shipped default) and by member
    for m in ("family", "mean"):
        (abl if m == mode else ablation(tab, abl_members, families, target, int(cfg.seed), m)).sort_values(
            ["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).to_csv(out / f"ablation_{m}.csv", index=False)
    sh = shapley(abl, abl_members)
    sh.to_csv(out / "shapley.csv", index=False)
    enabled = "+".join(k for k in available if members[k]["enabled"])
    row = abl[abl.coalition == enabled].iloc[0].to_dict()
    # the enabled coalition under each combination rule, for the report's comparison
    alt = pd.concat([ablation(tab, enabled.split("+"), families, target, int(cfg.seed), m).query("coalition == @enabled") for m in ("family", "mean", "max")])
    alt.to_csv(out / "combine_rules.csv", index=False)
    log.info("combination rules for %s:\n%s", enabled, alt[["combine", "dev_auc", "dev_bpcer@apcer0", "dev_apcer@thr", "dev_bpcer@thr"]].round(3).to_string(index=False))
    save_json({"enabled": enabled, "families": {k: families[k] for k in enabled.split("+")}, "bpcer_target": target, "n_coalitions": len(abl), **row}, out / "summary.json")
    # the frozen ensemble: everything needed to score new crops exactly as above (screenpad/ensemble/scorer.py)
    from .core import calibration_stats

    frozen = {"enabled": enabled.split("+"),
              "members": {k: {**members[k], "family": families[k], "calibration": calibration_stats(members[k], roots)} for k in available},
              "ablation_members": abl_members, "roots": roots, "hf_runs_repo": cfg.train.hf_runs_repo, "combine": mode, "bpcer_target": target, "threshold_z": row["threshold_z"]}
    save_json(frozen, out / "frozen_csv.json")  # template for `screenpad freeze`, which recalibrates through the scorer itself
    log.info("enabled ensemble %s: %s", enabled, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    log.info("top coalitions:\n%s", abl.sort_values(["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).head(10).round(3).to_string(index=False))
    log.info("member contributions:\n%s", sh.round(4).to_string(index=False))


def cmd_freeze(args) -> None:
    """Recalibrate the ensemble through the scorer itself on this device, so calibration and later scoring share code, device and precision.

    The training-time scores (CUDA, bf16) and re-scoring (e.g. MPS, fp32) differ by up to ~0.3 z, enough to flip a borderline image. Here the 240 calibration bona fide (val originals + camera-only variant 0) are re-scored by `EnsembleScorer`; each source's median and robust scale, and the threshold, come from those scores. Writes results/ensemble/frozen.json.
    """
    import json

    import numpy as np

    from ..data.cache import cache_frame, load_cache
    from ..utils.device import resolve_device
    from .core import combine, threshold
    from .scorer import EnsembleScorer

    cfg = _make_cfg(args)
    log = _setup(cfg)
    out = Path(cfg.output_dir) / str(cfg.ensemble.get("dir", "ensemble"))
    template = json.loads((out / "frozen_csv.json").read_text())
    device = resolve_device(cfg.device)
    sc = EnsembleScorer(template, device)
    pool = template.get("ablation_members", list(template["members"]))
    template["members"] = {k: v for k, v in template["members"].items() if k in template["enabled"] or (k in pool and sc.scorable(v))}
    size = int(cfg.preprocess.size)
    crops = {}
    for crop in sorted(sc.crops):
        real = load_cache(Path(cfg.data.cache_dir) / f"crops_{crop}_{size}_aligned.npz")
        f = cache_frame(real)
        a = real["images"][((f.split == "val") & (f.label == 0)).to_numpy()]
        del real
        syn = load_cache(Path(cfg.data.cache_dir) / f"synth_{crop}_{size}_train-val.npz")
        g = cache_frame(syn)
        b = syn["images"][((g.split == "val") & (g.kind == "camera_only") & (g.variant == 0)).to_numpy()]
        del syn
        crops[crop] = np.concatenate([a, b])
    n = len(next(iter(crops.values())))
    log.info("calibration set: %d bona fide (val originals + camera-only v0), device %s", n, device)
    z = {}
    for name, spec in template["members"].items():
        cal, zs = [], []
        for src, t in sc.member_t(name, crops).items():
            med = float(np.median(t))
            scale = max(float(1.4826 * np.median(np.abs(t - med))), 1e-9)
            cal.append({"source": src, "median": med, "scale": scale, "n_calibration": int(len(t))})
            zs.append((t - med) / scale)
        spec["calibration"] = cal
        z[name] = np.mean(zs, axis=0)
        log.info("%s: %s", name, [(c["source"], round(c["median"], 3), round(c["scale"], 3)) for c in cal])
    import pandas as pd

    families = {k: template["members"][k]["family"] for k in template["enabled"]}
    template["threshold_z"] = threshold(combine(pd.DataFrame(z), template["enabled"], families, template["combine"]), float(template["bpcer_target"]))
    template["calibrated_on"] = {"device": str(device), "n": int(n), "set": "val bona fide originals + camera-only variant 0"}
    save_json(template, out / "frozen.json")
    log.info("frozen ensemble %s, threshold z %.3f -> %s", template["enabled"], template["threshold_z"], out / "frozen.json")


def cmd_fetch_models(args) -> None:
    cfg = _make_cfg(args)
    log = _setup(cfg)
    fetch_models(Path(args.frozen), Path(args.dest), args.all_members, log)


def fetch_models(frozen_path: Path, dest: Path, all_members: bool = False, log=None) -> Path:
    """Download exactly the files the frozen ensemble needs from the public HF runs repo into `dest`, and write `dest`/frozen.json pointing at them (returned).

    Used by `screenpad infer` when the weights are not on disk, and by the Docker image (a volume holds the ~2 GB of weights). No token needed. Files kept: each run's config.yaml and ckpt/last.pt; each one-class member's bank.pt (and the run whose features it uses).
    """
    import json

    from huggingface_hub import snapshot_download

    frozen = json.loads(Path(frozen_path).read_text())
    names = frozen["enabled"] if not all_members else list(frozen["members"])
    patterns = []
    for n in names:
        spec = frozen["members"][n]
        if spec["type"] == "run":
            patterns += [f"runs/{r}/config.yaml" for r in spec["runs"]] + [f"runs/{r}/ckpt/last.pt" for r in spec["runs"]]
        else:
            patterns.append(f"runs/{spec['hf_run']}/{spec['path']}/bank.pt")
            spec["root"] = str(dest / "runs" / spec["hf_run"])
    for n in names:  # one-class members built on a fine-tuned backbone need that run's weights too
        spec = frozen["members"][n]
        if spec["type"] == "oneclass":
            import torch

            snapshot_download(frozen["hf_runs_repo"], allow_patterns=[f"runs/{spec['hf_run']}/{spec['path']}/bank.pt"], local_dir=dest)
            w = torch.load(Path(spec["root"]) / spec["path"] / "bank.pt", map_location="cpu", weights_only=True).get("weights_run")
            if w:
                patterns += [f"runs/{w}/config.yaml", f"runs/{w}/ckpt/last.pt"]
    snapshot_download(frozen["hf_runs_repo"], allow_patterns=sorted(set(patterns)), local_dir=dest)
    frozen["roots"] = {"runs": str(dest / "runs"), "oneclass": str(dest / "runs")}
    frozen["members"] = {n: frozen["members"][n] for n in names}
    frozen["enabled"] = [n for n in frozen["enabled"] if n in names]
    (dest / "frozen.json").write_text(json.dumps(frozen, indent=2))
    if log:
        log.info("fetched %d files for %s -> %s", len(set(patterns)), names, dest)
    return dest / "frozen.json"
