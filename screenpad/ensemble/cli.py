"""`screenpad ensemble`: calibrate members on bona fide, score the enabled ensemble, and run the coalition ablation on dev.

Writes results/ensemble/: member_scores.csv (calibrated z per member, val + dev), ablation.csv (every coalition), shapley.csv (per-member contribution), summary.json (the enabled ensemble's dev metrics and threshold).
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
    out = ensure_dir(Path(cfg.output_dir) / "ensemble")
    tab.to_csv(out / "member_scores.csv", index=False)
    target = float(e.bpcer_target)
    families = {k: FAMILY[members[k]["type"]] for k in available}
    mode = str(e.get("combine", "family"))
    abl = ablation(tab, available, families, target, int(cfg.seed), mode)
    abl.sort_values(["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).to_csv(out / "ablation.csv", index=False)
    # every coalition under each averaging rule, all reported: by family (shipped default) and by member
    for m in ("family", "mean"):
        (abl if m == mode else ablation(tab, available, families, target, int(cfg.seed), m)).sort_values(
            ["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).to_csv(out / f"ablation_{m}.csv", index=False)
    sh = shapley(abl, available)
    sh.to_csv(out / "shapley.csv", index=False)
    enabled = "+".join(k for k in available if members[k]["enabled"])
    row = abl[abl.coalition == enabled].iloc[0].to_dict()
    # the enabled coalition under each combination rule, for the report's comparison
    alt = pd.concat([ablation(tab, enabled.split("+"), families, target, int(cfg.seed), m).query("coalition == @enabled") for m in ("family", "mean", "max")])
    alt.to_csv(out / "combine_rules.csv", index=False)
    log.info("combination rules for %s:\n%s", enabled, alt[["combine", "dev_auc", "dev_bpcer@apcer0", "dev_apcer@thr", "dev_bpcer@thr"]].round(3).to_string(index=False))
    save_json({"enabled": enabled, "families": {k: families[k] for k in enabled.split("+")}, "bpcer_target": target, "n_coalitions": len(abl), **row}, out / "summary.json")
    log.info("enabled ensemble %s: %s", enabled, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    log.info("top coalitions:\n%s", abl.sort_values(["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).head(10).round(3).to_string(index=False))
    log.info("member contributions:\n%s", sh.round(4).to_string(index=False))
