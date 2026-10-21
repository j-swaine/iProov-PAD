"""
`screenpad mlflow-sync`: replay finished results into a local MLflow store for browsing and comparison.
-------------------------------------------------------------------------------------------------------

The run folders (config.yaml, provenance.json, metrics.csv, summary.json) stay the source of truth; this only mirrors them, so training needs no MLflow and Colab VMs no tracking server. Re-running replaces each mirrored run, so the store always matches the folders.

Experiments:
- `finetune`: one run per training run folder. Params are the `train.*` config and seed; metrics are per epoch (step = epoch), plus final_* and, if scored, the shrink test (results/shortcut/shrink_summary.csv). Tags record git commit, GPU and the HF location of the checkpoints, which are not copied.
- `oneclass`: one run per results/oneclass/<member>/summary.json.
- `ensemble`: the enabled ensemble's metrics, per-member Shapley contributions, and ablation.csv as an artifact.

View with: `uv run --extra torch mlflow ui --backend-store-uri sqlite:///results/mlflow.db --port 5001 --workers 1`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..cli_common import _make_cfg, _setup


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _num(d: dict) -> dict:
    return {k.replace("@", "_at_"): float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}


ARTIFACTS = Path("results/mlartifacts")
EXPERIMENTS = {
    "finetune": "Fine-tuned classifiers (one run per config x seed). Trained on synthetic attacks only; dev = 6 real attacks + 126 bona fide. "
                "Compare final_dev_auc / final_dev_bpcer_at_apcer0 across runs; the Chart view shows per-epoch curves; shrink_pct_at_* is the resampling shortcut test (50 = no effect).",
    "oneclass": "One-class scorers (PatchCore, Mahalanobis) fitted on train bona fide only; no attacks used.",
    "ensemble": "Bona-fide-calibrated, equal-weight ensemble of the enabled members. ablation.csv (artifact) scores every coalition on dev; shapley_* is each member's average contribution to dev AUC.",
}


def _fresh_run(mlflow, experiment: str, name: str, note: str = ""):
    if mlflow.get_experiment_by_name(experiment) is None:  # artifacts beside the database, under results/ (git-ignored)
        mlflow.create_experiment(experiment, artifact_location=(ARTIFACTS / experiment).resolve().as_uri())
    exp = mlflow.set_experiment(experiment)
    # MLflow 3 opens untagged experiments in its GenAI (traces) view; this selects the classic runs/metrics view
    mlflow.MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.experimentKind", "custom_model_development")
    mlflow.MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.note.content", EXPERIMENTS[experiment])
    for r in mlflow.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{name}'", output_format="list"):
        mlflow.delete_run(r.info.run_id)
    return mlflow.start_run(run_name=name, description=note or None)


def cmd_mlflow_sync(args) -> None:
    import mlflow

    cfg = _make_cfg(args)
    log = _setup(cfg)
    out = Path(cfg.output_dir)
    global ARTIFACTS
    ARTIFACTS = out / "mlartifacts"
    mlflow.set_tracking_uri(args.uri)
    shrink = pd.read_csv(out / "shortcut" / "shrink_summary.csv").set_index("run") if (out / "shortcut" / "shrink_summary.csv").exists() else None

    n = 0
    for runs_dir in args.runs_dirs:
        for d in sorted(Path(runs_dir).glob("*/summary.json")):
            d = d.parent
            conf = yaml.safe_load((d / "config.yaml").read_text())
            prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").exists() else {}
            summ = json.loads((d / "summary.json").read_text())
            f = summ["final"]
            note = (f"dev AUC {f['dev_auc']:.3f} [{f['dev_auc_lo95']:.3f}, {f['dev_auc_hi95']:.3f}], BPCER@APCER0 {f['dev_bpcer@apcer0']:.1%}, "
                    f"synthetic val AUC {f['val_auc']:.3f}, camera-vs-original AUC {f['camera_vs_original_auc']:.2f}")
            with _fresh_run(mlflow, "finetune", d.name, note):
                mlflow.log_params({**_flat(conf["train"], "train."), "seed": conf["seed"]})
                git = prov.get("git", {})
                mlflow.set_tags({"config_group": d.name.rsplit("_s", 1)[0], "source_version": str(git.get("commit") or git.get("source_version") or ""),
                                 "gpu": str(prov.get("torch", {}).get("device_name", "")), "torch": str(prov.get("torch", {}).get("version", "")),
                                 "checkpoints": f"hf://{conf['train']['hf_runs_repo']}/runs/{d.name}/ckpt"})
                for row in summ["history"]:
                    mlflow.log_metrics(_num({k: v for k, v in row.items() if k != "epoch"}), step=int(row["epoch"]))
                mlflow.log_metrics(_num({f"final_{k}": v for k, v in summ["final"].items()}))
                if shrink is not None and d.name in shrink.index:
                    mlflow.log_metrics(_num(shrink.loc[d.name].to_dict()))
                for f in ("config.yaml", "summary.json", "provenance.json", "metrics.csv"):
                    if (d / f).exists():
                        mlflow.log_artifact(str(d / f))
            n += 1
    log.info("finetune: %d runs", n)

    for s in sorted((out / "oneclass").glob("*/summary.json")):
        summ = json.loads(s.read_text())
        m = summ["metrics"]
        with _fresh_run(mlflow, "oneclass", s.parent.name, f"dev AUC {m['dev_auc']:.3f}, BPCER@APCER0 {m['dev_bpcer@apcer0']:.1%}, synthetic val AUC {m['val_auc']:.3f}"):
            mlflow.log_params({k: v for k, v in summ.items() if k != "metrics"})
            mlflow.log_metrics(_num(summ["metrics"]))
            mlflow.log_artifact(str(s))
        log.info("oneclass: %s", s.parent.name)

    ens = out / "ensemble"
    if (ens / "summary.json").exists():
        summ = json.loads((ens / "summary.json").read_text())
        with _fresh_run(mlflow, "ensemble", summ["enabled"], f"dev AUC {summ['dev_auc']:.3f} [{summ['dev_auc_lo95']:.3f}, {summ['dev_auc_hi95']:.3f}], "
                        f"BPCER@APCER0 {summ['dev_bpcer@apcer0']:.1%}; at the {summ['bpcer_target']:.0%} bona fide threshold: APCER {summ['dev_apcer@thr']:.1%}, BPCER {summ['dev_bpcer@thr']:.1%}"):
            mlflow.log_params({"enabled": summ["enabled"], "combine": summ.get("combine", "mean"), "bpcer_target": summ["bpcer_target"], "n_coalitions": summ["n_coalitions"]})
            mlflow.log_metrics(_num({k: v for k, v in summ.items() if k not in ("enabled", "families", "combine", "bpcer_target", "n_coalitions", "coalition", "size")}))
            sh = pd.read_csv(ens / "shapley.csv")
            mlflow.log_metrics({f"shapley_{m}": float(v) for m, v in zip(sh.member, sh.shapley_dev_auc)})
            for f in ("ablation.csv", "shapley.csv", "combine_rules.csv", "summary.json"):
                mlflow.log_artifact(str(ens / f))
        log.info("ensemble: %s", summ["enabled"])
    log.info("view: uv run --extra torch mlflow ui --backend-store-uri %s --port 5001 --workers 1", args.uri)
