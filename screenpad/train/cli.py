"""`screenpad train`: one fine-tuning run, written to <runs_dir>/<name>/ and optionally pushed to the private HF runs repo.

Run folder contents: config.yaml (resolved), provenance.json (git, packages, GPU, cache hashes), SOURCE_DIFF.patch (on a VM: uncommitted changes the code was built from), bench.json (batch-size probe, if run), metrics.csv (per epoch), scores_best.csv / scores_last.csv (per image, val and dev), summary.json, ckpt/best.pt and ckpt/last.pt (weights), train.log, STATUS (read by the orchestrator).
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

from ..cli_common import _make_cfg, _setup
from ..config import REPO_ROOT, save_config


def cmd_train(args) -> None:
    import torch

    from ..utils.device import resolve_device
    from ..utils.provenance import write_provenance
    from .data import load_pool
    from .loop import Trainer, overfit_check, probe_batch_size

    cfg = _make_cfg(args)
    c = cfg.train
    run_dir = Path(c.runs_dir) / c.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup(cfg, logfile=run_dir / "train.log")
    status = run_dir / "STATUS"
    status.write_text(json.dumps({"state": "starting"}))
    try:
        save_config(cfg, run_dir / "config.yaml")
        size = int(cfg.preprocess.size)
        cache_dir = Path(cfg.data.cache_dir)
        caches = [cache_dir / f"crops_{c.crop}_{size}_aligned.npz", cache_dir / f"synth_{c.crop}_{size}_train-val.npz"]
        write_provenance(run_dir / "provenance.json", REPO_ROOT, caches, extra={"run": c.name, "seed": int(cfg.seed)})
        if (REPO_ROOT / "SOURCE_DIFF.patch").exists():  # on a VM: the uncommitted changes the code tarball was built from
            shutil.copy(REPO_ROOT / "SOURCE_DIFF.patch", run_dir / "SOURCE_DIFF.patch")
        device = resolve_device(cfg.device)
        images, table = load_pool(cache_dir, c.crop, size)
        trainer = Trainer(dict(c), images, table, device, int(cfg.seed), log)
        log.info("run %s: %s on %s crops at %d px, device %s; %d train faces, %d per epoch",
                 c.name, c.backbone, c.crop, int(c.input_size), device, len(trainer.sampler.faces), len(trainer.sampler(0)))
        batch_size = int(c.batch_size)
        if args.overfit:
            res = overfit_check(trainer, log=log)
            (run_dir / "overfit.json").write_text(json.dumps(res, indent=2))
            status.write_text(json.dumps({"state": "done" if res["passed"] else "failed", "overfit": res["passed"]}))
            log.info("overfit check %s", "PASSED" if res["passed"] else "FAILED")
            if not args.bench:
                return
        if args.bench or c.auto_batch:
            bench = probe_batch_size(trainer, log=log)
            (run_dir / "bench.json").write_text(json.dumps(bench, indent=2))
            if args.bench:
                status.write_text(json.dumps({"state": "done", "bench_only": True}))
                return
            batch_size = bench["chosen"] or batch_size
        log.info("batch size %d", batch_size)
        resume = Path(c.runs_dir) / "_resume" / f"{c.name}.pt"
        resume.parent.mkdir(parents=True, exist_ok=True)
        result = trainer.fit(run_dir, resume, batch_size)
        (run_dir / "summary.json").write_text(json.dumps({"run": c.name, "batch_size": batch_size, **result}, indent=2, default=float))
        status.write_text(json.dumps({"state": "trained"}))
        if args.push:
            from .hub import ensure_repo, upload_run, verify_run

            ensure_repo(c.hf_runs_repo)
            upload_run(run_dir, c.hf_runs_repo, f"runs/{c.name}")
            problems = verify_run(run_dir, c.hf_runs_repo, f"runs/{c.name}")
            if problems:
                raise RuntimeError(f"upload verification failed: {problems[:5]}")
            log.info("pushed and verified -> %s/runs/%s", c.hf_runs_repo, c.name)
        status.write_text(json.dumps({"state": "done", "pushed": bool(args.push)}))
        log.info("final: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["final"].items()})
    except Exception:
        status.write_text(json.dumps({"state": "failed", "error": traceback.format_exc()[-4000:]}))
        log.exception("run failed")
        raise
