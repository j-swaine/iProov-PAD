"""
Command-line interface - the single entry point for the whole pipeline.
========================================================================

Run as ``screenpad <command>`` or ``python -m screenpad <command>``.

Commands
--------
    pull-data       download the dataset (raw tar, or --cache for crops) from the Hugging Face Hub
    cache           build + verify the preprocessed face-crop caches (--push to upload)
    manifest        scan the data folder and write the per-image manifest
    splits          assign every image ID to train/val/test
    audit           run the leakage checks on the split table
    eda             face detection table, attack/twin grid and spectra
    baseline-minifasnet   off-the-shelf anti-spoofing model, native and context-matched
    synth           synthetic screen-recapture attacks (and camera-only controls) as crop caches
    synth-gallery   contact sheets of synthetic attacks next to real crops, for visual checking
    probe           frozen-backbone linear probes: train on synthetic attacks, measure on real dev attacks
    train           fine-tune one backbone (--bench: batch-size probe only; --push: upload + verify on HF)
    oneclass        PatchCore + Mahalanobis: fitted on bona fide only, scored on val/dev/shrink sets
    ensemble        bona-fide-calibrated equal-weight ensemble + coalition ablation (Shapley, leave-one-out)
    mlflow-sync     mirror results into a local MLflow store (view: uv run mlflow ui --backend-store-uri sqlite:///results/mlflow.db)
    explain         face-vs-surroundings, fine-texture and occlusion tests (numbers + figures)
    freeze          recalibrate the enabled ensemble through its scorer on this device (calibration and test share code and precision)
    fetch-models    download the frozen ensemble's weights from HF into a folder (Docker volume)
    infer           score a folder of raw face images -> CSV (face detection, crop, frozen ensemble)
    preflight       check a machine before scoring: environment, weights, face detector, golden-score test, input scan
    evaluate        frozen-ensemble evaluation of dev (rehearsal) or test (one-off, guarded)
    robustness      BPCER and APCER under benign degradations (val, dev or test images)
    bench-inference seconds per image of the frozen ensemble, per member
    ssl-adapt       continue DINOv2 self-supervision on bona fide only (domain-adapted features for PatchCore)
    shortcut-test   resampling counterfactual: does a run's score move when only the crop's resampling changes?

Examples
--------
    screenpad manifest
    screenpad splits --set split.test_extra_bonafide=150
    screenpad audit

Commands live in the sub-package they call (data/cli.py, audit/cli.py, ...); this file wires them to argparse. Each command's module is imported only when that command runs, so a heavy or broken dependency only affects the commands that use it.
"""

from __future__ import annotations

import argparse
import importlib


def _lazy(module_path: str, func_name: str):
    """Return a callable that imports ``module_path`` and calls ``func_name`` only when invoked, not at CLI startup."""

    def _call(args):
        module = importlib.import_module(module_path)
        return getattr(module, func_name)(args)

    _call.module_path = module_path
    _call.func_name = func_name
    return _call


def _common(parser: argparse.ArgumentParser) -> None:
    """Add the shared `--set`/`--config`/`--run-group`/`--run-id`/`--device` flags to `parser`, in place."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="k.sub=val",
        help="override any config value (repeatable)",
    )
    parser.add_argument(
        "--config",
        dest="extra_files",
        action="append",
        default=[],
        metavar="FILE.yaml",
        help="extra YAML to merge (repeatable)",
    )
    parser.add_argument("--run-group", default=None, help="group runs under results/<group>/")
    parser.add_argument("--run-id", default=None, help="run id (default: timestamp)")
    parser.add_argument("--device", default=None, help="auto|cuda|cpu|mps")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser, with every subcommand wired to its `_lazy`-imported handler."""
    p = argparse.ArgumentParser(
        prog="screenpad",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pull-data", help="download the dataset from the HF Hub")
    _common(sp)
    sp.add_argument("--cache", action="store_true", help="download the preprocessed crop caches instead of the raw tar")
    sp.set_defaults(func=_lazy("screenpad.data.cli", "cmd_pull"))

    sp = sub.add_parser("manifest", help="write the per-image manifest")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.data.cli", "cmd_manifest"))

    sp = sub.add_parser("splits", help="assign image IDs to train/val/test")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.data.cli", "cmd_splits"))

    sp = sub.add_parser("audit", help="leakage checks on the split table")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.audit.cli", "cmd_audit"))

    sp = sub.add_parser("cache", help="build + verify the preprocessed crop caches")
    _common(sp)
    sp.add_argument("--push", action="store_true", help="upload the caches to the HF dataset repo")
    sp.set_defaults(func=_lazy("screenpad.data.cli", "cmd_cache"))

    sp = sub.add_parser("baseline-minifasnet", help="off-the-shelf MiniFASNet (Apache-2.0) scores and metrics")
    _common(sp)
    sp.add_argument("--splits", default="dev,val", help="comma-separated splits to score (default: dev,val)")
    sp.set_defaults(func=_lazy("screenpad.baselines.cli", "cmd_minifasnet"))

    sp = sub.add_parser("synth", help="synthetic recapture attacks as crop caches")
    _common(sp)
    sp.add_argument("--splits", default=None, help="comma-separated source splits (default: synth.splits; never test)")
    sp.add_argument("--variants", type=int, default=None, help="synthetic attacks per source image (default: synth.variants)")
    sp.add_argument("--push", action="store_true", help="upload the caches to the HF dataset repo")
    sp.set_defaults(func=_lazy("screenpad.synth.cli", "cmd_synth"))

    sp = sub.add_parser("synth-gallery", help="contact sheets of synthetic vs real crops")
    _common(sp)
    sp.add_argument("--n", type=int, default=24, help="train identities to show")
    sp.set_defaults(func=_lazy("screenpad.synth.gallery", "cmd_gallery"))

    sp = sub.add_parser("probe", help="frozen-backbone linear probes")
    _common(sp)
    sp.add_argument("--backbones", default="handcrafted,dinov2_s224,dinov2_b224,dinov2_b392,clip_b16")
    sp.add_argument("--crops", default="tight,loose")
    sp.set_defaults(func=_lazy("screenpad.probes.cli", "cmd_probe"))

    sp = sub.add_parser("train", help="fine-tune one backbone on synthetic attacks")
    _common(sp)
    sp.add_argument("--bench", action="store_true", help="only run the batch-size probe and write bench.json")
    sp.add_argument("--overfit", action="store_true", help="only run the overfit check (tiny set, no augmentation) and write overfit.json")
    sp.add_argument("--push", action="store_true", help="upload the run folder to the HF runs repo and verify it")
    sp.set_defaults(func=_lazy("screenpad.train.cli", "cmd_train"))

    sp = sub.add_parser("oneclass", help="PatchCore and Mahalanobis one-class scorers fitted on train bona fide")
    _common(sp)
    sp.add_argument("--members", default="patchcore,mahalanobis")
    sp.add_argument("--crop", default="loose")
    sp.add_argument("--input-size", type=int, default=224)
    sp.add_argument("--arch", default="dinov2_b", choices=["dinov2_b", "dinov2_l", "wrn50"], help="frozen feature backbone (see oneclass/models.py ARCHS)")
    sp.add_argument("--block", type=int, default=8, help="DINOv2 block whose patch tokens PatchCore uses (of 12 for base, 24 for large; ignored for wrn50)")
    sp.add_argument("--coreset", type=float, default=0.1, help="fraction of bona fide patch features kept in the memory bank")
    sp.add_argument("--weights-run", default=None, help="take features from this fine-tuned run's backbone (e.g. A_dinov2b_loose224_s42) instead of pretrained DINOv2")
    sp.add_argument("--ead-size", type=int, default=256, help="EfficientAD input size (the paper's 256 by default)")
    sp.add_argument("--steps", type=int, default=70000, help="EfficientAD training steps at batch size 1 (the paper's 70000)")
    sp.add_argument("--imagenet-dir", default="results/cache/imagenette", help="where anomalib downloads Imagenette for EfficientAD's penalty loss")
    sp.add_argument("--workers", type=int, default=4, help="dataloader workers for EfficientAD")
    sp.add_argument("--probe-workers", default="", help="EfficientAD: comma-separated dataloader worker counts to trial for 300 steps first; the fastest is used (bench.json)")
    sp.add_argument("--smoke", action="store_true", help="EfficientAD: use a tiny stand-in for Imagenette (local smoke tests only)")
    sp.add_argument("--run-name", default=None, help="write into <runs_dir>/<name>/ with a STATUS file (for scripts/orchestrate.py)")
    sp.add_argument("--push", action="store_true", help="with --run-name: upload and verify the folder on the HF runs repo")
    sp.set_defaults(func=_lazy("screenpad.oneclass.cli", "cmd_oneclass"))

    sp = sub.add_parser("ensemble", help="calibrate members on bona fide, score the enabled ensemble, ablate every coalition on dev")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.ensemble.cli", "cmd_ensemble"))

    sp = sub.add_parser("mlflow-sync", help="mirror finished run folders, one-class and ensemble results into a local MLflow store")
    _common(sp)
    sp.add_argument("--uri", default="sqlite:///results/mlflow.db", help="MLflow tracking URI")
    sp.add_argument("--runs-dirs", nargs="*", default=["results/sweep/hub/runs", "results/runs"], help="folders holding training run folders")
    sp.set_defaults(func=_lazy("screenpad.utils.tracking", "cmd_mlflow_sync"))

    sp = sub.add_parser("explain", help="face-vs-surroundings, fine-texture and occlusion tests on dev for the enabled ensemble members")
    _common(sp)
    sp.add_argument("--crop", default="loose")
    sp.add_argument("--cells", type=int, default=8, help="occlusion grid size (cells per side)")
    sp.set_defaults(func=_lazy("screenpad.explain.cli", "cmd_explain"))

    sp = sub.add_parser("freeze", help="recalibrate the ensemble through its own scorer on this device -> results/ensemble/frozen.json")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.ensemble.cli", "cmd_freeze"))

    sp = sub.add_parser("infer", help="score a folder of raw face images with the frozen ensemble -> CSV (weights downloaded on first use)")
    _common(sp)
    sp.add_argument("--input", required=True, help="folder of images (png, jpg, ...)")
    sp.add_argument("--output", default="scores.csv", help="CSV to write")
    sp.add_argument("--frozen", default="artefacts/frozen_photo.json", help="the ensemble to score with (default: the deployed photometric variant; artefacts/frozen.json is the pre-registered, test-evaluated system)")
    sp.add_argument("--models", default="models", help="where weights are downloaded if the frozen ensemble's own paths have none")
    sp.add_argument("--recursive", action="store_true", help="also score images in subfolders")
    sp.add_argument("--chunk", type=int, default=256, help="images cropped and scored per chunk (bounds memory)")
    sp.add_argument("--batch-size", type=int, default=16, help="images per forward pass (lower = less memory)")
    sp.set_defaults(func=_lazy("screenpad.infer", "cmd_infer"))

    sp = sub.add_parser("preflight", help="check a machine before scoring: environment, ensemble, weights, face detector, golden-score test, optional input scan")
    _common(sp)
    sp.add_argument("--frozen", default="artefacts/frozen_photo.json", help="the ensemble to check (default: the deployed one)")
    sp.add_argument("--models", default="models", help="where weights are (or are downloaded to)")
    sp.add_argument("--no-download", action="store_true", help="fail instead of downloading missing weights")
    sp.add_argument("--input", default=None, help="optional: folder to scan for formats and problem files before scoring")
    sp.add_argument("--recursive", action="store_true")
    sp.add_argument("--reference", default="artefacts/preflight_reference.json", help="golden scores recorded for the ensemble")
    sp.add_argument("--tolerance", type=float, default=0.1, help="largest allowed difference from the reference scores, in z (cross-platform float differences are up to ~0.03 z; wrong weights shift scores by whole units)")
    sp.add_argument("--write-reference", action="store_true", help="record the golden scores on this machine (author only)")
    sp.add_argument("--batch-size", type=int, default=16)
    sp.set_defaults(func=_lazy("screenpad.preflight", "cmd_preflight"))

    sp = sub.add_parser("fetch-models", help="download the frozen ensemble's weights from the public HF runs repo")
    _common(sp)
    sp.add_argument("--frozen", default="artefacts/frozen_photo.json", help="the ensemble whose weights to fetch (default: the deployed photometric variant)")
    sp.add_argument("--dest", default="models", help="folder to download into; a frozen.json pointing at it is written there")
    sp.add_argument("--all-members", action="store_true", help="also fetch disabled members (for a descriptive ablation)")
    sp.set_defaults(func=_lazy("screenpad.ensemble.cli", "cmd_fetch_models"))

    sp = sub.add_parser("evaluate", help="score a split with the frozen ensemble: metrics with exact CIs, per phone, per skin tone, DET, descriptive ablation")
    _common(sp)
    sp.add_argument("--split", choices=["dev", "test"], default="dev", help="dev = rehearsal; test = the one-off final evaluation (guarded)")
    sp.add_argument("--frozen", default="results/ensemble/frozen.json")
    sp.add_argument("--confirm-test", action="store_true", help="required for --split test")
    sp.add_argument("--tag", default="frozen", help="system name: 'frozen' writes results/evaluation/<split>, others results/evaluation/<tag>/<split>")
    sp.add_argument("--force", action="store_true", help="(kept for compatibility; re-runs are allowed and recorded)")
    sp.add_argument("--allow-dirty", action="store_true", help="(kept for compatibility; an uncommitted tree is allowed and recorded)")
    sp.set_defaults(func=_lazy("screenpad.eval.cli_evaluate", "cmd_evaluate"))

    sp = sub.add_parser("robustness", help="BPCER and APCER under benign degradations (JPEG, resolution, light, blur, noise)")
    _common(sp)
    sp.add_argument("--frozen", default="results/ensemble/frozen.json")
    sp.add_argument("--split", choices=["val", "dev", "test"], default="dev", help="val: val bona fide only; dev: + the 6 dev attacks and twins; test: the test split")
    sp.add_argument("--tag", default="frozen", help="name of the system being tested: results go to results/evaluation/robustness/<tag>/<split>/")
    sp.add_argument("--chunk", type=int, default=40, help="images degraded and scored at a time (bounds memory)")
    sp.add_argument("--suite", choices=["degradations", "edge"], default="degradations", help="edge: flatten the crop's outer 5/10/15%% instead (border test)")
    sp.add_argument("--batch-size", type=int, default=64, help="images per forward pass (lower = less memory)")
    sp.set_defaults(func=_lazy("screenpad.eval.cli_evaluate", "cmd_robustness"))

    sp = sub.add_parser("bench-inference", help="seconds per image of the frozen ensemble, per member, on this device")
    _common(sp)
    sp.add_argument("--frozen", default="results/ensemble/frozen.json")
    sp.add_argument("--n", type=int, default=64, help="images to time")
    sp.add_argument("--batch-size", type=int, default=64, help="images per forward pass (lower = less memory)")
    sp.set_defaults(func=_lazy("screenpad.eval.cli_evaluate", "cmd_bench"))

    sp = sub.add_parser("ssl-adapt", help="self-supervised domain adaptation of DINOv2-base on train bona fide only (for a one-class member)")
    _common(sp)
    sp.add_argument("--run-name", required=True)
    sp.add_argument("--crop", default="loose")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--n-local", type=int, default=6, help="local (98 px) crops per image")
    sp.add_argument("--lr", type=float, default=5e-5, help="backbone peak learning rate (head 10x)")
    sp.add_argument("--head-warmup", type=int, default=1, help="epochs training only the new head before the backbone is unfrozen")
    sp.add_argument("--grad-ckpt", action="store_true", help="activation (gradient) checkpointing: less GPU memory, ~30%% slower")
    sp.add_argument("--no-ckpt", action="store_true", help="smoke tests: do not save weights")
    sp.add_argument("--push", action="store_true")
    sp.set_defaults(func=_lazy("screenpad.oneclass.cli", "cmd_ssl_adapt"))

    sp = sub.add_parser("shortcut-test", help="resampling counterfactual: build the resized bona fide crops and/or score runs on them")
    _common(sp)
    sp.add_argument("--build", action="store_true", help="build shrink_{crop}_{size}_aligned.npz for every crop variant")
    sp.add_argument("--push", action="store_true", help="with --build: upload the caches to the HF dataset")
    sp.add_argument("--runs", nargs="*", default=[], help="run folders (config.yaml + ckpt/last.pt) to score")
    sp.add_argument("--control", action="store_true", help="positive control: can frozen DINOv2 / handcrafted features tell resized crops from originals?")
    sp.set_defaults(func=_lazy("screenpad.eval.cli", "cmd_shortcut_test"))

    sp = sub.add_parser("eda", help="face detection table, attack/twin grid and spectra")
    _common(sp)
    sp.set_defaults(func=_lazy("screenpad.eda.cli", "cmd_eda"))

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
