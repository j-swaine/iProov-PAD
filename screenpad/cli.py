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
    sp.add_argument("--block", type=int, default=8, help="DINOv2 block whose patch tokens PatchCore uses (of 12)")
    sp.add_argument("--coreset", type=float, default=0.1, help="fraction of bona fide patch features kept in the memory bank")
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
