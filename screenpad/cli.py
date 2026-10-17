"""
Command-line interface - the single entry point for the whole pipeline.
========================================================================

Run as ``screenpad <command>`` or ``python -m screenpad <command>``.

Commands
--------
    pull-data       download the dataset from the Hugging Face Hub
    manifest        scan the data folder and write the per-image manifest
    splits          assign every image ID to train/val/test
    audit           run the leakage checks on the split table

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

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
