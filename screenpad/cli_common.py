"""Used by the per-package screenpad.<pkg>.cli modules."""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_config, CONFIG_DIR
from .utils import seed_everything, get_logger, ensure_dir, timestamp, limit_threads


def _make_cfg(args, model=None, extra_files=()) -> Config:
    """Resolve the run `Config` for a CLI invocation from `args` plus any caller-supplied extra config files.

    `extra_files` are package-relative defaults the subcommand wants merged in ahead of `args.extra_files`; bare filenames resolve against `CONFIG_DIR`. `args.overrides` (the `--set key=value` flags) are applied last. If `args` carries `run_group`/`device`, they overwrite the matching top-level keys.
    """
    resolved = []
    for f in extra_files:
        p = Path(f)
        resolved.append(p if p.is_absolute() else CONFIG_DIR / p.name)
    files = resolved + list(getattr(args, "extra_files", []))
    cfg = load_config(model=model, extra_files=files, overrides=args.overrides)
    if getattr(args, "run_group", None):
        cfg["run_group"] = args.run_group
    if getattr(args, "device", None):
        cfg["device"] = args.device
    return cfg


def _setup(cfg: Config, logfile=None):
    """Seed RNGs, cap CPU threads, and build the run logger. Device resolution is left to the commands that need torch."""
    seed_everything(int(cfg.seed), bool(cfg.get("deterministic", False)))
    limit_threads(int(cfg.get("num_threads", 6)))
    return get_logger("screenpad", logfile=logfile)


def _run_dir(cfg: Config, name: str, run_id=None) -> Path:
    """Create (if needed) and return `<cfg.output_dir>/<cfg.run_group>/<name>/<run_id>`; `run_id` defaults to a timestamp."""
    rid = run_id or timestamp()
    return ensure_dir(Path(cfg.output_dir) / cfg.run_group / name / rid)
