"""`screenpad audit`."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..cli_common import _make_cfg, _setup
from .leakage import run_checks


def cmd_audit(args) -> None:
    cfg = _make_cfg(args)
    log = _setup(cfg)
    splits = pd.read_csv(Path(cfg.data.splits), dtype={"id": str})
    failed = False
    for name, bad in run_checks(splits).items():
        if len(bad):
            failed = True
            log.error("FAIL %s: %d rows, e.g. ids %s", name, len(bad), bad["id"].unique()[:5].tolist())
        else:
            log.info("ok   %s", name)
    if failed:
        sys.exit(1)

