"""
Logger for runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "screenpad",
    logfile: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger (stdout, plus `logfile` if given). Repeat calls with the same name return the same logger without re-adding handlers."""
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger._configured = True
    return logger
