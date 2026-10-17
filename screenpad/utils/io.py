"""
I/O utilities and general sanitisation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

pathlike = str | Path | None


def ensure_dir(path: pathlike) -> Path:
    """Ensure a directory exists, creating it if necessary. Return the Path object."""
    if path is None:
        raise ValueError("Path cannot be None")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Return a timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# json accepts a cls argument so we can convert numpy types to python types for easy serialisation. Format suggested in stack overflow forum https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types for serialization."""
    # This exists because json does not know how to serialise numpy types
    # Called as a literal method

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)  # fallback to default encoder of parent class


def save_json(obj: Any, path: pathlike, indent: int = 2) -> None:
    """Save an object to a JSON file, creating parent directories if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        # the default method is called automatically.
        json.dump(obj, fh, indent=indent, cls=_NumpyEncoder)


def load_json(path: pathlike) -> Any:
    """Load an object from a JSON file."""
    with open(path, "r") as fh:
        return json.load(fh)
