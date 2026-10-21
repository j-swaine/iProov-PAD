"""
Full model configuration
------------------------

Resolve a model configuration from the yaml files in ``configs/`` and any extra files or CLI overrides, returning a dict with attribute access.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from collections.abc import Mapping

import yaml

_INTERP = re.compile(r"\$\{([^}]+)\}")

# Repository root = .../PAD (two levels up from this file).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = REPO_ROOT / "configs"

# This class meets the Mapping requirements as it inherits from dict which has __getitem__, __iter__, and __len__ methods.

# we inherit from dict to later utilise the __init__ method which will excepts the yaml kwargs and concert them into a dict.


class Config(dict):
    """A dict with attribute access and recursive wrapping.

    ``cfg.data.root`` is equivalent to ``cfg["data"]["root"]``.
    """

    # allow access by attribute, e.g. cfg.data.root

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return (
            Config(value)
            if isinstance(value, dict) and not isinstance(value, Config)
            else value
        )

    # allow setting by attribute, e.g. cfg.data.root = "foo"

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_plain(self) -> Dict[str, Any]:
        """Return a plain (json-serialisable) nested dict."""
        return json.loads(json.dumps(self, default=str))  # dumps is dump string.


# merging and override helpers
# this is necessary because a shallow merge would replace a whole nested object instead of just swapping the keys
def deep_merge(base: Mapping, override: Mapping) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict. Values in ``override`` take precedence."""
    out: Dict[str, Any] = copy.deepcopy(
        dict(base)
    )  # we use deepcopy so that we don't point at the same object in memory
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# more robustness for CLI overrides


def _coerce_scalar(text: str) -> Any:
    """Infer the type of a CLI override value ('3' -> int, 'true' -> bool, ...)."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    # Comma lists -> list of coerced scalars (e.g. "0.5,9").
    if "," in text:
        return [_coerce_scalar(part) for part in text.split(",")]
    return text


def apply_overrides(
    cfg: dict[str, Any], overrides: Optional[Iterable[str]]
) -> dict[str, Any]:
    """Apply ``--set a.b=c`` style overrides in-place-ish (returns new dict)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value form.")
        # split on the first =
        dotted, raw = item.split("=", 1)
        # walk the dotted path, creating intermediate empty dicts as needed and set the final key to the coerced scalar val.
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_scalar(raw)
    return out


# loaders
def _lookup(root: Mapping, dotted: str) -> Any:
    """Look up a dotted path in a nested mapping, raising KeyError if any level is missing. Returns the value at the final key."""
    node: Any = root
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def resolve_interpolations(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve `${...}` interpolations in a config dictionary.
    The _walk function recursively traverses the config dictionary and replaces any string containing `${...}` with the corresponding value from the config.
    """
    env_root = os.environ.get("SCREENPAD_DATA_ROOT")
    if env_root and "data" in cfg and isinstance(cfg["data"], dict):
        cfg["data"]["root"] = env_root

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            # ternary operator to check for "${" in node
            # sub takes a function as a first argument and calls it internally for each match, where m is the match object and m.group(1) is the first group in the regex pattern, which is the dotted path inside ${...}
            return (
                # m is a re.Match object that we don't construct, it is passed by regex to our function automatically. m.group(1) is whatever text the pattern's parenthesised capture group matched
                _INTERP.sub(lambda m: str(_lookup(cfg, m.group(1))), node)
                if "${" in node
                else node
            )  # parentheses are only here to let the expression span multiple lines.
        return node

    # A few passes resolve nested/chained interpolations; bail out when stable.
    for _ in range(5):
        new = _walk(cfg)
        if new == cfg:
            break
        cfg = new
    return cfg


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, "r") as fh:
        # yaml is a mapping therefore it loads as a dict.
        return yaml.safe_load(fh) or {}


def load_config(
    model: Optional[str] = None,
    *,
    extra_files: Optional[Iterable[Path]] = None,
    overrides: Optional[Iterable[str]] = None,
    config_dir: Path = CONFIG_DIR,
) -> Config:
    """Build the fully-resolved config for a run. Returns a dict with attribute access via a Config object."""
    merged: Dict[str, Any] = {}
    for name in ("default.yaml", "data.yaml", "synth.yaml", "train.yaml", "ensemble.yaml"):
        path = config_dir / name
        if path.exists():
            merged = deep_merge(merged, load_yaml(path))

    if model:
        model_path = config_dir / "models" / f"{model}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No config for model '{model}' at {model_path}. "
                f"Available: {[p.stem for p in (config_dir / 'models').glob('*.yaml')]}"
            )
        merged = deep_merge(merged, load_yaml(model_path))

    for extra in extra_files or []:
        merged = deep_merge(merged, load_yaml(Path(extra)))

    merged = apply_overrides(merged, overrides)
    merged = resolve_interpolations(merged)
    # Record provenance so a saved config explains itself.
    merged.setdefault("_meta", {})
    merged["_meta"].update(model=model, overrides=list(overrides or []))
    return Config(merged)


def save_config(cfg: Mapping, path: Path) -> None:
    """Persist the resolved config beside a run (YAML, human-diffable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = (
        cfg.to_plain()
        if isinstance(cfg, Config)
        else json.loads(json.dumps(cfg, default=str))
    )
    with open(path, "w") as fh:
        yaml.safe_dump(plain, fh, sort_keys=False)
