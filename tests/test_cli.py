"""Every CLI command must point at a handler that exists, without running it."""

from __future__ import annotations

import importlib

from screenpad.cli import build_parser


def test_cli_targets_resolve():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    for name, sp in sub.choices.items():
        fn = sp.get_default("func")
        module = importlib.import_module(fn.module_path)
        assert hasattr(module, fn.func_name), f"{name} -> {fn.module_path}.{fn.func_name} missing"
