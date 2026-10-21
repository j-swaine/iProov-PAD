"""Sphinx configuration for the screenpad documentation.

Build (from the repository root):
    uv run --extra torch --group docs sphinx-build docs docs/_build/html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "screenpad"
author = "Jamie Swaine"
copyright = "2026, Jamie Swaine"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "_autosummary/**/.*", "report/**", "report_notes.md"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_mock_imports = ["anomalib", "lightning", "mlflow"]  # optional extras, not needed to read the API

html_theme = "furo"
html_title = "screenpad"
