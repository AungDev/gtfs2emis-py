"""Sphinx configuration for gtfs2emis-py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "gtfs2emis-py"
author = "Aung"
copyright = "2026, Aung"
version = "0.1.0"
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "schemas/*"]
