"""Version consistency between packaging metadata and the package."""

from __future__ import annotations

from pathlib import Path

try:  # tomllib is stdlib from Python 3.11; the dev extra pulls tomli on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the 3.10 CI job
    import tomli as tomllib

import gtfs2emis_py

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    project = tomllib.loads(PYPROJECT.read_text())
    assert gtfs2emis_py.__version__ == project["project"]["version"]
