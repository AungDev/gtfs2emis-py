"""Packaging and licence invariants."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

import gtfs2emis_py

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"


def test_package_docstring_and_metadata_are_mit() -> None:
    assert "AGPL" not in (gtfs2emis_py.__doc__ or "")
    assert "GNU Affero" not in (gtfs2emis_py.__doc__ or "")
    text = PYPROJECT.read_text()
    assert 'license = "MIT"' in text
    assert "License :: OSI Approved :: MIT License" in text


def test_sdist_excludes_fetch_only_data(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the sdist")
    assert uv is not None
    result = subprocess.run(
        [uv, "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"uv build failed:\n{result.stdout}\n{result.stderr}")
    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1, archives
    with tarfile.open(archives[0]) as archive:
        names = archive.getnames()

    assert not [name for name in names if name.endswith(".rda")]
    for excluded in (
        "brazil_cetesb.parquet",
        "usa_emfac.parquet",
        "usa_moves.parquet",
    ):
        assert not [name for name in names if name.endswith(excluded)]
    for required in ("europe_emep.parquet", "manifest.json", "NOTICE"):
        assert any(name.endswith(required) for name in names), required
    assert any("/LICENSES/" in name for name in names)
