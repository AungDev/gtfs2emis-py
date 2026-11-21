"""Shared pytest fixtures for the GTFS reader tests.

Provides:
- :func:`build_minimal_feed`: build a tiny in-memory ``gtfs_kit.Feed`` with the
  minimum tables the transport model needs. Used for fast unit tests that do
  not depend on the bundled sample feed.
- :func:`write_minimal_feed_zip`: write such a feed to a zip on disk, for tests
  that exercise the path-reading branch.
- :func:`det_gtfs_path`: path to the bundled Detroit sample feed, for
  integration tests. Skipped if the feed is not present (e.g. fresh checkout
  before ``tests/data`` is populated).
"""

from __future__ import annotations

import os
from pathlib import Path

import gtfs_kit as gk
import pytest

from tests._fixtures import build_minimal_feed, write_minimal_feed_zip

DATA_RAW = Path(__file__).resolve().parent / "data"
DET_GTFS_ZIP = DATA_RAW / "usa_det_gtfs.zip"


@pytest.fixture
def minimal_feed() -> gk.Feed:
    """A tiny in-memory valid GTFS feed."""
    return build_minimal_feed()


@pytest.fixture
def minimal_feed_zip(tmp_path: Path) -> Path:
    """Path to a tiny valid GTFS zip written to a tmp dir."""
    return write_minimal_feed_zip(tmp_path / "minimal.zip")


@pytest.fixture
def det_gtfs_path() -> Path:
    """Path to the bundled Detroit sample GTFS feed.

    Skips locally when the fixture is absent (e.g. a sparse checkout), but
    fails under CI: the feed is tracked in the repository and every CI job is
    required to exercise it, so a missing file there is a packaging regression
    rather than a reason to pass silently.
    """
    if not DET_GTFS_ZIP.exists():
        message = f"Bundled Detroit GTFS not found at {DET_GTFS_ZIP}"
        if os.environ.get("CI"):
            pytest.fail(f"{message} — CI must run the Detroit integration tests.")
        pytest.skip(message)
    return DET_GTFS_ZIP
