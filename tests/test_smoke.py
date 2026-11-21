"""Smoke test: package imports and exposes its version."""

import gtfs2emis_py


def test_version_is_string() -> None:
    assert isinstance(gtfs2emis_py.__version__, str)
    assert gtfs2emis_py.__version__
