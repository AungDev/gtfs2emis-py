"""Reference checks for the R GPS-point timestamp capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REFERENCE = Path(__file__).parents[1] / "../parity/reference/gps_timestamps.json"
pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(), reason="development-only timestamp reference is not published"
)


def test_gps_timestamp_reference_is_the_small_r_capture() -> None:
    payload = json.loads(REFERENCE.read_text())
    assert payload["schema_version"] == 1
    assert payload["gtfs2gps_version"] == "2.1.4"
    assert payload["shape_id"] == "shp-1-02"
    assert payload["trip_id"] == "2598020"
    assert len(payload["points"]) > 2
    assert len(payload["intervals"]) == 5

    first = payload["points"][0]
    assert first["id"] == 1
    assert first["timestamp_seconds"] == 29834
    assert first["cumtime_s"] == 0

    stop = next(row for row in payload["intervals"] if row.get("stop_sequence") == 40)
    assert stop["from_seconds"] == 29914
    assert stop["to_seconds"] == 30005
    assert stop["cumtime_s"] == 130.346445170341
