"""Shared in-memory GTFS builders.

These live outside ``conftest`` so test modules can import them directly
without depending on pytest's collection layout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import gtfs_kit as gk
import pandas as pd
import pytest


def packaged_ef_table(name: str) -> bool:
    """Return True when ``data/ef/<name>.parquet`` ships with this tree.

    EF tables other than ``europe_emep`` are fetch-only, so a release tree
    without the development Parquet files cannot exercise lookups against them.
    Tests that need the table call ``pytest.mark.skipif`` with this helper.
    """
    data = resources.files("gtfs2emis_py").joinpath("data", "ef")
    return data.joinpath(f"{name}.parquet").is_file()


def requires_ef_table(*names: str) -> pytest.MarkDecorator:
    """Skip a test when any named EF table is not packaged in this tree."""
    missing = [name for name in names if not packaged_ef_table(name)]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"fetch-only EF table not present in this tree: {', '.join(missing)}",
    )


def _minimal_tables() -> dict[str, pd.DataFrame]:
    """Return a dict of GTFS tables for a tiny but valid feed.

    Two stops, one route, one shape with three points, one trip, two stop
    times. Distances are in kilometres to match the wrapper's default units.
    """
    return {
        "agency": pd.DataFrame(
            [{"agency_id": "AG", "agency_name": "Test Agency", "agency_timezone": "UTC"}]
        ),
        "stops": pd.DataFrame(
            [
                {"stop_id": "S1", "stop_lat": 0.0, "stop_lon": 0.0, "stop_name": "Stop 1"},
                {"stop_id": "S2", "stop_lat": 0.01, "stop_lon": 0.0, "stop_name": "Stop 2"},
            ]
        ),
        "routes": pd.DataFrame(
            [{"route_id": "R1", "agency_id": "AG", "route_type": 3, "route_short_name": "1"}]
        ),
        "shapes": pd.DataFrame(
            [
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.0,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 0,
                },
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.005,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 1,
                },
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.01,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 2,
                },
            ]
        ),
        "trips": pd.DataFrame(
            [{"trip_id": "T1", "route_id": "R1", "service_id": "WK", "shape_id": "SH1"}]
        ),
        "stop_times": pd.DataFrame(
            [
                {
                    "trip_id": "T1",
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "departure_time": "08:00:00",
                    "arrival_time": "08:00:00",
                },
                {
                    "trip_id": "T1",
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "departure_time": "08:05:00",
                    "arrival_time": "08:05:00",
                },
            ]
        ),
        "calendar": pd.DataFrame(
            [
                {
                    "service_id": "WK",
                    "monday": 1,
                    "tuesday": 1,
                    "wednesday": 1,
                    "thursday": 1,
                    "friday": 1,
                    "saturday": 0,
                    "sunday": 0,
                    "start_date": "20220101",
                    "end_date": "20221231",
                }
            ]
        ),
    }


def build_minimal_feed() -> gk.Feed:
    """Build a tiny valid ``gtfs_kit.Feed`` in memory (no I/O)."""
    tables = _minimal_tables()
    feed = gk.Feed(dist_units="km", **tables)
    return feed


def write_minimal_feed_zip(path: Path) -> Path:
    """Write a minimal valid GTFS feed as a zip at ``path`` and return it."""
    path = Path(path)
    tables = _minimal_tables()
    # gtfs_kit.Feed.to_file writes a zip when the path ends in .zip
    feed = gk.Feed(dist_units="km", **tables)
    feed.to_file(path)
    return path
