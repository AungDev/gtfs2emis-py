"""Transport parity against the committed R fixtures.

Only shapes where every trip has the same stop_id sequence are compared. R's
gtfs2gps snaps one representative trip per shape and reuses its positions for
all trips; for mixed-pattern shapes that is an upstream assumption failure, not
a Python parity target (see docs/parity.md).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gtfs_kit as gk
import numpy as np
import pandas as pd
import pytest

from gtfs2emis_py import transport_model
from gtfs2emis_py.transport_model import TransportModel

FIXTURES = Path(__file__).parents[1] / "../parity/fixtures/transport"
pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(), reason="development-only parity fixtures are not published"
)
DATA = Path(__file__).parent / "data"

# The fixtures serialize six decimals. The comparison is exact at that
# precision; the only adjustment is removing R's leading boundary interval from
# its cumulative columns (Python intentionally emits real stop pairs only).
FIXTURE_DECIMALS = 6
FEEDS = ("bra_cur_gtfs", "usa_det_gtfs", "irl_dub_gtfs")


def _homogeneous_shapes(feed_name: str) -> set[str]:
    feed = gk.read_feed(DATA / f"{feed_name}.zip", dist_units="km")
    trips = feed.trips.set_index("trip_id")["shape_id"].astype(str).to_dict()
    patterns: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for trip_id, group in feed.stop_times.groupby("trip_id"):
        shape_id = trips.get(str(trip_id))
        if shape_id is not None:
            ordered = group.sort_values("stop_sequence")["stop_id"].astype(str)
            patterns[shape_id].add(tuple(ordered))
    return {shape_id for shape_id, seen in patterns.items() if len(seen) == 1}


@pytest.mark.parity
@pytest.mark.parametrize("feed_name", FEEDS)
def test_transport_matches_r_on_homogeneous_shapes(feed_name: str) -> None:
    fixture_path = FIXTURES / f"{feed_name}.csv"
    assert fixture_path.exists(), f"missing R fixture: {fixture_path}"
    homogeneous = _homogeneous_shapes(feed_name)

    fixture = pd.read_csv(
        fixture_path,
        dtype={
            "shape_id": str,
            "trip_id": str,
            "from_stop_id": str,
            "to_stop_id": str,
        },
    )
    filtered = fixture[
        fixture["shape_id"].isin(homogeneous)
        & fixture["stop_sequence"].notna()
        & (fixture["from_stop_id"] != "-")
        & (fixture["to_stop_id"] != "-")
    ]

    python = transport_model(DATA / f"{feed_name}.zip", cumtime_origin="shape_start").table
    python = python[python["shape_id"].astype(str).isin(homogeneous)].copy()
    python = python[
        (python["from_stop_id"].astype(str) != "-") & (python["to_stop_id"].astype(str) != "-")
    ]
    python = python.reset_index(drop=True)
    for column in ("shape_id", "trip_id", "from_stop_id", "to_stop_id"):
        python[column] = python[column].astype(str)

    merged = filtered.merge(
        python,
        on=["shape_id", "trip_id", "from_stop_id", "to_stop_id"],
        suffixes=("_r", "_p"),
        validate="one_to_one",
    )
    assert len(merged) == len(filtered), (
        "R fixture rows without a Python counterpart on homogeneous shapes: "
        f"{len(filtered) - len(merged)}"
    )
    assert len(merged) == len(python), (
        f"Python rows without an R counterpart on homogeneous shapes: {len(python) - len(merged)}"
    )

    # Exact link distance, speed, cumulative distance and shape-start cumtime
    # are all gated here; no model-level tolerance remains.
    expected = {
        "dist_km": pd.to_numeric(merged["dist_km_r"], errors="coerce"),
        "speed_kmh": pd.to_numeric(merged["speed_kmh_r"], errors="coerce"),
        "cumdist_m": pd.to_numeric(merged["cumdist_m"], errors="coerce"),
        "cumtime_s": pd.to_numeric(merged["cumtime_s_r"], errors="coerce"),
    }
    actual = {
        "dist_km": pd.to_numeric(merged["dist_km_p"], errors="coerce"),
        "speed_kmh": pd.to_numeric(merged["speed_kmh_p"], errors="coerce"),
        "cumdist_m": pd.to_numeric(merged["cumdist_km"], errors="coerce") * 1000.0,
        "cumtime_s": pd.to_numeric(merged["cumtime_s_p"], errors="coerce"),
    }
    for column in expected:
        # R leaves NA where its input has no usable speed/timestamp. Those are
        # undefined reference values, not zeroes or a numeric tolerance.
        valid = expected[column].notna().to_numpy()
        r = np.round(expected[column].to_numpy(dtype=float)[valid], FIXTURE_DECIMALS)
        p = np.round(actual[column].to_numpy(dtype=float)[valid], FIXTURE_DECIMALS)
        mismatch = ~np.isclose(r, p, rtol=0.0, atol=0.0, equal_nan=True)
        if mismatch.any():
            first = merged.loc[valid].loc[mismatch].iloc[0]
            raise AssertionError(
                f"{feed_name}: {column} differs on {int(mismatch.sum())} rows; "
                f"first {first['shape_id']}/{first['trip_id']} "
                f"R={r[mismatch][0]!r} P={p[mismatch][0]!r}"
            )


@pytest.mark.parity
def test_shape_start_origin_matches_r_first_row() -> None:
    """Opt-in absolute cumtime reproduces R's leading shape-start interval."""
    model = transport_model(DATA / "usa_det_gtfs.zip", cumtime_origin="shape_start")
    assert isinstance(model, TransportModel)
    row = model.table[model.table["trip_id"] == "2185020"].iloc[0]
    fixture = pd.read_csv(FIXTURES / "usa_det_gtfs.csv", dtype={"trip_id": str})
    first = fixture[
        (fixture["trip_id"] == "2185020")
        & fixture["stop_sequence"].notna()
        & (fixture["from_stop_id"] != "-")
        & (fixture["to_stop_id"] != "-")
    ].sort_values("stop_sequence")
    expected = first["cumtime_s"].iloc[0]
    # R's first-row absolute cumtime is exact at the fixture's six decimals.
    assert round(float(row["cumtime_s"]), 6) == round(float(expected), 6)

    default = transport_model(DATA / "usa_det_gtfs.zip")
    assert isinstance(default, TransportModel)
    default_row = default.table[default.table["trip_id"] == "2185020"].iloc[0]
    assert float(default_row["cumtime_s"]) < float(row["cumtime_s"])
