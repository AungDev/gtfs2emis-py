"""Speed-adjustment tests."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from gtfs2emis_py.gtfs.interpolation import build_transport_table
from gtfs2emis_py.gtfs.speed import adjust_speeds

DATA = Path(__file__).resolve().parent / "data"
SYNTHETIC = DATA / "synthetic_edge_cases.zip"
DETROIT = DATA / "usa_det_gtfs.zip"


def _frame(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a minimal contract-shaped table for branch tests."""
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    frame["from_timestamp"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["to_timestamp"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return frame


def _row(
    trip: str = "T1",
    start: int = 0,
    end: int = 100,
    distance: float = 0.5,
    speed: float = 18.0,
    cumtime: float = 100.0,
) -> dict:
    return {
        "trip_id": trip,
        "from_service_day_seconds": start,
        "to_service_day_seconds": end,
        "dist_km": distance,
        "speed_kmh": speed,
        "cumtime_s": cumtime,
        "geometry": LineString([(0.0, 0.0), (0.0, 0.001)]),
    }


@pytest.fixture(scope="module")
def synthetic() -> gpd.GeoDataFrame:
    return build_transport_table(SYNTHETIC)


def test_replaces_a_200_kmh_interval(synthetic: gpd.GeoDataFrame) -> None:
    table = synthetic.copy()
    table.loc[table["trip_id"] == "T_LATE", "speed_kmh"] = 200.0
    adjusted = adjust_speeds(table, new_speed=20.0)
    row = adjusted[adjusted["trip_id"] == "T_LATE"].iloc[0]
    assert row["speed_kmh"] == 20.0
    assert row["to_service_day_seconds"] - row["from_service_day_seconds"] == round(
        row["dist_km"] / 20.0 * 3600
    )


def test_replaces_a_0_5_kmh_interval(synthetic: gpd.GeoDataFrame) -> None:
    table = synthetic.copy()
    table.loc[table["trip_id"] == "T_LATE", "speed_kmh"] = 0.5
    adjusted = adjust_speeds(table, new_speed=15.0)
    assert adjusted.loc[adjusted["trip_id"] == "T_LATE", "speed_kmh"].iloc[0] == 15.0


def test_missing_speed_is_replaced(synthetic: gpd.GeoDataFrame) -> None:
    table = synthetic.copy()
    table.loc[table["trip_id"] == "T_LATE", "speed_kmh"] = float("nan")
    adjusted = adjust_speeds(table, new_speed=12.0)
    assert adjusted.loc[adjusted["trip_id"] == "T_LATE", "speed_kmh"].iloc[0] == 12.0


def test_zero_distance_interval_is_exempt() -> None:
    table = _frame([_row(distance=0.0, speed=0.0, start=0, end=100), _row(start=200, end=300)])
    adjusted = adjust_speeds(table, new_speed=20.0)
    assert adjusted.iloc[0]["speed_kmh"] == 0.0
    # R computes cumtime as cumsum(dist / speed); a zero-distance interval has
    # zero travel time and the scheduled 100 s dwell is not carried over.
    assert (
        adjusted.iloc[0]["to_service_day_seconds"] - adjusted.iloc[0]["from_service_day_seconds"]
        == 0
    )
    assert adjusted.iloc[1]["from_service_day_seconds"] == 0
    assert adjusted.iloc[1]["to_service_day_seconds"] == 100


def test_default_new_speed_is_distance_weighted() -> None:
    table = _frame(
        [
            _row(distance=1.0, speed=10.0, start=0, end=360),
            _row(distance=1.0, speed=200.0, start=400, end=500),
        ]
    )
    adjusted = adjust_speeds(table)
    assert adjusted["speed_kmh"].tolist() == [10.0, 10.0]
    assert (
        adjusted.iloc[1]["to_service_day_seconds"] - adjusted.iloc[1]["from_service_day_seconds"]
        == 360
    )


def test_default_new_speed_is_arithmetic_weighted() -> None:
    table = _frame(
        [
            _row(distance=1.0, speed=10.0, start=0, end=100),
            _row(distance=1.0, speed=20.0, start=100, end=200),
            _row(distance=1.0, speed=200.0, start=200, end=300),
        ]
    )
    adjusted = adjust_speeds(table)
    # R: weighted.mean(c(10, 20), c(1, 1)) = 15, not 2 / (0.1 + 0.05) = 13.33.
    assert adjusted.loc[adjusted["speed_kmh"] == 15.0, "trip_id"].tolist() == ["T1"]
    assert adjusted.iloc[-1]["speed_kmh"] == 15.0


def test_no_valid_speed_without_new_speed_is_undefined() -> None:
    table = _frame([_row(speed=200.0)])
    adjusted = adjust_speeds(table)
    # R's weighted.mean() over an empty valid set returns NaN; it does not
    # invent a replacement speed.
    assert math.isnan(float(adjusted.iloc[0]["speed_kmh"]))
    assert math.isnan(float(adjusted.iloc[0]["cumtime_s"]))


def test_no_valid_speed_with_explicit_new_speed_works() -> None:
    table = _frame([_row(speed=200.0)])
    adjusted = adjust_speeds(table, new_speed=30.0)
    assert adjusted.iloc[0]["speed_kmh"] == 30.0


def test_first_service_day_second_is_preserved(synthetic: gpd.GeoDataFrame) -> None:
    table = synthetic.copy()
    table.loc[table["trip_id"] == "T_LOOP", "speed_kmh"] = [200.0, 200.0]
    adjusted = adjust_speeds(table, new_speed=10.0)
    loop = adjusted[adjusted["trip_id"] == "T_LOOP"].sort_values("stop_sequence")
    original = table[table["trip_id"] == "T_LOOP"].sort_values("stop_sequence")
    assert loop.iloc[0]["from_service_day_seconds"] == original.iloc[0]["from_service_day_seconds"]
    assert loop["cumtime_s"].is_monotonic_increasing
    assert loop["to_service_day_seconds"].is_monotonic_increasing


def test_midnight_is_not_wrapped() -> None:
    table = _frame([_row(start=86_000, end=86_100, distance=1.0, speed=2.0)])
    adjusted = adjust_speeds(table)
    assert adjusted.iloc[0]["to_service_day_seconds"] > 86_400


def test_timestamps_are_shifted_with_the_times() -> None:
    table = build_transport_table(SYNTHETIC, service_date="2026-03-02")
    table.loc[table["trip_id"] == "T_LATE", "speed_kmh"] = 20.0
    adjusted = adjust_speeds(table)
    row = adjusted[adjusted["trip_id"] == "T_LATE"].iloc[0]
    duration = row["to_service_day_seconds"] - row["from_service_day_seconds"]
    assert (row["to_timestamp"] - row["from_timestamp"]).total_seconds() == duration
    assert row["from_timestamp"].date().isoformat() == "2026-03-03"


def test_does_not_mutate_the_input(synthetic: gpd.GeoDataFrame) -> None:
    table = synthetic.copy()
    before = table["speed_kmh"].copy()
    adjust_speeds(table, new_speed=20.0)
    assert table["speed_kmh"].equals(before)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_speed": 1.5}, ValueError),
        ({"max_speed": 2.0}, ValueError),
        ({"new_speed": 0.5}, ValueError),
        ({"min_speed": 90.0, "max_speed": 80.0}, ValueError),
        ({"min_speed": True}, TypeError),
        ({"new_speed": "fast"}, TypeError),
    ],
)
def test_parameter_validation(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        adjust_speeds(_frame([_row()]), **kwargs)


def test_missing_columns_are_rejected() -> None:
    table = _frame([_row()]).drop(columns=["dist_km"])
    with pytest.raises(ValueError, match="dist_km"):
        adjust_speeds(table)


def test_detroit_speeds_are_in_range() -> None:
    table = build_transport_table(DETROIT, spatial_resolution=200.0)
    adjusted = adjust_speeds(table)
    positive = adjusted[adjusted["dist_km"] > 0]
    assert positive["speed_kmh"].between(2.0, 80.0).all()
    for _, group in adjusted.groupby("trip_id"):
        assert group["cumtime_s"].is_monotonic_increasing
        assert group["speed_kmh"].map(math.isfinite).all()


def test_replacement_speed_is_the_shape_level_r_weighted_mean() -> None:
    rows = []
    for trip, shape, valid_speed in (("T1", "SA", 10.0), ("T2", "SB", 30.0)):
        for index in range(2):
            row = _row(
                trip=trip, start=index * 100, end=(index + 1) * 100, distance=1.0, speed=valid_speed
            )
            row["shape_id"] = shape
            rows.append(row)
        bad = _row(trip=trip, start=200, end=300, distance=1.0, speed=200.0)
        bad["shape_id"] = shape
        rows.append(bad)
    adjusted = adjust_speeds(_frame(rows))
    bad = adjusted[adjusted["speed_kmh"].between(199, 201)]
    assert bad.empty
    # R calls adjust_speed() per shape file: each shape gets its own mean.
    assert adjusted[adjusted["shape_id"] == "SA"]["speed_kmh"].iloc[-1] == pytest.approx(10.0)
    assert adjusted[adjusted["shape_id"] == "SB"]["speed_kmh"].iloc[-1] == pytest.approx(30.0)


def test_cumtime_is_the_cumulative_travel_time() -> None:
    table = _frame([_row(trip="T1", start=0, end=100, distance=1.0, speed=10.0)])
    extra = _row(trip="T1", start=100, end=200, distance=1.0, speed=20.0)
    table = _frame([table.iloc[0].to_dict(), extra])
    adjusted = adjust_speeds(table)
    assert adjusted["cumtime_s"].tolist() == pytest.approx([360.0, 540.0])


def test_cumtime_origin_shape_start_adds_the_leading_interval() -> None:
    table = _frame(
        [
            _row(trip="T1", start=0, end=100, distance=1.0, speed=10.0),
            _row(trip="T1", start=100, end=200, distance=1.0, speed=10.0),
        ]
    )
    adjusted = adjust_speeds(
        table,
        new_speed=10.0,
        cumtime_origin="shape_start",
        shape_start_km={"T1": 0.5},
    )
    # 0.5 km at 10 km/h = 180 s, then 360 s per 1 km interval.
    assert adjusted["cumtime_s"].tolist() == pytest.approx([540.0, 900.0])


def test_shape_start_origin_requires_the_distance_mapping() -> None:
    table = _frame([_row()])
    with pytest.raises(ValueError, match="shape_start_km"):
        adjust_speeds(table, cumtime_origin="shape_start")


def test_shape_start_origin_reports_a_missing_trip() -> None:
    table = _frame([_row(trip="T1")])
    with pytest.raises(ValueError, match="no entry for trip 'T1'"):
        adjust_speeds(
            table,
            new_speed=10.0,
            cumtime_origin="shape_start",
            shape_start_km={"OTHER": 0.5},
        )


def test_scheduled_dwell_is_not_carried_into_the_adjusted_timetable() -> None:
    table = _frame(
        [
            _row(trip="T1", start=0, end=100, distance=1.0, speed=10.0),
            _row(trip="T1", start=200, end=300, distance=1.0, speed=10.0),
        ]
    )
    adjusted = adjust_speeds(table)
    # R recomputes cumsum(dist / speed); the 100 s gap is schedule dwell, not
    # travel time, so the second interval starts at the first one's end.
    assert adjusted.iloc[1]["from_service_day_seconds"] == 360
    assert adjusted.iloc[1]["to_service_day_seconds"] == 720


def test_no_valid_speed_leaves_the_timetable_undefined() -> None:
    table = _frame(
        [
            _row(trip="T_BAD", start=0, end=100, distance=1.0, speed=1.0),
            _row(trip="T_BAD", start=100, end=200, distance=1.0, speed=1.0),
        ]
    )
    stamp = pd.Timestamp("2026-03-02T00:00:00+00:00")
    table["from_timestamp"] = [stamp, stamp]
    table["to_timestamp"] = [stamp, stamp]
    adjusted = adjust_speeds(table)

    assert adjusted["speed_kmh"].isna().all()
    assert adjusted["cumtime_s"].isna().all()
    assert adjusted["to_service_day_seconds"].isna().all()
    assert adjusted["to_timestamp"].isna().all()
    assert pd.isna(adjusted["from_service_day_seconds"].iloc[1])
    assert adjusted["from_service_day_seconds"].iloc[0] == 0


def test_shape_start_mapping_types_are_validated() -> None:
    table = _frame([_row()])
    with pytest.raises(ValueError, match="requires"):
        adjust_speeds(table, cumtime_origin="shape_start")
    with pytest.raises(ValueError, match="must not be empty"):
        adjust_speeds(table, cumtime_origin="shape_start", shape_start_km={})
    with pytest.raises(TypeError, match="mapping"):
        adjust_speeds(table, shape_start_km=1)
    with pytest.raises(TypeError, match="shape_speed_kmh"):
        adjust_speeds(table, shape_speed_kmh=1)
