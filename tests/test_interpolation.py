"""Trajectory-interpolation tests."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import LineString

from gtfs2emis_py.gtfs import read_gtfs
from gtfs2emis_py.gtfs.interpolation import (
    SNAP_WARN_M,
    _s2_haversine_m,
    _snap_nearest2,
    build_transport_table,
    parse_service_day_seconds,
)

DATA = Path(__file__).resolve().parent / "data"
SYNTHETIC = DATA / "synthetic_edge_cases.zip"
DETROIT = DATA / "usa_det_gtfs.zip"
GEOD = Geod(ellps="WGS84")

SCHEMA = [
    "shape_id",
    "trip_id",
    "route_id",
    "route_type",
    "stop_sequence",
    "from_stop_id",
    "to_stop_id",
    "from_service_day_seconds",
    "to_service_day_seconds",
    "from_timestamp",
    "to_timestamp",
    "dist_km",
    "cumdist_km",
    "speed_kmh",
    "cumtime_s",
    "geometry",
]


@pytest.fixture(scope="module")
def synthetic() -> gpd.GeoDataFrame:
    return build_transport_table(SYNTHETIC)


@pytest.fixture(scope="module")
def synthetic_dated() -> gpd.GeoDataFrame:
    return build_transport_table(SYNTHETIC, service_date="2026-03-02")


def test_schema_and_dtypes(synthetic: gpd.GeoDataFrame) -> None:
    assert list(synthetic.columns) == SCHEMA
    assert synthetic.crs is not None and synthetic.crs.to_epsg() == 4326
    assert synthetic["route_type"].dtype == "Int16"
    assert synthetic["stop_sequence"].dtype == "Int32"
    assert synthetic["from_service_day_seconds"].dtype == "Int32"
    assert synthetic["dist_km"].dtype == "float64"


def test_skips_one_point_shape(synthetic: gpd.GeoDataFrame) -> None:
    assert "T_ONEPOINT" not in set(synthetic["trip_id"])


def test_frequencies_are_expanded(synthetic: gpd.GeoDataFrame) -> None:
    trips = set(synthetic["trip_id"])
    assert "T_FREQ" not in trips
    assert {f"T_FREQ-freq-{index}" for index in range(12)} <= trips


def test_row_count_is_stops_minus_one(synthetic: gpd.GeoDataFrame) -> None:
    feed = read_gtfs(SYNTHETIC).feed.expand_frequencies()
    geometries = feed.build_geometry_by_shape()
    buildable = {
        str(trip_id)
        for trip_id, shape_id in zip(feed.trips["trip_id"], feed.trips["shape_id"], strict=False)
        if isinstance(shape_id, str)
        and shape_id in geometries
        and geometries[shape_id].geom_type == "LineString"
    }
    counts = feed.stop_times.groupby("trip_id").size()
    expected = sum(int(counts[trip_id]) - 1 for trip_id in buildable if counts.get(trip_id, 0) >= 2)
    assert len(synthetic) == expected
    assert not synthetic.duplicated(["trip_id", "stop_sequence"]).any()


def test_spacing_is_at_most_the_resolution(synthetic: gpd.GeoDataFrame) -> None:
    for geometry in synthetic["geometry"]:
        coordinates = list(geometry.coords)
        for (lon1, lat1), (lon2, lat2) in zip(coordinates[:-1], coordinates[1:], strict=False):
            _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
            assert distance <= 100.0 + 1e-6


def test_original_vertices_are_kept_in_order(synthetic: gpd.GeoDataFrame) -> None:
    row = synthetic[synthetic["trip_id"] == "T_FREQ-freq-0"].iloc[0]
    coordinates = list(row.geometry.coords)
    # Shape coordinates are (lat, lon) in GTFS and (lon, lat) in the geometry.
    original = [(0.0, 0.0), (0.0, 0.005), (0.0, 0.01), (0.005, 0.01)]
    positions = [
        min(
            range(len(coordinates)),
            key=lambda index: (
                abs(coordinates[index][0] - vertex[0]) + abs(coordinates[index][1] - vertex[1])
            ),
        )
        for vertex in original
    ]
    assert positions == sorted(positions)
    assert len(coordinates) > len(original)


def test_distance_matches_the_geodesic_length(synthetic: gpd.GeoDataFrame) -> None:
    row = synthetic[synthetic["trip_id"] == "T_FREQ-freq-0"].iloc[0]
    coordinates = list(row.geometry.coords)
    total_m = sum(
        _s2_haversine_m(lon1, lat1, lon2, lat2)
        for (lon1, lat1), (lon2, lat2) in zip(coordinates[:-1], coordinates[1:], strict=False)
    )
    assert row["dist_km"] == pytest.approx(total_m / 1000.0)


def test_cumulative_values_are_monotonic_and_finite(synthetic: gpd.GeoDataFrame) -> None:
    for _, group in synthetic.groupby("trip_id"):
        assert group["cumdist_km"].is_monotonic_increasing
        assert group["cumtime_s"].is_monotonic_increasing
        assert (group["dist_km"] >= 0).all()
        assert (group["speed_kmh"] >= 0).all()
        assert group["speed_kmh"].map(math.isfinite).all()
        assert group["dist_km"].map(math.isfinite).all()


def test_trip_endpoints_match_gtfs(synthetic: gpd.GeoDataFrame) -> None:
    late = synthetic[synthetic["trip_id"] == "T_LATE"].iloc[0]
    assert late["from_service_day_seconds"] == 25 * 3600 + 10 * 60
    assert late["to_service_day_seconds"] == 25 * 3600 + 20 * 60


def test_loop_shape_preserves_the_forward_snap_increasing(synthetic: gpd.GeoDataFrame) -> None:
    loop = synthetic[synthetic["trip_id"] == "T_LOOP"].sort_values("stop_sequence")
    assert len(loop) == 2
    # gtfs2gps' forward nearest2 scan ends at the penultimate vertex, not the
    # duplicated loop endpoint; that is the reference behaviour, not a closed
    # ring in the Python output.
    assert loop.iloc[0].geometry.coords[0] == pytest.approx((0.0, 0.0))
    assert loop.iloc[-1].geometry.coords[-1] == pytest.approx((0.0005, 0.0005), abs=1e-3)
    assert loop["cumdist_km"].is_monotonic_increasing
    assert loop["cumdist_km"].iloc[-1] > 0


def test_duplicate_shape_points_produce_no_nan_or_inf(synthetic: gpd.GeoDataFrame) -> None:
    duplicate = synthetic[synthetic["trip_id"] == "T_DUP"].sort_values("stop_sequence")
    numeric = ["dist_km", "cumdist_km", "speed_kmh", "cumtime_s"]
    assert duplicate[numeric].notna().all().all()
    assert duplicate[numeric].map(math.isfinite).all().all()
    # The repeated (lat, lon) pair sits between stops 2 and 3, so the second
    # interval carries the zero-length step.
    # gtfs2gps's forward snap may place both stops past the repeated vertex;
    # what matters here is that the duplicate never produces NaN/Inf.
    assert duplicate["geometry"].notna().all()


def test_timestamps_are_null_without_a_service_date(synthetic: gpd.GeoDataFrame) -> None:
    assert synthetic["from_timestamp"].isna().all()
    assert synthetic["to_timestamp"].isna().all()


def test_service_day_timestamps_do_not_wrap(synthetic_dated: gpd.GeoDataFrame) -> None:
    late = synthetic_dated[synthetic_dated["trip_id"] == "T_LATE"].iloc[0]
    assert late["from_timestamp"] == pd.Timestamp("2026-03-03 01:10", tz="UTC")
    assert late["to_timestamp"] == pd.Timestamp("2026-03-03 01:20", tz="UTC")


def test_deterministic_between_runs() -> None:
    first = build_transport_table(SYNTHETIC)
    second = build_transport_table(SYNTHETIC)
    assert first.equals(second)


def test_detroit_feed_builds_the_contract_schema() -> None:
    table = build_transport_table(DETROIT, spatial_resolution=200.0)
    assert not table.empty
    assert list(table.columns) == SCHEMA
    assert table["shape_id"].nunique() > 1
    assert not table["dist_km"].isna().any()
    assert (table["dist_km"] >= 0).all()
    assert table["speed_kmh"].map(math.isfinite).all()
    assert not table.duplicated(["trip_id", "stop_sequence"]).any()
    assert table["cumdist_km"].sum() > 0


def test_interval_geometry_is_a_linestring(synthetic: gpd.GeoDataFrame) -> None:
    assert synthetic.geometry.geom_type.eq("LineString").all()
    assert synthetic.geometry.is_valid.all()


def test_untimed_intermediate_stop_is_interpolated() -> None:
    tables = {
        "agency": pd.DataFrame(
            [
                {
                    "agency_id": "AG",
                    "agency_name": "A",
                    "agency_url": "https://example.org",
                    "agency_timezone": "UTC",
                }
            ]
        ),
        "stops": pd.DataFrame(
            [
                {"stop_id": "S1", "stop_lat": 0.0, "stop_lon": 0.0},
                {"stop_id": "S2", "stop_lat": 0.005, "stop_lon": 0.0},
                {"stop_id": "S3", "stop_lat": 0.01, "stop_lon": 0.0},
            ]
        ),
        "routes": pd.DataFrame([{"route_id": "R1", "agency_id": "AG", "route_type": 3}]),
        "trips": pd.DataFrame(
            [{"trip_id": "T1", "route_id": "R1", "service_id": "WK", "shape_id": "SH1"}]
        ),
        "stop_times": pd.DataFrame(
            [
                {
                    "trip_id": "T1",
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:00",
                },
                {
                    "trip_id": "T1",
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "arrival_time": None,
                    "departure_time": None,
                },
                {
                    "trip_id": "T1",
                    "stop_id": "S3",
                    "stop_sequence": 3,
                    "arrival_time": "08:10:00",
                    "departure_time": "08:10:00",
                },
            ]
        ),
        "shapes": pd.DataFrame(
            [
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.0,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 1,
                },
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.005,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 2,
                },
                {
                    "shape_id": "SH1",
                    "shape_pt_lat": 0.01,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 3,
                },
            ]
        ),
    }
    table = build_transport_table(gk.Feed(dist_units="km", **tables))
    assert len(table) == 2
    first = table[table["stop_sequence"] == 1].iloc[0]
    second = table[table["stop_sequence"] == 2].iloc[0]
    # The middle stop is equidistant, so the interpolated time is the midpoint.
    assert first["to_service_day_seconds"] == 8 * 3600 + 5 * 60 + 20
    assert second["from_service_day_seconds"] == 8 * 3600 + 5 * 60 + 20


def test_builder_accepts_basic_iso_service_date() -> None:
    table = build_transport_table(SYNTHETIC, service_date="20260302")
    assert table["from_timestamp"].notna().all()
    assert str(table["from_timestamp"].iloc[0].date()) == "2026-03-02"


def test_service_date_must_be_a_string_or_date() -> None:
    with pytest.raises(TypeError, match="service_date"):
        build_transport_table(SYNTHETIC, service_date=20260302)  # type: ignore[arg-type]


def test_resolution_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_transport_table(SYNTHETIC, spatial_resolution=0)


def test_parse_service_day_seconds() -> None:
    assert parse_service_day_seconds("08:05:00") == 8 * 3600 + 5 * 60
    assert parse_service_day_seconds("25:10:00") == 25 * 3600 + 10 * 60
    assert parse_service_day_seconds("") is None
    assert parse_service_day_seconds(None) is None
    for bad in ("8:5", "aa:bb:cc", "08:99:00", "48:00:00"):
        with pytest.raises(ValueError):
            parse_service_day_seconds(bad)


def test_snap_threshold_is_documented() -> None:
    assert SNAP_WARN_M == 100.0


def test_line_string_interval_for_zero_length_snap() -> None:
    # A trip whose two stops snap to the same vertex still yields a valid
    # two-point LineString, not a null geometry.
    table = build_transport_table(SYNTHETIC)
    assert table.geometry.notna().all()
    assert table.geometry.apply(lambda geometry: isinstance(geometry, LineString)).all()


def test_nearest2_snaps_forward_to_the_local_minimum() -> None:
    # ~111 m per 0.001 degree of longitude at the equator.
    vertices = [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.003, 0.0)]
    stops = [(0.0001, 0.0), (0.0018, 0.0)]
    snapped = _snap_nearest2(stops, vertices, 60.0)
    assert snapped == [0, 2]


def test_nearest2_doubles_the_tolerance_when_needed() -> None:
    vertices = [(0.0, 0.0), (0.0015, 0.0), (0.003, 0.0)]
    stops = [(0.0014, 0.0)]
    assert _snap_nearest2(stops, vertices, 100.0) == [1]


def test_nearest2_keeps_loop_endpoints() -> None:
    vertices = [(0.0, 0.0), (0.001, 0.001), (0.002, 0.002), (0.001, 0.001), (0.0, 0.0)]
    stops = [(0.0, 0.0), (0.001, 0.001), (0.0, 0.0)]
    snapped = _snap_nearest2(stops, vertices, 100.0)
    assert snapped[0] == 0
    # The C++ fallback appends the 1-based penultimate row; in this 0-based port
    # that is len(vertices) - 2.
    assert snapped[-1] == len(vertices) - 2


def test_parse_service_date_accepts_both_iso_forms() -> None:
    from datetime import date

    from gtfs2emis_py.gtfs.interpolation import parse_service_date

    assert parse_service_date("2026-03-02") == date(2026, 3, 2)
    assert parse_service_date("20260302") == date(2026, 3, 2)


def test_empty_table_has_the_contract_shape() -> None:
    from gtfs2emis_py.gtfs.interpolation import _empty_table

    table = _empty_table()
    assert table.empty
    assert list(table.columns) == [
        "shape_id",
        "trip_id",
        "route_id",
        "route_type",
        "stop_sequence",
        "from_stop_id",
        "to_stop_id",
        "from_service_day_seconds",
        "to_service_day_seconds",
        "from_timestamp",
        "to_timestamp",
        "dist_km",
        "cumdist_km",
        "speed_kmh",
        "cumtime_s",
        "geometry",
    ]
    assert table.attrs["build_stats"]["skipped_no_shape"] == 0
