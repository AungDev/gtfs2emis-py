"""Analysis helper tests."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from gtfs2emis_py import (
    EmissionResult,
    emis_grid,
    emis_summary,
    emission_model,
)
from tests._fixtures import packaged_ef_table

# Every result in this module is built through emission_model with the CETESB
# table, which ships fetch-only, so the module has nothing to read in a tree
# without the development Parquet files.
pytestmark = pytest.mark.skipif(
    not packaged_ef_table("brazil_cetesb"),
    reason="analysis fixtures are built from the fetch-only CETESB table",
)


def _transport(seconds: list[int] | None = None) -> gpd.GeoDataFrame:
    count = 2
    return gpd.GeoDataFrame(
        {
            "shape_id": ["SH1"] * count,
            "trip_id": ["T1", "T2"],
            "route_id": ["R1", "R1"],
            "stop_sequence": [0, 1],
            "from_service_day_seconds": seconds or [0, 90000],
            "speed_kmh": [20.0, 20.0],
            "dist_km": [1.0, 2.0],
        },
        geometry=[LineString([(0, 0), (0.001, 0)]), LineString([(0.001, 0), (0.002, 0)])],
        crs="EPSG:4326",
    )


def _fleet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type_name_br": ["BUS_URBAN_D"],
            "model_year": [2015],
            "fuel": ["D"],
            "fleet_composition": [1.0],
        }
    )


def _result() -> EmissionResult:
    return emission_model(_transport(), "ef_brazil_cetesb", _fleet(), ["CO"])


def _table(result: EmissionResult):
    return getattr(result.tp_model, "table", result.tp_model)


def test_summary_matches_manual_groupby() -> None:
    result = _result()
    summary = emis_summary(result)
    manual = result.emi.groupby(["pollutant", "process"], as_index=False)["value"].sum()
    assert summary["value"].tolist() == pytest.approx(manual["value"].tolist())
    assert summary["unit"].tolist() == ["g"]


def test_time_buckets_across_midnight() -> None:
    summary = emis_summary(_result(), by=["pollutant", "time"])
    by_hour = summary.drop_duplicates("service_hour").sort_values("service_hour")
    assert by_hour["hour_of_day"].tolist() == [0, 1]
    assert by_hour["service_hour"].tolist() == [0, 25]


def test_vehicle_and_segment_variables() -> None:
    summary = emis_summary(
        _result(),
        by=["pollutant", "vehicle"],
        segment_vars=["route_id"],
    )
    assert set(summary.columns) == {
        "pollutant",
        "process",
        "vehicle_type",
        "route_id",
        "value",
        "unit",
    }
    assert summary["route_id"].unique().tolist() == ["R1"]


@pytest.mark.parametrize("vehicle_var", ["vehicle_type", "fleet_row"])
def test_vehicle_by_with_explicit_vehicle_var_is_deduplicated(vehicle_var: str) -> None:
    summary = emis_summary(_result(), by=["pollutant", "vehicle"], veh_vars=[vehicle_var])
    assert vehicle_var in summary.columns
    assert summary.columns.tolist().count(vehicle_var) == 1


@pytest.mark.parametrize("segment_var", ["shape_id", "trip_id", "stop_sequence"])
def test_segment_vars_may_overlap_emission_columns(segment_var: str) -> None:
    summary = emis_summary(_result(), segment_vars=[segment_var])
    assert segment_var in summary.columns
    assert not summary.empty


def test_summary_without_time_does_not_require_service_day_column() -> None:
    result = _result()
    table = _table(result).drop(columns=["from_service_day_seconds"])
    result = EmissionResult.from_long(
        result.emi, table, metadata=result.metadata, units=result.units
    )
    summary = emis_summary(result, by="pollutant")
    assert summary["value"].sum() == pytest.approx(result.emi["value"].sum())


def test_summary_rejects_unknown_dimensions() -> None:
    with pytest.raises(ValueError, match="unknown by"):
        emis_summary(_result(), by="route")
    with pytest.raises(ValueError, match="unknown segment_vars"):
        emis_summary(_result(), segment_vars=["nope"])
    with pytest.raises(TypeError, match="process_vars"):
        emis_summary(_result(), process_vars=["process"])


def _grid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "cell": ["west", "east"],
        },
        geometry=[
            Polygon([(0, -0.001), (0.001, -0.001), (0.001, 0.001), (0, 0.001)]),
            Polygon([(0.001, -0.001), (0.002, -0.001), (0.002, 0.001), (0.001, 0.001)]),
        ],
        crs="EPSG:4326",
    )


def test_grid_conserves_total_and_splits_by_length() -> None:
    result = _result()
    grid = emis_grid(result, _grid())
    assert grid["value"].sum() == pytest.approx(result.emi["value"].sum())
    per_segment = result.emi.groupby("segment_id")["value"].sum()
    west, east = grid.sort_values("cell")["value"].tolist()
    assert west == pytest.approx(per_segment.loc[0] + per_segment.loc[1] / 2)
    assert east == pytest.approx(per_segment.loc[1] / 2)


def test_grid_time_bucket_and_unaligned() -> None:
    grid = emis_grid(_result(), _grid(), time_resolution="hour", aggregate=False)
    assert set(grid["time_bucket"]) == {0, 25}
    assert grid["value"].sum() == pytest.approx(_result().emi["value"].sum())


def test_grid_crs_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="CRS mismatch"):
        emis_grid(_result(), _grid().to_crs("EPSG:3857"))


def test_round_trip_through_long_frame() -> None:
    result = _result()
    rebuilt = EmissionResult.from_long(
        result.to_long(), result.tp_model, result.metadata, result.units
    )
    assert rebuilt.emi.equals(result.emi)
    assert rebuilt.units == result.units


def test_summary_rejects_invalid_result_and_vehicle_vars() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        emis_summary(object())
    with pytest.raises(ValueError, match="unknown veh_vars"):
        emis_summary(_result(), veh_vars=["driver"])
    result = _result()
    missing_time = _table(result).drop(columns=["from_service_day_seconds"])
    fake = type("Fake", (), {"emi": result.emi, "tp_model": missing_time})()
    with pytest.raises(ValueError, match="from_service_day_seconds"):
        emis_summary(fake, by=["pollutant", "time"])


def test_summary_rejects_a_segment_missing_from_the_transport_table() -> None:
    result = _result()
    broken = result.emi.copy()
    broken.loc[0, "segment_id"] = 999
    fake = type("Fake", (), {"emi": broken, "tp_model": result.tp_model})()
    with pytest.raises(ValueError, match="not present"):
        emis_summary(fake)


def test_grid_validates_grid_and_transport_inputs() -> None:
    result = _result()
    with pytest.raises(TypeError, match="GeoDataFrame"):
        emis_grid(result, pd.DataFrame({"cell": [1]}))
    with pytest.raises(ValueError, match="no CRS"):
        emis_grid(result, _grid().set_crs(None, allow_override=True))
    with pytest.raises(ValueError, match="empty"):
        emis_grid(result, _grid().iloc[0:0])
    invalid = _grid()
    invalid.loc[0, "geometry"] = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])
    with pytest.raises(ValueError, match="invalid|empty"):
        emis_grid(result, invalid)
    with pytest.raises(ValueError, match="Polygon"):
        emis_grid(
            result,
            gpd.GeoDataFrame(
                {"cell": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326"
            ),
        )
    with pytest.raises(ValueError, match="time_resolution"):
        emis_grid(result, _grid(), time_resolution="week")
    no_crs = _table(result).set_crs(None, allow_override=True)  # type: ignore[arg-type]
    fake = type("Fake", (), {"emi": result.emi, "tp_model": no_crs})()
    with pytest.raises(ValueError, match="transport table has no CRS"):
        emis_grid(fake, _grid())
    no_geometry = _table(result).drop(columns=["geometry"])
    fake = type("Fake", (), {"emi": result.emi, "tp_model": no_geometry})()
    with pytest.raises(TypeError, match="GeoDataFrame"):
        emis_grid(fake, _grid())


def test_grid_rejects_a_missing_segment_and_no_intersection() -> None:
    result = _result()
    broken = result.emi.copy()
    broken.loc[0, "segment_id"] = 999
    fake = type("Fake", (), {"emi": broken, "tp_model": result.tp_model})()
    with pytest.raises(ValueError, match="not in the transport"):
        emis_grid(fake, _grid())
    far = gpd.GeoDataFrame(
        {"cell": [1]},
        geometry=[Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="outside it"):
        emis_grid(result, far)


def _west_grid():
    return _grid().iloc[[0]].copy()


def test_grid_partial_coverage_raises_by_default() -> None:
    with pytest.raises(ValueError, match="outside it|covered by the grid"):
        emis_grid(_result(), _west_grid())


def test_grid_partial_coverage_warns_with_the_ratio() -> None:
    with pytest.warns(UserWarning, match="outside it"):
        grid = emis_grid(_result(), _west_grid(), on_uncovered="warn")
    assert grid["value"].sum() < _result().emi["value"].sum()


def test_grid_partial_coverage_drop_is_explicit_and_records_coverage() -> None:
    grid = emis_grid(_result(), _west_grid(), on_uncovered="drop", aggregate=False)
    assert set(grid.columns) >= {"coverage", "value"}
    assert grid["coverage"].between(0, 1).all()
    assert grid["value"].sum() < _result().emi["value"].sum()


def test_grid_accepts_a_string_segment_index() -> None:
    result = _result()
    table = _table(result).copy()
    table.index = pd.Index(["seg-a", "seg-b"])
    emi = result.emi.copy()
    emi["segment_id"] = ["seg-a", "seg-b"]
    string_result = EmissionResult.from_long(
        emi, table, metadata=result.metadata, units=result.units
    )
    grid = emis_grid(string_result, _grid(), aggregate=False)
    assert set(grid["segment_id"]) == {"seg-a", "seg-b"}


def test_summary_rejects_invalid_result_shapes_and_columns() -> None:
    result = _result()
    with pytest.raises(TypeError, match="must be a string or a sequence"):
        emis_summary(result, by=123)
    with pytest.raises(TypeError, match="must expose a pandas DataFrame"):
        emis_summary(
            EmissionResult(None, result.ef, result.tp_model, result.metadata, result.units)
        )
    table = _table(result).drop(columns=["route_id"])
    stripped = EmissionResult.from_long(
        result.emi, table, metadata=result.metadata, units=result.units
    )
    with pytest.raises(ValueError, match="unknown segment_vars"):
        emis_summary(stripped, segment_vars=["route_id"])
    emi = result.emi.copy()
    emi["segment_id"] = emi["segment_id"].astype(object)
    emi.loc[0, "segment_id"] = "not-in-table"
    missing = EmissionResult.from_long(
        emi, _table(result), metadata=result.metadata, units=result.units
    )
    with pytest.raises(ValueError, match="not present"):
        emis_summary(missing, segment_vars=["route_id"])


def test_grid_rejects_invalid_inputs() -> None:
    result = _result()
    with pytest.raises(TypeError, match="GeoDataFrame"):
        emis_grid(result, pd.DataFrame())
    with pytest.raises(ValueError, match="no CRS"):
        emis_grid(
            result, gpd.GeoDataFrame({"x": [1]}, geometry=[_grid().geometry.iloc[0]], crs=None)
        )
    with pytest.raises(ValueError, match="empty"):
        emis_grid(result, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    point = gpd.GeoDataFrame(geometry=[gpd.points_from_xy([0.0], [0.0])[0]], crs="EPSG:4326")
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        emis_grid(result, point)
    with pytest.raises(ValueError, match="time_resolution"):
        emis_grid(result, _grid(), time_resolution="week")
    with pytest.raises(ValueError, match="on_uncovered"):
        emis_grid(result, _grid(), on_uncovered="ignore")
    no_crs = _table(result).set_crs(None, allow_override=True)
    no_crs_result = EmissionResult.from_long(
        result.emi, no_crs, metadata=result.metadata, units=result.units
    )
    with pytest.raises(ValueError, match="transport table has no CRS"):
        emis_grid(no_crs_result, _grid())
    missing = _table(result).drop(columns=["from_service_day_seconds"])
    missing_result = EmissionResult.from_long(
        result.emi, missing, metadata=result.metadata, units=result.units
    )
    with pytest.raises(ValueError, match="missing column"):
        emis_grid(missing_result, _grid())


def test_grid_uncovered_warn_and_no_intersection() -> None:
    result = _result()
    west = _grid().iloc[[0]]
    with pytest.warns(UserWarning, match="covered by the grid"):
        emis_grid(result, west, on_uncovered="warn")
    assert emis_grid(result, west, on_uncovered="drop")["value"].sum() < result.emi["value"].sum()
    far = gpd.GeoDataFrame(
        geometry=[Polygon([(1, 1), (1.001, 1), (1.001, 1.001), (1, 1.001)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="no grid cell"):
        emis_grid(result, far, on_uncovered="drop")
