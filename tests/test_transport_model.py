"""transport_model() tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gtfs2emis_py.gtfs.reader import GtfsFeed, read_gtfs
from gtfs2emis_py.transport_model import (
    TransportManifest,
    TransportModel,
    TransportModelError,
    transport_model,
)

DATA = Path(__file__).parent / "data"
DETROIT = DATA / "usa_det_gtfs.zip"
CURITIBA = DATA / "bra_cur_gtfs.zip"
COLUMNS = [
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
def detroit_model() -> TransportModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = transport_model(DETROIT, route_types=(3,), parallel=False)
    assert isinstance(result, TransportModel)
    return result


def test_detroit_contract_schema(detroit_model: TransportModel) -> None:
    table = detroit_model.table
    assert list(table.columns) == COLUMNS
    assert detroit_model.manifest.counts["rows"] == len(table)
    assert detroit_model.manifest.counts["shapes"] > 0
    assert set(table["route_type"].unique()) <= {3}
    assert table["speed_kmh"].ge(0).all()
    assert table["dist_km"].ge(0).all()


def test_coverage_serial_switch_forces_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GTFS2EMIS_COVERAGE_SERIAL", "1")
    model = transport_model(DETROIT, route_types=(3,), parallel=True, ncores=2)
    assert model.manifest.parameters["parallel"] is False


def test_ncores_accepts_full_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    model = transport_model(DATA / "synthetic_edge_cases.zip", parallel=True, ncores=2)
    assert isinstance(model, TransportModel)
    assert model.manifest.parameters["ncores"] == 2


def test_ncores_above_cpu_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    with pytest.raises(ValueError, match="ncores must be <= 2"):
        transport_model(DATA / "synthetic_edge_cases.zip", parallel=True, ncores=3)


def test_root_exports_transport_write_errors() -> None:
    from gtfs2emis_py import ManifestError, TransportWriteError

    assert issubclass(TransportWriteError, RuntimeError)
    assert issubclass(ManifestError, ValueError)


def test_no_trips_error_carries_a_manifest() -> None:
    with pytest.raises(TransportModelError) as exc_info:
        transport_model(DETROIT, route_types=(999,), parallel=False)
    manifest = exc_info.value.manifest
    assert manifest is not None
    assert manifest.counts == {"shapes": 0, "trips": 0, "rows": 0}
    assert manifest.errors


def test_empty_build_error_carries_a_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import geopandas as gpd

    module = importlib.import_module("gtfs2emis_py.transport_model")
    columns = {
        "shape_id": [],
        "trip_id": [],
        "from_service_day_seconds": [],
        "to_service_day_seconds": [],
        "dist_km": [],
        "speed_kmh": [],
        "cumtime_s": [],
    }

    def empty_build(*args, **kwargs):
        return gpd.GeoDataFrame(columns, geometry=[], crs="EPSG:4326")

    monkeypatch.setattr(module, "build_transport_table", empty_build)
    with pytest.raises(TransportModelError) as exc_info:
        transport_model(DETROIT, route_types=(3,), parallel=False)
    assert exc_info.value.manifest is not None
    assert exc_info.value.manifest.counts["rows"] == 0


def test_write_failure_raises_with_the_written_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gtfs2emis_py import TransportWriteError
    from gtfs2emis_py.gtfs import writer

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_shape", fail_write)
    with pytest.raises(TransportWriteError) as exc_info:
        transport_model(DETROIT, route_types=(3,), parallel=False, output_path=tmp_path)
    manifest = exc_info.value.manifest
    assert manifest["shapes"]
    assert all(entry["status"] == "failed" for entry in manifest["shapes"])
    assert "disk full" in manifest["shapes"][0]["error"]


def test_parallel_and_serial_agree(
    detroit_model: TransportModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GTFS2EMIS_COVERAGE_SERIAL", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel = transport_model(DETROIT, route_types=(3,), parallel=True, ncores=2)
    assert isinstance(parallel, TransportModel)
    left = detroit_model.table.drop(columns="geometry")
    right = parallel.table.drop(columns="geometry")
    assert left.equals(right)
    assert detroit_model.table.geometry.geom_equals(parallel.table.geometry).all()


@pytest.mark.parametrize(
    "feed_name",
    ["bra_cur_gtfs.zip", "irl_dub_gtfs.zip", "usa_det_gtfs.zip"],
)
def test_parallel_and_serial_tables_match_on_all_fixtures(
    feed_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GTFS2EMIS_COVERAGE_SERIAL", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = transport_model(DATA / feed_name, route_types=(3,), parallel=False)
        parallel = transport_model(DATA / feed_name, route_types=(3,), parallel=True, ncores=2)
    assert serial.table.drop(columns="geometry").equals(parallel.table.drop(columns="geometry"))
    assert serial.table.geometry.geom_equals(parallel.table.geometry).all()
    assert serial.manifest.counts == parallel.manifest.counts


def test_curitiba_end_to_end() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = transport_model(CURITIBA, route_types=(3,), parallel=False)
    assert isinstance(result, TransportModel)
    assert len(result.table) > 0
    assert list(result.table.columns) == COLUMNS


def test_route_type_selection_with_no_trips_raises() -> None:
    with pytest.raises(TransportModelError, match="no trips survived"):
        transport_model(DETROIT, route_types=(999,), parallel=False)


def test_service_ids_selection() -> None:
    feed = read_gtfs(DETROIT)
    service_id = str(feed.feed.trips["service_id"].iloc[0])
    result = transport_model(
        DETROIT,
        route_types=(3,),
        service_ids=[service_id],
        parallel=False,
    )
    assert isinstance(result, TransportModel)
    trips = feed.feed.trips.copy()
    allowed = set(trips.loc[trips["service_id"].astype(str) == service_id, "trip_id"].astype(str))
    assert set(result.table["trip_id"].astype(str)) <= allowed


def test_directory_feed_is_fingerprinted(tmp_path: Path) -> None:
    import zipfile

    from gtfs2emis_py.transport_model import _sha256

    feed_dir = tmp_path / "feed"
    with zipfile.ZipFile(DATA / "synthetic_edge_cases.zip") as archive:
        archive.extractall(feed_dir)
    model = transport_model(feed_dir, parallel=False)
    assert model.manifest.gtfs_sha256
    before = _sha256(str(feed_dir))
    assert before == model.manifest.gtfs_sha256
    (feed_dir / "agency.txt").write_text((feed_dir / "agency.txt").read_text() + "\n")
    assert _sha256(str(feed_dir)) != before


def test_service_date_materialises_timestamps() -> None:
    feed = read_gtfs(DETROIT)
    calendar = feed.feed.calendar
    row = calendar.iloc[0]
    start = str(row["start_date"]).replace("-", "")
    service_date = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    result = transport_model(
        DETROIT,
        route_types=(3,),
        service_date=service_date,
        parallel=False,
    )
    assert isinstance(result, TransportModel)
    assert result.table["from_timestamp"].notna().all()
    assert result.table["from_timestamp"].dt.tz is not None


def test_feed_is_hashed_once_per_transport_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    module = importlib.import_module("gtfs2emis_py.transport_model")
    calls = []
    original = module._sha256

    def counted(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(module, "_sha256", counted)
    transport_model(DETROIT, route_types=(3,), parallel=False)
    assert len(calls) == 1


def test_in_memory_manifest_records_full_run_parameters() -> None:
    result = transport_model(DETROIT, route_types=(3,), parallel=False)
    parameters = result.manifest.parameters
    assert parameters["parallel"] is False
    assert parameters["ncores"] is None
    assert parameters["output_path"] is None
    assert parameters["resume"] is False
    assert parameters["gtfs_sha256"] == result.manifest.gtfs_sha256


def test_timestamps_and_seconds_use_one_rounded_cumtime() -> None:
    model = transport_model(DETROIT, route_types=(3,), parallel=False)
    table = model.table.sort_values(["trip_id", "stop_sequence"], kind="stable")
    for _trip_id, group in table.groupby("trip_id", sort=False):
        first_from = group["from_service_day_seconds"].iloc[0]
        to_seconds = group["to_service_day_seconds"].to_numpy(dtype=float)
        expected = np.rint(group["cumtime_s"].to_numpy(dtype=float))
        assert np.array_equal(to_seconds - first_from, expected)


def test_service_date_forms_produce_the_same_timestamps() -> None:
    feed = read_gtfs(DETROIT)
    start = str(feed.feed.calendar.iloc[0]["start_date"]).replace("-", "")
    extended = transport_model(
        DETROIT,
        route_types=(3,),
        service_date=f"{start[:4]}-{start[4:6]}-{start[6:]}",
        parallel=False,
    )
    basic = transport_model(
        DETROIT,
        route_types=(3,),
        service_date=start,
        parallel=False,
    )
    assert isinstance(extended, TransportModel)
    assert isinstance(basic, TransportModel)
    assert extended.table.equals(basic.table)
    assert basic.table["from_timestamp"].notna().all()


@pytest.mark.parametrize("value", ["2026-02-30", "20260230", "not-a-date"])
def test_malformed_service_date_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD or YYYYMMDD"):
        transport_model(DETROIT, service_date=value, parallel=False)


def test_service_date_and_service_ids_are_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        transport_model(
            DETROIT,
            service_date="2022-01-03",
            service_ids=["X"],
            parallel=False,
        )


def test_output_path_returns_manifest_and_resumes(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = transport_model(
            DETROIT,
            route_types=(3,),
            parallel=False,
            output_path=tmp_path,
        )
    assert isinstance(first, TransportManifest)
    assert first.errors == ()
    assert first.shapes
    files = {shape.file for shape in first.shapes if shape.file}
    assert files and all((tmp_path / "shapes" / name).exists() for name in files)
    before = {name: (tmp_path / "shapes" / name).stat().st_mtime_ns for name in files}
    second = transport_model(
        DETROIT,
        route_types=(3,),
        parallel=False,
        output_path=tmp_path,
        resume=True,
    )
    assert isinstance(second, TransportManifest)
    after = {name: (tmp_path / "shapes" / name).stat().st_mtime_ns for name in files}
    assert before == after
    assert second.counts == first.counts


def test_resume_without_output_path_raises() -> None:
    with pytest.raises(ValueError, match="requires output_path"):
        transport_model(DETROIT, resume=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_speed": 1.0},
        {"max_speed": 2.0},
        {"min_speed": 80.0, "max_speed": 40.0},
        {"new_speed": 0.5},
        {"cumtime_origin": "middle"},
        {"spatial_resolution": 0.0},
        {"route_types": ()},
        {"ncores": 0},
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        transport_model(DETROIT, parallel=False, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_speed": "2"},
        {"max_speed": True},
        {"new_speed": "10"},
        {"parallel": "yes"},
        {"ncores": True},
        {"spatial_resolution": None},
        {"route_types": "3"},
        {"service_date": 20260101},
        {"service_ids": "WK"},
        {"resume": "yes"},
        {"output_path": "/does/not/exist"},
    ],
)
def test_invalid_parameter_types_are_rejected(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        transport_model(DETROIT, **kwargs)


def test_service_ids_with_no_active_service_raises() -> None:
    with pytest.raises(ValueError, match="no service runs"):
        transport_model(DETROIT, service_ids=["NO_SUCH_SERVICE"], parallel=False)


def test_skipped_counters_and_snap_warnings_are_populated() -> None:
    result = transport_model(DATA / "synthetic_edge_cases.zip", route_types=(3,), parallel=False)
    assert isinstance(result, TransportModel)
    assert result.manifest.skipped["skipped_short_shape"] == 1
    assert result.manifest.warnings
    assert any("snapped" in warning for warning in result.manifest.warnings)


def test_rail_trip_is_dropped_before_the_build() -> None:
    feed = read_gtfs(DATA / "synthetic_edge_cases.zip")
    import copy

    raw = copy.deepcopy(feed.feed)
    raw.routes = pd.concat(
        [
            raw.routes,
            pd.DataFrame([{"route_id": "R_RAIL", "agency_id": "AG", "route_type": 2}]),
        ],
        ignore_index=True,
    )
    raw.trips = pd.concat(
        [
            raw.trips,
            pd.DataFrame(
                [{"trip_id": "T_RAIL", "route_id": "R_RAIL", "service_id": "WK", "shape_id": None}]
            ),
        ],
        ignore_index=True,
    )
    wrapped = GtfsFeed(feed=raw, validation=None, source="<rail-test>")
    result = transport_model(wrapped, route_types=(3,), parallel=False)
    assert "T_RAIL" not in set(result.table["trip_id"])
    assert result.manifest.skipped["dropped_route_type"] == 1


def test_parallel_path_calls_map_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GTFS2EMIS_COVERAGE_SERIAL", raising=False)
    import importlib

    module = importlib.import_module("gtfs2emis_py.transport_model")
    from gtfs2emis_py.gtfs.parallel import MapOutcome

    calls = []

    def fake_map(fn, tasks, n_jobs=None):
        calls.append((fn, len(tasks), n_jobs))
        return MapOutcome(results=tuple(None for _ in tasks), failures=())

    monkeypatch.setattr(module, "map_shapes", fake_map)
    transport_model(DETROIT, route_types=(3,), parallel=True, ncores=2)
    assert calls and calls[0][2] == 2 and calls[0][1] > 1


def test_parallel_payload_is_a_per_shape_mini_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GTFS2EMIS_COVERAGE_SERIAL", raising=False)
    import importlib

    module = importlib.import_module("gtfs2emis_py.transport_model")
    from gtfs2emis_py.gtfs.parallel import MapOutcome

    captured = []

    def fake_map(fn, tasks, n_jobs=None):
        captured.extend(tasks)
        return MapOutcome(results=tuple(None for _ in tasks), failures=())

    monkeypatch.setattr(module, "map_shapes", fake_map)
    feed = read_gtfs(DETROIT)
    transport_model(DETROIT, route_types=(3,), parallel=True, ncores=2)
    assert captured
    full_trips = len(feed.feed.trips)
    for task in captured:
        mini = task.payload["feed"]
        assert set(mini.trips["shape_id"].astype(str)) == {task.shape_id}
        assert set(mini.trips["trip_id"].astype(str)) == set(task.payload["trip_ids"])
        assert len(mini.trips) < full_trips


def test_parallel_shape_failure_names_the_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GTFS2EMIS_COVERAGE_SERIAL", raising=False)
    import importlib

    module = importlib.import_module("gtfs2emis_py.transport_model")
    from gtfs2emis_py.gtfs.parallel import MapOutcome, ShapeFailure, ShapeProcessingError

    failure = ShapeFailure(
        shape_id="shp-3-05",
        exception_type="RuntimeError",
        message="boom",
        traceback="",
    )

    def fake_map(fn, tasks, n_jobs=None):
        return MapOutcome(results=tuple(None for _ in tasks), failures=(failure,))

    monkeypatch.setattr(module, "map_shapes", fake_map)
    with pytest.raises(ShapeProcessingError, match="shp-3-05"):
        transport_model(DETROIT, route_types=(3,), parallel=True, ncores=2)


def test_parallel_and_serial_skip_counters_agree() -> None:
    serial = transport_model(DATA / "synthetic_skips.zip", parallel=False)
    parallel = transport_model(DATA / "synthetic_skips.zip", parallel=True, ncores=2)
    assert serial.manifest.counts == parallel.manifest.counts
    assert serial.manifest.skipped == parallel.manifest.skipped


def test_all_skip_branches_in_one_fixture() -> None:
    model = transport_model(DATA / "synthetic_skips.zip", parallel=False)
    assert model.manifest.skipped == {
        "dropped_route_type": 1,
        "dropped_service": 0,
        "skipped_no_shape": 1,
        "skipped_short_shape": 1,
        "skipped_short_trip": 1,
        "skipped_no_timed_stops": 0,
        "zero_duration_intervals": 1,
    }
    assert set(model.table["trip_id"].astype(str)) == {"T_OK", "T_FAR", "T_ZERO"}
    assert any("T_FAR" in warning and "snapped" in warning for warning in model.manifest.warnings)


def test_empty_service_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="service_ids must not be empty"):
        transport_model(DETROIT, service_ids=[], parallel=False)
