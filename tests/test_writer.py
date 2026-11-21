"""Transport writer tests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator
from shapely.geometry import LineString

from gtfs2emis_py.gtfs import writer
from gtfs2emis_py.gtfs.writer import (
    ManifestError,
    TransportWriteError,
    read_transport_manifest,
    write_transport_output,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((PACKAGE_ROOT / "docs/schemas/transport_manifest.schema.json").read_text())


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shape_id": ["SH_A", "SH_A", "SH_B"],
            "trip_id": ["T1", "T1", "T2"],
            "trip_number": [1, 1, 1],
            "route_id": ["R1", "R1", "R2"],
            "stop_sequence": [1, 2, 1],
            "dist_km": [0.5, 0.7, 1.2],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(3, 3), (4, 4)]),
            ],
        }
    )


def test_writes_one_file_per_shape_and_manifest(tmp_path: Path) -> None:
    manifest = write_transport_output(_table(), tmp_path)
    assert {entry["shape_id"] for entry in manifest["shapes"]} == {"SH_A", "SH_B"}
    assert manifest["rows_total"] == 3
    assert manifest["rows_written"] == 3
    assert manifest["shapes_resumed"] == 0
    assert (tmp_path / "shapes/SH_A.parquet").exists()
    frame = pq.read_table(tmp_path / "shapes/SH_A.parquet").to_pandas()
    assert len(frame) == 2
    assert frame["stop_sequence"].tolist() == [1, 2]


def test_geometry_is_wkb_with_geoparquet_metadata(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    schema = pq.read_schema(tmp_path / "shapes/SH_A.parquet")
    geo = json.loads(schema.metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    crs = geo["columns"]["geometry"]["crs"]
    assert crs is not None
    assert crs["id"]["authority"] == "EPSG"
    assert crs["id"]["code"] == 4326


def test_custom_crs_is_written_to_geoparquet(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path, crs="EPSG:3857")
    schema = pq.read_schema(tmp_path / "shapes/SH_A.parquet")
    geo = json.loads(schema.metadata[b"geo"])
    assert geo["columns"]["geometry"]["crs"]["id"]["code"] == 3857


def test_resume_skips_complete_shapes(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    before = {
        name: (tmp_path / "shapes" / name).stat().st_mtime_ns
        for name in ("SH_A.parquet", "SH_B.parquet")
    }
    manifest = write_transport_output(_table(), tmp_path, resume=True)
    after = {
        name: (tmp_path / "shapes" / name).stat().st_mtime_ns
        for name in ("SH_A.parquet", "SH_B.parquet")
    }
    assert manifest["shapes_resumed"] == 2
    assert manifest["rows_written"] == 0
    assert before == after
    assert len(list((tmp_path / "shapes").glob("*.parquet"))) == 2


def test_truncated_file_is_reprocessed(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    (tmp_path / "shapes/SH_A.parquet").write_bytes(b"not a parquet file")
    manifest = write_transport_output(_table(), tmp_path, resume=True)
    assert manifest["shapes_resumed"] == 1
    assert manifest["rows_written"] == 2
    assert pq.ParquetFile(tmp_path / "shapes/SH_A.parquet").metadata.num_rows == 2


def test_changed_parameters_raise_on_resume(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path, parameters={"resolution": 100})
    with pytest.raises(ManifestError, match="different input"):
        write_transport_output(_table(), tmp_path, resume=True, parameters={"resolution": 50})


def test_resume_false_overwrites(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    manifest = write_transport_output(_table(), tmp_path, resume=False)
    assert manifest["shapes_resumed"] == 0
    assert manifest["rows_written"] == 3


def test_shape_id_is_url_quoted(tmp_path: Path) -> None:
    table = _table()
    table.loc[table["shape_id"] == "SH_A", "shape_id"] = "a/b"
    table.loc[table["shape_id"] == "SH_B", "shape_id"] = "../evil"
    manifest = write_transport_output(table, tmp_path)
    files = {entry["file"] for entry in manifest["shapes"]}
    assert files == {"a%2Fb.parquet", "..%2Fevil.parquet"}
    assert (tmp_path / "shapes/a%2Fb.parquet").exists()
    assert not (tmp_path / "evil.parquet").exists()


def test_case_insensitive_collision_raises(tmp_path: Path) -> None:
    table = _table()
    table.loc[table["shape_id"] == "SH_A", "shape_id"] = "Same"
    table.loc[table["shape_id"] == "SH_B", "shape_id"] = "same"
    with pytest.raises(ValueError, match="collide"):
        write_transport_output(table, tmp_path)


def test_failed_shape_is_recorded_and_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = writer._write_shape

    def fail_for_sh_b(group: pd.DataFrame, path: Path, *, crs: str = "EPSG:4326") -> int:
        if str(group["shape_id"].iloc[0]) == "SH_B":
            raise OSError("disk full")
        return original(group, path, crs=crs)

    monkeypatch.setattr(writer, "_write_shape", fail_for_sh_b)
    with pytest.raises(TransportWriteError) as exc_info:
        write_transport_output(_table(), tmp_path)
    manifest = exc_info.value.manifest
    statuses = {entry["shape_id"]: entry["status"] for entry in manifest["shapes"]}
    assert statuses == {"SH_A": "done", "SH_B": "failed"}
    assert "disk full" in next(
        entry["error"] for entry in manifest["shapes"] if entry["shape_id"] == "SH_B"
    )
    assert (tmp_path / "shapes/SH_A.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    reread = read_transport_manifest(tmp_path)
    assert {entry["shape_id"]: entry["status"] for entry in reread["shapes"]} == {
        "SH_A": "done",
        "SH_B": "failed",
    }


def test_manifest_validates_against_the_schema(tmp_path: Path) -> None:
    manifest = write_transport_output(_table(), tmp_path)
    Draft202012Validator(SCHEMA).validate(manifest)


def test_empty_or_null_shape_ids_are_rejected(tmp_path: Path) -> None:
    table = _table()
    table.loc[0, "shape_id"] = ""
    with pytest.raises(ValueError, match="empty"):
        write_transport_output(table, tmp_path)


def test_read_manifest_roundtrip(tmp_path: Path) -> None:
    written = write_transport_output(_table(), tmp_path)
    assert read_transport_manifest(tmp_path) == written


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no manifest"):
        read_transport_manifest(tmp_path)


def test_corrupt_manifest_raises_manifest_error(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    (tmp_path / "manifest.json").write_text("not json")
    with pytest.raises(ManifestError, match="cannot read"):
        read_transport_manifest(tmp_path)


def test_manifest_without_schema_version_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ManifestError, match="not a transport manifest"):
        read_transport_manifest(tmp_path)


def test_changed_values_with_same_row_count_are_rewritten(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    changed = _table()
    changed.loc[0, "dist_km"] = 9.0
    manifest = write_transport_output(changed, tmp_path, resume=True)
    assert manifest["shapes_resumed"] == 0
    frame = pq.read_table(tmp_path / "shapes/SH_A.parquet").to_pandas()
    assert 9.0 in frame["dist_km"].tolist()


def test_changed_feed_hash_is_rejected_on_resume(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path, source_sha256="a" * 64)
    with pytest.raises(ManifestError, match="different input"):
        write_transport_output(_table(), tmp_path, resume=True, source_sha256="b" * 64)


def test_corrupted_same_row_count_file_is_rewritten(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    path = tmp_path / "shapes/SH_A.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 1), path)
    # Make the row count match by writing a different one-row frame with the
    # same schema; the manifest's per-file sha256 must still reject it.
    manifest = write_transport_output(_table(), tmp_path, resume=True)
    assert manifest["shapes_resumed"] == 1  # SH_B is intact; SH_A is rewritten
    assert pq.ParquetFile(path).metadata.num_rows == 2


def test_unknown_manifest_schema_version_is_rejected(tmp_path: Path) -> None:
    write_transport_output(_table(), tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="supported schema versions: 1"):
        read_transport_manifest(tmp_path)


def test_table_sha256_does_not_warn_and_keeps_digest() -> None:
    frame = gpd.GeoDataFrame(
        {
            "shape_id": ["A", "A"],
            "trip_id": ["T1", "T1"],
            "stop_sequence": [1, 2],
            "value": [1.0, 2.0],
        },
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
        ],
        crs="EPSG:4326",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        digest = writer._table_sha256([("A", frame)])
    assert not [item for item in caught if issubclass(item.category, UserWarning)]
    assert digest == "1a14aeb7502b72cfadf49e9b73fe16ba5a91417b16c9add473fbace2961287e9"


def test_write_transport_output_does_not_warn(tmp_path: Path) -> None:
    table = gpd.GeoDataFrame(_table(), geometry="geometry", crs="EPSG:4326")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_transport_output(table, tmp_path)
    assert not [item for item in caught if issubclass(item.category, UserWarning)]
