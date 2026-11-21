"""emission writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from gtfs2emis_py.emission_writer import (
    EmissionWriteError,
    ManifestError,
    read_emission_manifest,
    write_emission_output,
)


class _Result:
    def __init__(self, table: pd.DataFrame) -> None:
        self.emi = table
        self.metadata = {
            "ef_model": "ef_brazil_cetesb",
            "pollutant": ["CO"],
            "process": ["hot_exhaust"],
        }


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1, 2],
            "shape_id": ["SH1", "SH1", "SH/2"],
            "trip_id": ["T1", "T1", "T2"],
            "stop_sequence": [0, 1, 0],
            "vehicle_type": ["BUS_URBAN_D"] * 3,
            "pollutant": ["CO"] * 3,
            "process": ["hot_exhaust"] * 3,
            "value": [1.0, 2.0, 3.0],
            "unit": ["g/km"] * 3,
        }
    )


def test_writes_one_file_per_shape_and_manifest(tmp_path: Path) -> None:
    written = write_emission_output(_Result(_table()), tmp_path)
    shapes = sorted((tmp_path / "shapes").iterdir())
    assert len(shapes) == 2
    assert any(name.name == "SH%2F2.parquet" for name in shapes)
    manifest = read_emission_manifest(tmp_path)
    assert manifest["rows"] == 3
    assert len(manifest["shapes"]) == 2
    assert {entry["shape_id"] for entry in manifest["shapes"]} == {"SH1", "SH/2"}
    assert written.manifest == manifest


def test_manifest_matches_documented_schema(tmp_path: Path) -> None:
    write_emission_output(_Result(_table()), tmp_path)
    schema = json.loads(
        (Path(__file__).parents[1] / "docs/schemas/emission_manifest.schema.json").read_text()
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    Draft202012Validator(schema).validate(manifest)


def test_resume_reuses_complete_output(tmp_path: Path) -> None:
    write_emission_output(_Result(_table()), tmp_path)
    manifest_path = tmp_path / "manifest.json"
    before = manifest_path.stat().st_mtime_ns
    write_emission_output(_Result(_table()), tmp_path, resume=True)
    assert manifest_path.stat().st_mtime_ns == before


def test_resume_rewrites_after_a_missing_file(tmp_path: Path) -> None:
    write_emission_output(_Result(_table()), tmp_path)
    manifest = read_emission_manifest(tmp_path)
    (tmp_path / manifest["shapes"][0]["file"]).unlink()
    write_emission_output(_Result(_table()), tmp_path, resume=True)
    assert all(
        (tmp_path / entry["file"]).exists() for entry in read_emission_manifest(tmp_path)["shapes"]
    )


def test_result_must_expose_a_dataframe_and_metadata(tmp_path: Path) -> None:
    class _BadFrame:
        emi = []
        metadata = {}

    class _BadMetadata:
        emi = _table()
        metadata = []

    with pytest.raises(TypeError, match="pandas DataFrame"):
        write_emission_output(_BadFrame(), tmp_path)
    with pytest.raises(TypeError, match="metadata mapping"):
        write_emission_output(_BadMetadata(), tmp_path)


def test_invalid_table_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape_id"):
        write_emission_output(_Result(_table().drop(columns=["shape_id"])), tmp_path)
    bad = _table()
    bad.loc[0, "value"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        write_emission_output(_Result(bad), tmp_path)


def test_output_path_must_be_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(EmissionWriteError, match="not a directory"):
        write_emission_output(_Result(_table()), target)


def test_read_manifest_rejects_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no emission manifest"):
        read_emission_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("not json")
    with pytest.raises(ManifestError, match="cannot read"):
        read_emission_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ManifestError, match="schema-version"):
        read_emission_manifest(tmp_path)


def test_resume_rewrites_when_values_change_with_the_same_rows(tmp_path: Path) -> None:
    write_emission_output(_Result(_table()), tmp_path)
    changed = _table()
    changed.loc[0, "value"] = 99.0
    write_emission_output(_Result(changed), tmp_path, resume=True)
    manifest = read_emission_manifest(tmp_path)
    values = [
        value
        for entry in manifest["shapes"]
        for value in pd.read_parquet(tmp_path / entry["file"])["value"].tolist()
    ]
    assert 99.0 in values


def test_resume_rewrites_when_the_request_changes(tmp_path: Path) -> None:
    write_emission_output(_Result(_table()), tmp_path)
    manifest_path = tmp_path / "manifest.json"
    before = manifest_path.stat().st_mtime_ns
    changed = _Result(_table())
    changed.metadata = {
        "ef_model": "ef_europe_emep",
        "pollutant": ["CO", "NOx"],
        "process": ["hot_exhaust", "tyre"],
    }
    write_emission_output(changed, tmp_path, resume=True)
    assert manifest_path.stat().st_mtime_ns != before
    manifest = read_emission_manifest(tmp_path)
    assert manifest["ef_model"] == "ef_europe_emep"
    assert manifest["pollutant"] == ["CO", "NOx"]


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing column"):
        write_emission_output(_Result(_table().drop(columns=["unit"])), tmp_path)


def test_negative_value_is_rejected(tmp_path: Path) -> None:
    table = _table()
    table.loc[0, "value"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        write_emission_output(_Result(table), tmp_path)
