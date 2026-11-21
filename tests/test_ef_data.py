"""EF data pipeline tests."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pyarrow as pa
import pytest

from gtfs2emis_py.ef import EFDataError, list_ef_tables, load_ef_table
from gtfs2emis_py.ef import data as ef_data
from tests._fixtures import packaged_ef_table, requires_ef_table

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PACKAGE_ROOT / "data-raw" / "raw"
# Tables that ship as Parquet in this tree; the rest are fetch-only.
PACKAGED = tuple(
    name
    for name in ("brazil_cetesb", "europe_emep", "usa_emfac", "usa_moves")
    if packaged_ef_table(name)
)

SHAPES = {
    "brazil_cetesb": (1098, 9),
    "europe_emep": (9913, 22),
    "usa_emfac": (79198, 8),
    "usa_moves": (198144, 9),
}

COLUMNS = {
    "brazil_cetesb": [
        "calendar_year",
        "model_year",
        "pollutant",
        "Proconve_HDV",
        "Euro_EqHDV",
        "BUS_URBAN_D",
        "BUS_MICRO_D",
        "BUS_COACH_D",
        "BUS_ARTIC_D",
    ],
    "europe_emep": [
        "Category",
        "Fuel",
        "Segment",
        "Euro",
        "Technology",
        "Pol",
        "Slope",
        "Load",
        "Vmin",
        "Vmax",
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zita",
        "Hta",
        "RF",
        "Function",
        "Function_ID",
        "k",
        "Thita",
    ],
    "usa_emfac": [
        "reference_year",
        "model_year",
        "fuel",
        "vkt",
        "ef",
        "pollutant",
        "lower_speed_interval",
        "upper_speed_interval",
    ],
    "usa_moves": [
        "fuel_type",
        "reference_year",
        "model_year",
        "pollutant",
        "lower_speed_interval",
        "upper_speed_interval",
        "ef",
        "source_type",
        "id_speed",
    ],
}


def _manifest() -> dict:
    return json.loads(resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_text())


@pytest.mark.parametrize(("name", "shape"), [(name, SHAPES[name]) for name in PACKAGED])
def test_table_shapes(name: str, shape: tuple[int, int]) -> None:
    table = load_ef_table(name)
    assert table.shape == shape


@pytest.mark.parametrize("name", PACKAGED)
def test_column_names_are_verbatim(name: str) -> None:
    assert list(load_ef_table(name).columns) == COLUMNS[name]


def test_list_ef_tables() -> None:
    assert list_ef_tables() == tuple(sorted(SHAPES))


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        (alias, canonical)
        for alias, canonical in [
            ("ef_brazil_cetesb_db", "brazil_cetesb"),
            ("ef_europe_emep_db", "europe_emep"),
            ("usa_emfac", "usa_emfac"),
        ]
        if packaged_ef_table(canonical)
    ],
)
def test_loader_accepts_aliases(alias: str, canonical: str) -> None:
    assert load_ef_table(alias).shape == SHAPES[canonical]


def test_unknown_table_lists_the_available_names() -> None:
    with pytest.raises(EFDataError, match="brazil_cetesb"):
        load_ef_table("no_such_table")


@pytest.mark.parametrize("name", ["brazil_cetesb", "usa_emfac", "usa_moves"])
def test_fetch_only_tables_point_at_the_original_rda(name: str) -> None:
    remote = _manifest()["tables"][name]["remote"]
    assert remote["format"] == "rda"
    assert remote["url"].startswith(
        "https://raw.githubusercontent.com/ipeaGIT/gtfs2emis/"
        "0dbca7d779857145888ab695f36dc9e71ff57d98/data/"
    )
    assert remote["url"].endswith(remote["raw_file"])
    assert len(remote["raw_sha256"]) == 64


def test_fetch_only_table_requires_an_explicit_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_data = tmp_path / "package"
    package_data.mkdir()
    (package_data / "manifest.json").write_bytes(
        resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_bytes()
    )
    monkeypatch.setattr("gtfs2emis_py.ef.data._data_dir", lambda: package_data)
    monkeypatch.setenv("GTFS2EMIS_CACHE", str(tmp_path / "cache"))
    from gtfs2emis_py.ef import data as ef_data

    ef_data._CACHE.clear()
    with pytest.raises(EFDataError, match="fetch=True"):
        load_ef_table("usa_emfac")


def test_fetch_verifies_the_pinned_hash_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_data = tmp_path / "package"
    package_data.mkdir()
    (package_data / "manifest.json").write_bytes(
        resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_bytes()
    )
    raw = RAW_DIR / "ef_usa_emfac_db.rda"
    if not raw.is_file():
        pytest.skip("development raw .rda fixture is not present")
    pytest.importorskip("pyreadr")
    manifest = json.loads(
        resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_text()
    )
    manifest["tables"]["usa_emfac"]["remote"]["url"] = raw.as_uri()
    (package_data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("gtfs2emis_py.ef.data._data_dir", lambda: package_data)
    monkeypatch.setenv("GTFS2EMIS_CACHE", str(tmp_path / "cache"))
    from gtfs2emis_py.ef import data as ef_data

    ef_data._CACHE.clear()
    table = load_ef_table("usa_emfac", fetch=True)
    assert table.shape == SHAPES["usa_emfac"]
    assert (tmp_path / "cache" / "ef" / "ef_usa_emfac_db.rda").is_file()
    ef_data._CACHE.clear()
    assert load_ef_table("usa_emfac").shape == SHAPES["usa_emfac"]


def test_fetch_rejects_a_changed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_data = tmp_path / "package"
    package_data.mkdir()
    (package_data / "manifest.json").write_bytes(
        resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_bytes()
    )
    source = tmp_path / "source"
    source.mkdir()
    changed = source / "ef_usa_emfac_db.rda"
    changed.write_bytes(b"not the pinned data")
    manifest = json.loads(
        resources.files("gtfs2emis_py.data.ef").joinpath("manifest.json").read_text()
    )
    manifest["tables"]["usa_emfac"]["remote"]["url"] = changed.as_uri()
    (package_data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("gtfs2emis_py.ef.data._data_dir", lambda: package_data)
    monkeypatch.setenv("GTFS2EMIS_CACHE", str(tmp_path / "cache"))
    from gtfs2emis_py.ef import data as ef_data

    ef_data._CACHE.clear()
    with pytest.raises(EFDataError, match="expected .* got .* update the manifest deliberately"):
        load_ef_table("usa_emfac", fetch=True)


@requires_ef_table("brazil_cetesb")
def test_loader_returns_a_copy() -> None:
    first = load_ef_table("brazil_cetesb")
    first.loc[0, "pollutant"] = "XX"
    assert load_ef_table("brazil_cetesb").loc[0, "pollutant"] != "XX"


@requires_ef_table("brazil_cetesb", "usa_emfac", "usa_moves")
def test_reference_nulls_are_preserved() -> None:
    cetesb = load_ef_table("brazil_cetesb")
    for column in ("BUS_URBAN_D", "BUS_MICRO_D", "BUS_COACH_D", "BUS_ARTIC_D"):
        assert cetesb[column].isna().sum() == 183
    emep = load_ef_table("europe_emep")
    assert emep["Delta"].isna().sum() == 368
    assert emep["Epsilon"].isna().sum() == 494
    assert load_ef_table("usa_emfac").isna().sum().sum() == 0
    assert load_ef_table("usa_moves").isna().sum().sum() == 0


def test_manifest_is_the_full_bill_of_materials() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert set(manifest["tables"]) == set(SHAPES)
    required = {
        "file",
        "r_object",
        "rows",
        "columns",
        "sha256",
        "content_sha256",
        "source",
        "edition",
        "source_url",
        "retrieved",
        "licence",
        "build_script",
        "raw_inputs",
        "pyarrow_version",
        "schema",
    }
    for name, entry in manifest["tables"].items():
        assert required <= set(entry), name
        assert entry["rows"] == SHAPES[name][0]
        assert entry["columns"] == SHAPES[name][1]
        assert len(entry["sha256"]) == 64
        assert len(entry["content_sha256"]) == 64
        assert entry["raw_inputs"] and len(entry["raw_inputs"][0]["sha256"]) == 64
        assert [column["name"] for column in entry["schema"]] == COLUMNS[name]


def test_schema_types_match_the_parquet_files() -> None:
    import pyarrow.parquet as pq

    manifest = _manifest()
    for name, entry in manifest["tables"].items():
        if name not in PACKAGED:
            continue
        path = resources.files("gtfs2emis_py.data.ef").joinpath(entry["file"])
        schema = pq.read_schema(path)
        assert [(field.name, str(field.type)) for field in schema] == [
            (column["name"], column["type"]) for column in entry["schema"]
        ], name


@pytest.mark.skipif(
    not (PACKAGE_ROOT / "data-raw").is_dir(),
    reason="the raw EF generators are not part of this tree",
)
@pytest.mark.parametrize("script", [f"ef_{name}_db.py" for name in PACKAGED])
def test_check_packaged_passes(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "data-raw" / script), "--check-packaged"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "match the manifest" in result.stdout


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw .rda inputs are git-ignored")
@pytest.mark.parametrize(
    ("script", "raw_name"),
    [
        ("ef_brazil_cetesb_db.py", "ef_brazil_cetesb_db.rda"),
        ("ef_europe_emep_db.py", "ef_europe_emep_db.rda"),
    ],
)
def test_rebuild_reproduces_content_sha256(script: str, raw_name: str) -> None:
    pytest.importorskip("pyreadr")
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "data-raw" / script),
            "--raw",
            str(RAW_DIR / raw_name),
            "--check",
        ],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "content_sha256 matches" in result.stdout


def test_content_hash_is_independent_of_arrow_chunks() -> None:
    from gtfs2emis_py.ef.data import _content_sha256

    column = pa.chunked_array([[1, 2, 3]])
    one = pa.table({"a": column, "b": ["x", "y", None]})
    split = pa.table(
        {
            "a": pa.chunked_array([[1], [2, 3]]),
            "b": pa.chunked_array([["x"], ["y", None]]),
        }
    )
    assert _content_sha256(one) == _content_sha256(split)


def test_packaged_manifest_errors_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ef_data, "_data_dir", lambda: tmp_path)
    with pytest.raises(EFDataError, match="missing from the package data"):
        ef_data._read_manifest()
    (tmp_path / "manifest.json").write_text("not json")
    with pytest.raises(EFDataError, match="not valid JSON"):
        ef_data._read_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(EFDataError, match="unsupported manifest schema_version"):
        ef_data._read_manifest()


def test_cache_and_remote_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GTFS2EMIS_CACHE", str(tmp_path))
    assert ef_data._cache_dir() == tmp_path / "ef"
    monkeypatch.setenv("GTFS2EMIS_DATA_URL", "https://example.invalid/data/")
    entry = {"file": "table.parquet", "remote": {"file": "table.bin"}}
    assert ef_data._remote_url(entry) == "https://example.invalid/data/table.bin"
    assert ef_data._raw_file(entry) == "table.rda"
    with pytest.raises(EFDataError, match="raw_sha256"):
        ef_data._raw_sha256({"file": "table.parquet", "remote": {}})
    with pytest.raises(EFDataError, match="no remote URL"):
        ef_data._read_payload(
            {"file": "missing.parquet", "sha256": "0" * 64},
            "missing",
            fetch=True,
        )
