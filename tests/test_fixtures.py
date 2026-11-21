"""Fixture contract tests.

The suite depends on fixtures that are committed rather than generated at test
time, so their shape is asserted here: each sample feed must load and validate,
each fleet table must carry the columns the emission models expect, the DEM must
be usable for slope classification, and the synthetic feed must still contain
the edge cases it exists for.

Provenance, hashes and the size budget are documented in ``tests/data/README.md``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import rasterio

from gtfs2emis_py.gtfs import read_gtfs

DATA = Path(__file__).resolve().parent / "data"
PACKAGE_ROOT = DATA.parents[1]
SYNTHETIC = DATA / "synthetic_edge_cases.zip"

FEEDS = ("bra_cur_gtfs.zip", "irl_dub_gtfs.zip", "usa_det_gtfs.zip")

# fleet table -> columns its consumer will require
FLEETS = {
    "usa_det_fleet.txt": {"year", "N", "fuel", "fleet_composition"},
    "irl_dub_fleet.txt": {"veh_type", "euro", "fuel", "N", "fleet_composition", "tech"},
    "bra_cur_fleet.txt": {
        "year",
        "euro",
        "shape_id",
        "type_name_br",
        "veh_type",
        "total",
        "fuel",
    },
}


def _read_member(archive: Path, member: str) -> pd.DataFrame:
    """Read one CSV member out of a GTFS zip."""
    with zipfile.ZipFile(archive) as zf, zf.open(member) as handle:
        return pd.read_csv(handle, dtype=str)


@pytest.mark.parametrize("name", FEEDS)
def test_sample_feed_loads_and_validates(name: str) -> None:
    gf = read_gtfs(DATA / name)
    assert gf.ok is True, gf.validation.errors if gf.validation else None
    assert gf.validation is not None
    assert not [
        warning
        for warning in gf.validation.warnings
        if not warning.startswith("clean=True dropped rows:")
    ]
    feed = gf.feed
    assert not feed.trips.empty
    assert not feed.stops.empty
    assert not feed.shapes.empty
    # All three sample feeds are urban bus services; the emission models are
    # bus-only.
    assert set(feed.routes["route_type"]) == {3}


@pytest.mark.parametrize(("name", "expected"), FLEETS.items())
def test_fleet_table_schema(name: str, expected: set[str]) -> None:
    fleet = pd.read_csv(DATA / name)
    assert expected <= set(fleet.columns)
    assert not fleet.empty
    if "fleet_composition" in fleet.columns:
        composition = fleet["fleet_composition"]
        assert composition.between(0, 1).all(), "fleet composition must be a share"
        assert composition.sum() == pytest.approx(1.0, abs=1e-6)
    count_column = "N" if "N" in fleet.columns else "total"
    assert (fleet[count_column] > 0).all()


def test_curitiba_fleet_mostly_joins_the_feed() -> None:
    """The per-shape Curitiba fleet joins the shipped feed, with two upstream strays.

    Checked against the raw ``shapes.txt`` in the zip: 47 of the fleet's 49
    ``shape_id`` values exist there, and two (``1709``, ``1898``) do not. The
    mismatch is in the reference data, so the test records it rather than
    pretending a full join.

    ``read_gtfs`` cleans the feed, and ``Feed.clean`` drops shapes no trip
    references (205 raw shapes become 179), so a post-clean join loses six more
    fleet ids. The fleet table is a raw-input artefact; tests that use it must
    join against the feed they were given, not assume every row lands.
    """
    fleet = pd.read_csv(DATA / "bra_cur_fleet.txt", dtype={"shape_id": "string"})
    shapes = _read_member(DATA / "bra_cur_gtfs.zip", "shapes.txt")
    fleet_shapes = set(fleet["shape_id"])
    strays = fleet_shapes - set(shapes["shape_id"])
    assert strays == {"1709", "1898"}
    assert len(fleet_shapes.intersection(set(shapes["shape_id"]))) / len(fleet_shapes) >= 0.95


def test_dem_fixture_is_usable_for_slope_classification() -> None:
    with rasterio.open(DATA / "bra_cur-srtm.tif") as src:
        assert src.crs is not None
        assert src.nodata is not None
        elevation = src.read(1)
        valid = elevation[elevation != src.nodata]
        assert valid.size > 0
        # Slope classification needs relief, not a flat plane.
        assert float(valid.max()) > float(valid.min())


def test_synthetic_feed_contains_the_edge_cases() -> None:
    shapes = _read_member(SYNTHETIC, "shapes.txt")
    stop_times = _read_member(SYNTHETIC, "stop_times.txt")
    frequencies = _read_member(SYNTHETIC, "frequencies.txt")

    per_shape = shapes.groupby("shape_id")
    assert len(per_shape.get_group("SH_ONE")) == 1, "one-point shape missing"

    duplicate = per_shape.get_group("SH_DUP")[["shape_pt_lat", "shape_pt_lon"]]
    assert duplicate.duplicated().any(), "no duplicate consecutive shape point"

    loop = per_shape.get_group("SH_LOOP")[["shape_pt_lat", "shape_pt_lon"]]
    assert loop.iloc[0].tolist() == loop.iloc[-1].tolist(), "loop shape is not closed"

    assert stop_times["arrival_time"].str.startswith(("24:", "25:")).any(), (
        "no service-day time past 24:00:00"
    )
    assert "shape_dist_traveled" not in shapes.columns
    assert (
        "shape_dist_traveled"
        not in pd.read_csv(zipfile.ZipFile(SYNTHETIC).open("stop_times.txt"), dtype=str).columns
    )
    assert set(frequencies["trip_id"]) == {"T_FREQ"}


def test_synthetic_feed_loads_and_survives_cleaning() -> None:
    gf = read_gtfs(SYNTHETIC)
    assert gf.ok is True
    feed = gf.feed
    assert set(feed.trips["trip_id"]) == {
        "T_ONEPOINT",
        "T_DUP",
        "T_LOOP",
        "T_LATE",
        "T_FREQ",
    }
    shape_sizes = feed.shapes.groupby("shape_id").size()
    assert shape_sizes["SH_ONE"] == 1
    assert feed.stop_times["arrival_time"].max() == "25:20:00"
    assert feed.frequencies is not None and not feed.frequencies.empty


def test_synthetic_feed_is_reproducible(tmp_path: Path) -> None:
    regenerated = tmp_path / "synthetic_edge_cases.zip"
    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts" / "make_synthetic_feed.py"),
            "--out",
            str(regenerated),
        ],
        check=True,
        capture_output=True,
    )
    committed = hashlib.sha256(SYNTHETIC.read_bytes()).hexdigest()
    rebuilt = hashlib.sha256(regenerated.read_bytes()).hexdigest()
    assert rebuilt == committed, "regenerate with scripts/make_synthetic_feed.py"


def test_fixture_size_budget() -> None:
    total = sum(path.stat().st_size for path in DATA.iterdir() if path.is_file())
    assert total < 1_000_000, f"fixtures total {total} bytes; budget is 1 MB"
