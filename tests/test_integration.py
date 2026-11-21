"""End-to-end integration tests on the bundled feeds."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtfs2emis_py import emission_model, load_fleet, transport_model
from gtfs2emis_py.fleet import UnknownFleetModelError
from gtfs2emis_py.gtfs.reader import GtfsReadError
from tests._fixtures import packaged_ef_table, requires_ef_table

DATA = Path(__file__).parent / "data"
CONTRACT_COLUMNS = {
    "segment_id",
    "shape_id",
    "trip_id",
    "stop_sequence",
    "vehicle_type",
    "fleet_row",
    "pollutant",
    "process",
    "value",
    "unit",
}
RUNS = [
    run
    for run in [
        (
            "bra_cur_gtfs.zip",
            "bra_cur_fleet.txt",
            "ef_brazil_cetesb",
            282.6392512695265,
            1364.7581172558885,
        ),
        (
            "irl_dub_gtfs.zip",
            "irl_dub_fleet.txt",
            "ef_europe_emep",
            4795.231210158518,
            15712.060779037862,
        ),
        (
            "usa_det_gtfs.zip",
            "usa_det_fleet.txt",
            "ef_usa_moves",
            506.538144857717,
            986.3806007661527,
        ),
    ]
    # CETESB and MOVES ship fetch-only, so a release tree has no table to run
    # those feeds against and only the EMEP feed remains.
    if packaged_ef_table(run[2].removeprefix("ef_"))
]


@pytest.mark.slow
@pytest.mark.parametrize("feed,fleet_file,model,co_total,nox_total", RUNS)
def test_feed_end_to_end(
    feed: str, fleet_file: str, model: str, co_total: float, nox_total: float
) -> None:
    transport = transport_model(DATA / feed)
    fleet = load_fleet(DATA / fleet_file, ef_model=model)
    result = emission_model(transport, model, fleet, ["CO", "NOx"], reference_year=2020)
    assert set(result.emi.columns) == CONTRACT_COLUMNS
    assert result.emi["value"].notna().all()
    assert (result.emi["value"] >= 0).all()
    totals = result.emi.groupby("pollutant")["value"].sum()
    assert totals["CO"] == pytest.approx(co_total, rel=1e-6)
    assert totals["NOx"] == pytest.approx(nox_total, rel=1e-6)
    assert set(result.emi["unit"]) == {"g"}


@pytest.mark.slow
@requires_ef_table("brazil_cetesb")
def test_curitiba_is_stable_across_two_runs() -> None:
    transport = transport_model(DATA / "bra_cur_gtfs.zip")
    fleet = load_fleet(DATA / "bra_cur_fleet.txt", ef_model="ef_brazil_cetesb")
    first = emission_model(transport, "ef_brazil_cetesb", fleet, ["CO", "NOx"])
    second = emission_model(transport, "ef_brazil_cetesb", fleet, ["CO", "NOx"])
    assert first.emi.equals(second.emi)
    assert first.units == second.units


def test_missing_feed_raises_the_reader_error() -> None:
    with pytest.raises(GtfsReadError, match="Could not read GTFS feed"):
        transport_model(DATA / "does-not-exist.zip")


def test_unknown_fleet_model_raises_the_fleet_error() -> None:
    with pytest.raises(UnknownFleetModelError):
        load_fleet(DATA / "bra_cur_fleet.txt", ef_model="ef_not_a_model")
