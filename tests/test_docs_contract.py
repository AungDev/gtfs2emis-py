"""Documentation follows the implemented public surface."""

from __future__ import annotations

from pathlib import Path

import pytest

import gtfs2emis_py
from gtfs2emis_py import emission_model, transport_model
from gtfs2emis_py.analysis import emis_grid
from gtfs2emis_py.ef import ef_brazil_cetesb, ef_europe_emep, ef_usa_emfac, ef_usa_moves
from gtfs2emis_py.emission_writer import write_emission_output
from gtfs2emis_py.fleet import load_fleet

DOCS = Path(__file__).parents[1] / "docs"
ROOT_CONTRACT = Path(__file__).parents[2] / "docs/transport_model_contract.md"


def test_documented_public_symbols_import() -> None:
    for symbol in (
        gtfs2emis_py,
        transport_model,
        emission_model,
        load_fleet,
        emis_grid,
        write_emission_output,
        ef_brazil_cetesb,
        ef_europe_emep,
        ef_usa_emfac,
        ef_usa_moves,
    ):
        assert symbol is not None


def test_docs_do_not_claim_unimplemented_output_or_wear() -> None:
    offenders: list[str] = []
    for path in DOCS.rglob("*.md"):
        text = path.read_text()
        for phrase in (
            "output_path`/`resume` are reserved",
            "output_path/resume are reserved",
            "wear processes belong to",
            "all 20 `Function_ID`",
            "all 20 Function_ID",
        ):
            if phrase in text:
                offenders.append(f"{path.name}: {phrase}")
    assert not offenders, offenders


@pytest.mark.skipif(
    not ROOT_CONTRACT.exists(), reason="development-only root contract is not published"
)
def test_transport_contract_describes_implemented_algorithms() -> None:
    contract = ROOT_CONTRACT.read_text()
    assert "each consecutive pair is measured with a WGS84 geodesic" not in contract
    assert "own polyline length" not in contract
    assert "6,371,010 m" in contract
    assert "S1ChordAngle" in contract
    assert "nearest2" in contract
    assert "haversine" in contract
    assert "atan2(|a × b|" not in contract
    assert "ceil(d / spatial_resolution)" not in contract
    assert "Geod.inv" not in contract
