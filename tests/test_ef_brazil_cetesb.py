"""CETESB EF lookup tests."""

from __future__ import annotations

import pytest

from gtfs2emis_py.ef import (
    CETESB_VEHICLES,
    EmissionFactors,
    MissingEmissionFactorError,
    UnsupportedPollutantError,
    ef_brazil_cetesb,
    load_ef_table,
)
from gtfs2emis_py.pollutants import UnknownPollutantError, pollutant_unit
from tests._fixtures import packaged_ef_table

pytestmark = pytest.mark.skipif(
    not packaged_ef_table("brazil_cetesb"),
    reason="brazil_cetesb is fetch-only, so this tree has no table to read",
)

UNSUPPORTED = {"FC", "FS", "KML"}


def test_single_combination_matches_the_table() -> None:
    database = load_ef_table("brazil_cetesb")
    expected = database[(database["pollutant"] == "CO") & (database["model_year"] == 2011)][
        "BUS_URBAN_D"
    ].iloc[0]
    factors = ef_brazil_cetesb("CO", "BUS_URBAN_D", 2011)
    assert isinstance(factors, EmissionFactors)
    assert len(factors) == 1
    assert factors.table.loc[0, "ef"] == pytest.approx(expected)
    assert factors.table.loc[0, "unit"] == "g/km"
    assert factors.process == "hot_exhaust"


def test_grid_is_the_cartesian_product() -> None:
    factors = ef_brazil_cetesb(["CO", "NOx"], ["BUS_URBAN_D", "BUS_ARTIC_D"], [2010, 2011])
    assert len(factors) == 8
    assert set(factors.table["pollutant"]) == {"CO", "NOx"}
    assert set(factors.table["veh_type"]) == {"BUS_URBAN_D", "BUS_ARTIC_D"}
    assert set(factors.table["model_year"]) == {2010, 2011}


def test_every_supported_pollutant_looks_up() -> None:
    database = load_ef_table("brazil_cetesb")
    for pollutant, group in database.groupby("pollutant"):
        if pollutant in UNSUPPORTED:
            continue
        valid = group[group["BUS_URBAN_D"].notna()]
        if valid.empty:
            continue
        row = valid.iloc[0]
        factors = ef_brazil_cetesb(pollutant, "BUS_URBAN_D", int(row["model_year"]))
        assert len(factors) == 1
        assert factors.table.loc[0, "ef"] == pytest.approx(float(row["BUS_URBAN_D"]))


def test_energy_metrics_carry_their_own_unit() -> None:
    factors = ef_brazil_cetesb("gCO2/KWH", "BUS_URBAN_D", 2011)
    assert factors.table.loc[0, "unit"] == "g/kWh"
    assert pollutant_unit("gCO2/KWH") == "g/kWh"


def test_unsupported_pollutant_names_the_reason() -> None:
    with pytest.raises(UnsupportedPollutantError, match="FC"):
        ef_brazil_cetesb("FC", "BUS_URBAN_D", 2011)


def test_unknown_pollutant_fails_before_the_lookup() -> None:
    with pytest.raises(UnknownPollutantError):
        ef_brazil_cetesb("CO3", "BUS_URBAN_D", 2011)


def test_unknown_vehicle_lists_the_supported_classes() -> None:
    with pytest.raises(ValueError, match="BUS_URBAN_D"):
        ef_brazil_cetesb("CO", "BUS_MAGIC", 2011)


def test_year_outside_the_table_range() -> None:
    with pytest.raises(ValueError, match="1960"):
        ef_brazil_cetesb("CO", "BUS_URBAN_D", 1950)


def test_missing_combination_is_an_error_not_a_zero() -> None:
    database = load_ef_table("brazil_cetesb")
    null_row = database[database["BUS_URBAN_D"].isna()].iloc[0]
    with pytest.raises(MissingEmissionFactorError, match="no emission factor"):
        ef_brazil_cetesb(str(null_row["pollutant"]), "BUS_URBAN_D", int(null_row["model_year"]))


def test_positions_and_dtypes() -> None:
    factors = ef_brazil_cetesb(["CO", "NOx"], list(CETESB_VEHICLES), [2010, 2011])
    assert list(factors.table.columns) == ["pollutant", "veh_type", "model_year", "ef", "unit"]
    assert factors.table["ef"].dtype == "float64"
    assert factors.table["model_year"].dtype == "int32"
    assert factors.table["unit"].dtype == "string"
    assert factors.units["CO"] == "g/km"


def test_to_frame_returns_a_copy() -> None:
    factors = ef_brazil_cetesb("CO", "BUS_URBAN_D", 2011)
    frame = factors.to_frame()
    frame.loc[0, "ef"] = -1.0
    assert factors.table.loc[0, "ef"] > 0


def test_empty_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ef_brazil_cetesb([], "BUS_URBAN_D", 2011)
