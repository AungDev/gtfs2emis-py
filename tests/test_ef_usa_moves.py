"""MOVES3 EF lookup tests."""

from __future__ import annotations

import numpy as np
import pytest

from gtfs2emis_py.ef import (
    MissingEmissionFactorError,
    ef_usa_moves,
    load_ef_table,
)
from tests._fixtures import packaged_ef_table

pytestmark = pytest.mark.skipif(
    not packaged_ef_table("usa_moves"),
    reason="usa_moves is fetch-only, so this tree has no table to read",
)

R_POLLUTANTS = (
    "CH4",
    "CO",
    "CO2",
    "EC",
    "HONO",
    "N2O",
    "NH3",
    "NH4",
    "NO",
    "NO2",
    "NO3",
    "NOx",
    "PM10",
    "PM25",
    "SO2",
    "THC",
    "TOG",
    "VOC",
)


def test_single_value_matches_the_table() -> None:
    table = load_ef_table("usa_moves")
    expected = table[
        (table["pollutant"] == "CO")
        & (table["model_year"] == 2015)
        & (table["fuel_type"] == "D")
        & (table["reference_year"] == 2019)
        & (table["upper_speed_interval"] == 52.3)
    ]["ef"].iloc[0]
    factors = ef_usa_moves("CO", 2015, reference_year=2019, speed=50)
    assert len(factors) == 1
    assert factors.table.loc[0, "ef"] == pytest.approx(expected)
    assert factors.table.loc[0, "bin_lower_kmh"] == 44.3
    assert factors.table.loc[0, "bin_upper_kmh"] == 52.3
    assert factors.table.loc[0, "unit"] == "g/km"


def test_bin_boundaries_are_right_closed() -> None:
    factors = ef_usa_moves("CO", 2015, reference_year=2019, speed=[4, 4.1, 12.1, 12.2])
    assert factors.table["bin_upper_kmh"].tolist() == [4.0, 12.1, 12.1, 20.1]


def test_no_interpolation_within_a_bin() -> None:
    factors = ef_usa_moves("CO", 2015, reference_year=2019, speed=[50, 52])
    assert factors.table.loc[0, "ef"] == pytest.approx(factors.table.loc[1, "ef"])


def test_top_bin_and_its_limit() -> None:
    factors = ef_usa_moves("CO", 2015, reference_year=2019, speed=120.7)
    assert factors.table.loc[0, "bin_upper_kmh"] == 120.7
    with pytest.raises(ValueError, match="120.7"):
        ef_usa_moves("CO", 2015, reference_year=2019, speed=121)


def test_grid_is_pollutant_x_vehicle_x_speed() -> None:
    factors = ef_usa_moves(
        ["CO", "NOx"],
        2015,
        reference_year=2019,
        speed=[10, 50],
        fuel=["D", "CNG"],
    )
    assert len(factors) == 2 * 2 * 2
    assert set(factors.table["fuel"]) == {"D", "CNG"}


def test_pollutant_coverage_matches_the_reference_docs() -> None:
    table = load_ef_table("usa_moves")
    available = set(table["pollutant"].unique())
    assert set(R_POLLUTANTS) == available
    for pollutant in R_POLLUTANTS:
        factors = ef_usa_moves(pollutant, 2015, reference_year=2019, speed=50)
        assert len(factors) == 1


def test_ec_is_kj_per_km() -> None:
    factors = ef_usa_moves("EC", 2015, reference_year=2019, speed=50)
    assert factors.table.loc[0, "unit"] == "kJ/km"


def test_absent_reference_year_is_missing() -> None:
    with pytest.raises(MissingEmissionFactorError, match="2014"):
        ef_usa_moves("CO", 2015, reference_year=2014, speed=50)


def test_reference_year_above_the_validator_range_is_rejected() -> None:
    # The table has 2021 rows, but the reference validator stops at 2020.
    with pytest.raises(ValueError, match="2010-2020"):
        ef_usa_moves("CO", 2015, reference_year=2021, speed=50)


def test_model_year_range() -> None:
    with pytest.raises(ValueError, match="1989-2022"):
        ef_usa_moves("CO", 1985, reference_year=2019, speed=50)


def test_unknown_pollutant_and_fuel() -> None:
    with pytest.raises(ValueError, match="pollutant"):
        ef_usa_moves("MAGIC", 2015, reference_year=2019, speed=50)
    with pytest.raises(ValueError, match="fuel"):
        ef_usa_moves("CO", 2015, reference_year=2019, speed=50, fuel="MAGIC")


def test_year_and_fuel_lengths_must_agree() -> None:
    with pytest.raises(ValueError, match="length 1 or 3"):
        ef_usa_moves(
            "CO",
            [2010, 2015, 2019],
            reference_year=2019,
            speed=50,
            fuel=["D", "CNG"],
        )


def test_bin_limits_are_the_filtered_set() -> None:
    factors = ef_usa_moves("CO", 2015, reference_year=2019, speed=np.array([10.0, 80.0]))
    assert factors.table["ef"].notna().all()
    assert factors.table["bin_upper_kmh"].tolist() == [12.1, 84.5]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pollutant": ""}, ValueError),
        ({"fuel": ""}, ValueError),
        ({"model_year": True}, TypeError),
        ({"speed": True}, TypeError),
        ({"speed": 0}, ValueError),
        ({"speed": 121}, ValueError),
    ],
)
def test_validation_edges(kwargs: dict, error: type[Exception]) -> None:
    base = {"pollutant": "CO", "reference_year": 2019, "fuel": "D", "model_year": 2015, "speed": 50}
    base.update(kwargs)
    with pytest.raises(error):
        ef_usa_moves(**base)


def test_empty_or_bad_moves_parameters() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_moves([], 2015, reference_year=2019, speed=50)
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_moves("CO", 2015, reference_year=2019, speed=[])
    with pytest.raises(TypeError, match="model_year"):
        ef_usa_moves("CO", True, reference_year=2019, speed=50)
    with pytest.raises(TypeError, match="speed"):
        ef_usa_moves("CO", 2015, reference_year=2019, speed="fast")
