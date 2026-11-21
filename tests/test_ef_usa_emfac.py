"""EMFAC2017 EF lookup tests."""

from __future__ import annotations

import numpy as np
import pytest

from gtfs2emis_py.ef import (
    MissingEmissionFactorError,
    ef_usa_emfac,
    load_ef_table,
)
from tests._fixtures import packaged_ef_table

pytestmark = pytest.mark.skipif(
    not packaged_ef_table("usa_emfac"),
    reason="usa_emfac is fetch-only, so this tree has no table to read",
)


def test_single_value_matches_the_table() -> None:
    table = load_ef_table("usa_emfac")
    expected = table[
        (table["pollutant"] == "CO")
        & (table["model_year"] == 2015)
        & (table["fuel"] == "D")
        & (table["reference_year"] == 2019)
        & (table["upper_speed_interval"] == 56.0)
    ]["ef"].iloc[0]
    factors = ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=50)
    assert len(factors) == 1
    assert factors.table.loc[0, "ef"] == pytest.approx(expected)
    assert factors.table.loc[0, "bin_lower_kmh"] == 49.0
    assert factors.table.loc[0, "bin_upper_kmh"] == 56.0
    assert factors.table.loc[0, "unit"] == "g/km"


def test_bin_boundaries_are_right_closed() -> None:
    factors = ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=[8, 8.1, 16, 16.1])
    assert factors.table["bin_upper_kmh"].tolist() == [8.0, 16.0, 16.0, 24.0]


def test_no_interpolation_within_a_bin() -> None:
    factors = ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=[50, 55])
    assert factors.table.loc[0, "ef"] == pytest.approx(factors.table.loc[1, "ef"])


def test_top_bin_and_its_limit() -> None:
    factors = ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=113)
    assert factors.table.loc[0, "bin_lower_kmh"] == 106.0
    assert factors.table.loc[0, "bin_upper_kmh"] == 113.0
    with pytest.raises(ValueError, match="113"):
        ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=114)


def test_grid_is_pollutant_x_vehicle_x_speed() -> None:
    factors = ef_usa_emfac(
        ["CO", "PM10"],
        2019,
        "D",
        model_year=[2010, 2015],
        speed=[10, 50],
    )
    assert len(factors) == 2 * 2 * 2
    assert set(factors.table["pollutant"]) == {"CO", "PM10"}
    assert set(factors.table["model_year"]) == {2010, 2015}


def test_absent_year_inside_the_documented_range_is_an_error() -> None:
    with pytest.raises(MissingEmissionFactorError, match="1982|no EMFAC factor"):
        ef_usa_emfac("CO", 2019, "D", model_year=1982, speed=50)


def test_absent_reference_year_is_an_error() -> None:
    with pytest.raises(MissingEmissionFactorError, match="2018"):
        ef_usa_emfac("CO", 2018, "D", model_year=2015, speed=50)


def test_unknown_pollutant_and_fuel() -> None:
    with pytest.raises(ValueError, match="pollutant"):
        ef_usa_emfac("MAGIC", 2019, "D", model_year=2015, speed=50)
    with pytest.raises(ValueError, match="fuel"):
        ef_usa_emfac("CO", 2019, "MAGIC", model_year=2015, speed=50)


def test_year_and_fuel_lengths_must_agree() -> None:
    with pytest.raises(ValueError, match="length 1 or 3"):
        ef_usa_emfac(
            "CO",
            2019,
            ["D", "CNG"],
            model_year=[2010, 2015, 2019],
            speed=50,
        )


def test_reference_year_is_scalar_and_validated() -> None:
    with pytest.raises(ValueError, match="2010-2020"):
        ef_usa_emfac("CO", 2009, "D", model_year=2015, speed=50)
    with pytest.raises(TypeError, match="reference_year"):
        ef_usa_emfac("CO", 2019.0, "D", model_year=2015, speed=50)  # type: ignore[arg-type]


def test_bin_limits_are_the_filtered_set() -> None:
    factors = ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=np.array([10.0, 80.0]))
    assert factors.table["ef"].notna().all()
    assert factors.table["bin_upper_kmh"].tolist() == [16.0, 80.0]


def test_rog_units() -> None:
    factors = ef_usa_emfac("ROG", 2019, "D", model_year=2015, speed=50)
    assert factors.table.loc[0, "unit"] == "g/km"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pollutant": ""}, ValueError),
        ({"fuel": ""}, ValueError),
        ({"model_year": True}, TypeError),
        ({"speed": True}, TypeError),
        ({"speed": 0}, ValueError),
        ({"speed": 114}, ValueError),
    ],
)
def test_validation_edges(kwargs: dict, error: type[Exception]) -> None:
    base = {"pollutant": "CO", "reference_year": 2019, "fuel": "D", "model_year": 2015, "speed": 50}
    base.update(kwargs)
    with pytest.raises(error):
        ef_usa_emfac(**base)


def test_empty_or_bad_emfac_parameters() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_emfac([], 2019, "D", model_year=2015, speed=50)
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_emfac("CO", 2019, [], model_year=2015, speed=50)
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_emfac("CO", 2019, "D", model_year=[], speed=50)
    with pytest.raises(ValueError, match="must not be empty"):
        ef_usa_emfac("CO", 2019, "D", model_year=2015, speed=[])
    with pytest.raises(TypeError, match="model_year"):
        ef_usa_emfac("CO", 2019, "D", model_year=True, speed=50)
    with pytest.raises(TypeError, match="speed"):
        ef_usa_emfac("CO", 2019, "D", model_year=2015, speed="fast")
