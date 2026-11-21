"""EMEP/EEA EF lookup tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gtfs2emis_py.ef import (
    EmepFallbackWarning,
    MissingEmissionFactorError,
    ef_europe_emep,
    load_ef_table,
)
from gtfs2emis_py.ef import europe_emep as emep

SEGMENT = "Ubus Std 15 - 18 t"

# One synthetic row per family, evaluated at speed 10 with
# Alpha=1, Beta=2, Gamma=3, Delta=4, Epsilon=5, Zita=6, Hta=7, RF=0.1,
# k=2 and fcorr=1.5. Values were computed independently from the R switch.
GOLDEN = {
    "1": 0.5876190476190477,
    "2019_1": 0.5876190476190477,
    "2013_3": 3072000.0,
    "2007_2": 3072000.0,
    "2013_12": 2.9999999999994387,
    "2013_9": 0.0014992503748125937,
    "2013_11": 3.0,
    "2013_4": 90300.0,
    "2007_3": 90300.0,
    "2013_16": 9960.350768209652,
    "2007_11": 9960.350768209652,
    "2013_10": 0.14285714285714285,
    "2013_1": 3702.0,
    "2007_7": 3702.0,
    "2007_1": 15.000000006183463,
    "2007_4": 3.0,
    "2007_5": 63.75,
    "2007_6": 1455495595.2293708,
    "2007_9": 0.009345794392523364,
    "2007_10": 2.9999999999994387,
    "2007_8": 9.00000000618346,
}


def _synthetic_rows() -> pd.DataFrame:
    records = []
    for family in GOLDEN:
        records.append(
            {
                "Alpha": 1.0,
                "Beta": 2.0,
                "Gamma": 3.0,
                "Delta": 4.0,
                "Epsilon": 5.0,
                "Zita": 6.0,
                "Hta": 7.0,
                "RF": 0.1,
                "k": 2.0,
                "Function_ID": family,
            }
        )
    return pd.DataFrame(records)


def test_every_table_family_has_a_formula() -> None:
    table = load_ef_table("europe_emep")
    assert set(table["Function_ID"].astype(str)) <= set(emep._FORMULAS)


@pytest.mark.parametrize("family", sorted(GOLDEN))
def test_function_family_matches_reference_value(family: str) -> None:
    rows = _synthetic_rows()
    rows = rows[rows["Function_ID"] == family]
    value = emep._evaluate(rows, np.array([10.0]), 1.5)[0]
    assert value == pytest.approx(GOLDEN[family], rel=1e-12)


def test_no_x_euro_iv_matches_the_reference_formula() -> None:
    table = load_ef_table("europe_emep")
    row = table[
        (table["Pol"] == "NOx")
        & (table["Segment"] == SEGMENT)
        & (table["Fuel"] == "D")
        & (table["Euro"] == "IV")
        & (table["Technology"] == "SCR")
        & (table["Slope"] == 0.0)
        & (table["Load"] == 0.5)
    ].iloc[0]
    speed = 50.0
    expected = (
        (row.Alpha * speed**2 + row.Beta * speed + row.Gamma + row.Delta / speed)
        / (row.Epsilon * speed**2 + row.Zita * speed + row.Hta)
        * (1 - row.RF)
        * row.k
    )
    factors = ef_europe_emep(speed, SEGMENT, "IV", "NOx", tech="SCR")
    assert factors.table.loc[0, "ef"] == pytest.approx(expected)
    assert factors.table.loc[0, "unit"] == "g/km"


def test_speed_is_clamped_to_each_rows_range() -> None:
    factors = ef_europe_emep([86, 130], SEGMENT, "IV", "NOx", tech="SCR")
    assert factors.table.loc[0, "ef"] == pytest.approx(factors.table.loc[1, "ef"])


def test_slope_and_load_change_the_factor() -> None:
    downhill = ef_europe_emep(50, SEGMENT, "IV", "NOx", tech="SCR", slope=-0.06)
    uphill = ef_europe_emep(50, SEGMENT, "IV", "NOx", tech="SCR", slope=0.06)
    empty = ef_europe_emep(50, SEGMENT, "IV", "NOx", tech="SCR", load=0.0)
    full = ef_europe_emep(50, SEGMENT, "IV", "NOx", tech="SCR", load=1.0)
    assert downhill.table.loc[0, "ef"] != uphill.table.loc[0, "ef"]
    assert empty.table.loc[0, "ef"] != full.table.loc[0, "ef"]


def test_single_row_slope_fallback_warns() -> None:
    with pytest.warns(EmepFallbackWarning, match="tabulated only at slope"):
        factors = ef_europe_emep(
            50, SEGMENT, "IV", "CH4", tech="SCR", slope=0.02, on_missing_slope="nearest"
        )
    assert len(factors) == 1
    assert math.isfinite(factors.table.loc[0, "ef"])
    with pytest.raises(MissingEmissionFactorError, match="on_missing_slope='nearest'"):
        ef_europe_emep(50, SEGMENT, "IV", "CH4", tech="SCR", slope=0.02)


def test_euro_vi_forces_dpf_scr() -> None:
    with pytest.warns(EmepFallbackWarning, match="DPF\\+SCR"):
        forced = ef_europe_emep(50, SEGMENT, "VI", "NOx", tech="EGR")
    explicit = ef_europe_emep(50, SEGMENT, "VI", "NOx", tech="DPF+SCR")
    assert forced.table.loc[0, "ef"] == pytest.approx(explicit.table.loc[0, "ef"])


def test_euro_iii_drops_the_technology() -> None:
    with pytest.warns(EmepFallbackWarning, match="uses '-'"):
        forced = ef_europe_emep(50, SEGMENT, "III", "NOx", tech="SCR")
    explicit = ef_europe_emep(50, SEGMENT, "III", "NOx", tech="-")
    assert forced.table.loc[0, "ef"] == pytest.approx(explicit.table.loc[0, "ef"])


def test_fc_on_euro_v_asserts_scr() -> None:
    with pytest.warns(EmepFallbackWarning, match="'SCR'"):
        factors = ef_europe_emep(50, SEGMENT, "V", "CO2", tech="-")
    explicit = ef_europe_emep(50, SEGMENT, "V", "CO2", tech="SCR")
    assert factors.table.loc[0, "ef"] == pytest.approx(explicit.table.loc[0, "ef"])


def test_missing_combination_names_the_keys() -> None:
    with (
        pytest.warns(EmepFallbackWarning),
        pytest.raises(MissingEmissionFactorError, match="CO2|no EMEP emission factor"),
    ):
        ef_europe_emep(50, SEGMENT, "Conventional", "CO2", fuel="BD")


def test_ambiguous_edition_rows_abort() -> None:
    with pytest.raises(MissingEmissionFactorError, match="ambiguous"):
        ef_europe_emep(50, SEGMENT, "V", "FC", tech="SCR")


def test_ec_is_returned_in_mj_per_km() -> None:
    factors = ef_europe_emep(50, SEGMENT, "IV", "EC", tech="SCR")
    assert factors.table.loc[0, "unit"] == "MJ/km"


def test_fc_is_g_per_km() -> None:
    factors = ef_europe_emep(50, SEGMENT, "VI", "FC", tech="DPF+SCR")
    assert factors.table.loc[0, "unit"] == "g/km"


def test_output_is_the_cartesian_product() -> None:
    factors = ef_europe_emep(
        [10, 50, 90],
        [SEGMENT, "Ubus Artic >18 t"],
        "IV",
        ["NOx", "PM10"],
        tech="SCR",
    )
    assert len(factors) == 3 * 2 * 2
    assert set(factors.table["pollutant"]) == {"NOx", "PM10"}
    assert set(factors.table["speed_kmh"]) == {10.0, 50.0, 90.0}
    assert factors.table["vehicle_index"].max() == 1


def test_slope_per_speed() -> None:
    factors = ef_europe_emep(
        [10, 50],
        SEGMENT,
        "IV",
        "NOx",
        tech="SCR",
        slope=[-0.02, 0.02],
    )
    assert factors.table.loc[0, "slope"] == pytest.approx(-0.02)
    assert factors.table.loc[1, "slope"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"speed": 0}, ValueError),
        ({"speed": 131}, ValueError),
        ({"slope": 0.03}, ValueError),
        ({"load": 0.25}, ValueError),
        ({"fcorr": -1}, ValueError),
        ({"tech": "MAGIC"}, ValueError),
        ({"euro": "VII"}, ValueError),
        ({"pollutant": "SPN23 [#/kWh]"}, ValueError),
    ],
)
def test_invalid_inputs(kwargs: dict, error: type[Exception]) -> None:
    base = {"speed": 50, "veh_type": SEGMENT, "euro": "IV", "pollutant": "NOx", "tech": "SCR"}
    base.update(kwargs)
    with pytest.raises(error):
        ef_europe_emep(**base)


def test_vehicle_fields_share_one_length() -> None:
    with pytest.raises(ValueError, match="length 1 or 3"):
        ef_europe_emep(
            50,
            [SEGMENT, "Ubus Artic >18 t"],
            ["IV", "V", "VI"],
            "NOx",
            tech="SCR",
        )


def test_slope_must_match_speed_length() -> None:
    with pytest.raises(ValueError, match="slope and speed"):
        ef_europe_emep([10, 50, 90], SEGMENT, "IV", "NOx", tech="SCR", slope=[0.0, 0.02])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pollutant": []},
        {"speed": []},
        {"veh_type": []},
        {"euro": []},
        {"tech": []},
        {"fuel": []},
        {"speed": ["fast"]},
        {"speed": [float("nan")]},
        {"fcorr": -1},
        {"fcorr": [1, 2, 3]},
        {"slope": 0.5},
        {"load": 0.7},
        {"on_missing_slope": "closest"},
    ],
)
def test_invalid_vector_parameters_are_rejected(kwargs: dict) -> None:
    base = {
        "speed": 50,
        "veh_type": "Ubus Std 15 - 18 t",
        "euro": "V",
        "pollutant": "CO",
    }
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ef_europe_emep(**base)
