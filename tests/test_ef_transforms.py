"""EF transform tests."""

from __future__ import annotations

import numpy as np
import pytest

from gtfs2emis_py.ef import ef_brazil_cetesb, ef_brazil_scaled_euro, ef_europe_emep, ef_scaled_euro
from gtfs2emis_py.ef.transforms import fuel_correction
from tests._fixtures import requires_ef_table

UNITS = {"BUS_URBAN_D": "Ubus Std 15 - 18 t"}


def _fuel_factor(pollutant: str, fuel: dict[str, float]) -> float:
    den, s, pah, cn, t95 = fuel["den"], fuel["s"], fuel["pah"], fuel["cn"], fuel["t95"]
    if pollutant == "CO":
        return 2.24407 - 0.0011 * den + 0.00007 * pah - 0.00768 * cn - 0.00087 * t95
    if pollutant == "NOx":
        return -1.75444 + 0.00906 * den - 0.0163 * pah + 0.00493 * cn + 0.00266 * t95
    if pollutant == "PM10":
        return (0.06959 + 0.00006 * den + 0.00065 * pah - 0.00001 * cn) * (
            1 - 0.0086 * (450 - s) / 100
        )
    raise ValueError(pollutant)


def test_fuel_correction_matches_the_formula() -> None:
    improved = {"den": 820.0, "s": 10.0, "pah": 1.0, "cn": 55.0, "t95": 300.0}
    frame = fuel_correction(["CO", "NOx", "PM10"], ["PRE", "V", "VI"], improved)
    base_1996 = {"den": 840.0, "s": 400.0, "pah": 9.0, "cn": 51.0, "t95": 350.0}
    base_2005 = improved  # the improved fuel is the 2005 row; base === improved for V/VI
    for _, row in frame.iterrows():
        base = base_1996 if row["euro_stage"] == "PRE" else base_2005
        expected = _fuel_factor(row["pollutant"], improved) / _fuel_factor(row["pollutant"], base)
        assert row["fcorr"] == pytest.approx(expected)


def test_fuel_correction_defaults_are_the_reference_diesel() -> None:
    frame = fuel_correction("CO", ["I", "IV"])
    assert len(frame) == 2
    # I uses the 1996 base, IV the 2005 base; both ratios differ from 1.
    assert frame["fcorr"].notna().all()


def test_fuel_correction_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="pollutant"):
        fuel_correction("SO2", "IV")
    with pytest.raises(ValueError, match="euro_stage"):
        fuel_correction("CO", "VII")
    with pytest.raises(ValueError, match="keys"):
        fuel_correction("CO", "IV", {"den": 835})


def test_scaled_euro_is_local_times_ratio() -> None:
    speeds = [20, 50]
    pollutants = ["CO", "NOx"]
    vehicles = ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"]
    local = [10.0, 20.0, 30.0, 40.0]  # pollutant-major, vehicle-minor
    result = ef_scaled_euro(
        local,
        speeds,
        vehicles,
        ["IV", "V"],
        pollutants,
        tech=["SCR", "SCR"],
    )
    sdc = ef_europe_emep(
        19, vehicles, ["IV", "V"], pollutants, tech=["SCR", "SCR"], slope=0.0, load=0.5
    )
    speed = ef_europe_emep(speeds, vehicles, ["IV", "V"], pollutants, tech=["SCR", "SCR"])
    ratio = sdc.table["ef"].to_numpy() / 1.0
    factors = np.asarray(local) / ratio
    expected = []
    for index in range(len(pollutants)):
        block = factors[index * len(vehicles) : (index + 1) * len(vehicles)]
        expected.extend(np.tile(block, len(speeds)))
    assert result.table["ef"].to_numpy() == pytest.approx(
        speed.table["ef"].to_numpy() * np.asarray(expected)
    )
    assert set(result.table["unit"]) == {"g/km"}


def test_scaled_euro_rejects_ec_and_bad_lengths() -> None:
    with pytest.raises(ValueError, match="EC"):
        ef_scaled_euro([1.0], 50, "Ubus Std 15 - 18 t", "IV", ["EC"])
    with pytest.raises(ValueError, match="values"):
        ef_scaled_euro([1.0], 50, "Ubus Std 15 - 18 t", "IV", ["CO", "NOx"])


@requires_ef_table("brazil_cetesb")
def test_brazil_composite_scales_cetesb_factors() -> None:
    result = ef_brazil_scaled_euro(
        ["CO", "PM10"],
        50,
        ["BUS_URBAN_D", "BUS_MICRO_D"],
        [2015, 2015],
        ["Ubus Std 15 - 18 t", "Ubus Midi <=15 t"],
        ["IV", "IV"],
        tech=["SCR", "SCR"],
    )
    assert len(result) == 2 * 2
    assert result.ef_model == "ef_brazil_scaled_euro"
    assert set(result.table["unit"]) == {"g/km"}
    assert (result.table["ef"] >= 0).all()


def test_composite_requires_matching_vehicle_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ef_brazil_scaled_euro(
            "CO",
            50,
            ["BUS_URBAN_D"],
            2015,
            ["Ubus Std 15 - 18 t", "Ubus Midi <=15 t"],
            ["IV", "IV"],
        )


@pytest.mark.parametrize("slope", [-0.02, 0.0, 0.02])
@pytest.mark.parametrize("load", [0.0, 0.5, 1.0])
def test_scaled_euro_reference_uses_the_requested_slope_and_load(slope: float, load: float) -> None:
    speeds = [10, 50]
    pollutants = ["CO", "NOx"]
    vehicles = ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"]
    euros = ["IV", "V"]
    local = [10.0, 20.0, 30.0, 40.0]
    result = ef_scaled_euro(
        local,
        speeds,
        vehicles,
        euros,
        pollutants,
        tech=["SCR", "SCR"],
        slope=slope,
        load=load,
    )
    reference = ef_europe_emep(
        19,
        vehicles,
        euros,
        pollutants,
        tech=["SCR", "SCR"],
        slope=slope,
        load=load,
    )
    at_speed = ef_europe_emep(
        speeds,
        vehicles,
        euros,
        pollutants,
        tech=["SCR", "SCR"],
        slope=slope,
        load=load,
    )
    ratio = reference.table["ef"].to_numpy(dtype=float)
    factor: list[float] = []
    for index in range(len(pollutants)):
        block = (
            np.asarray(local[index * len(vehicles) : (index + 1) * len(vehicles)])
            / (ratio[index * len(vehicles) : (index + 1) * len(vehicles)])
        )
        factor.extend(np.tile(block, len(speeds)))
    expected = at_speed.table["ef"].to_numpy(dtype=float) * np.asarray(factor)
    assert result.table["ef"].to_numpy(dtype=float) == pytest.approx(expected)


@requires_ef_table("brazil_cetesb")
def test_brazil_composite_passes_nonzero_slope_to_the_reference() -> None:
    result = ef_brazil_scaled_euro(
        ["CO"],
        50,
        ["BUS_URBAN_D"],
        2015,
        ["Ubus Std 15 - 18 t"],
        ["IV"],
        tech=["SCR"],
        slope=0.02,
    )
    default = ef_brazil_scaled_euro(
        ["CO"],
        50,
        ["BUS_URBAN_D"],
        2015,
        ["Ubus Std 15 - 18 t"],
        ["IV"],
        tech=["SCR"],
    )
    assert float(result.table["ef"].iloc[0]) != pytest.approx(float(default.table["ef"].iloc[0]))


@requires_ef_table("brazil_cetesb")
def test_brazil_scaled_euro_matches_r_at_nonzero_slope() -> None:
    # R 4.5.3, gtfs2emis 0.1.2:
    # ef_brazil_cetesb("CO", "BUS_URBAN_D", 2015, as_list = FALSE) |>
    #   ef_scaled_euro(c(10, 50) km/h, ..., slope = 0.02, load = 0.5)
    result = ef_brazil_scaled_euro(
        "CO",
        [10, 50],
        ["BUS_URBAN_D"],
        2015,
        ["Ubus Std 15 - 18 t"],
        ["IV"],
        tech=["SCR"],
        slope=0.02,
        load=0.5,
    )
    assert result.table["ef"].to_numpy(dtype=float) == pytest.approx(
        [0.771689394297314, 0.282044248225087], abs=1e-6
    )


@requires_ef_table("brazil_cetesb")
def test_scaled_euro_accepts_a_dataframe_factor_table() -> None:
    local = ef_brazil_cetesb(["CO"], "BUS_URBAN_D", 2015)
    from_table = ef_scaled_euro(
        local.table,
        50,
        "Ubus Std 15 - 18 t",
        "V",
        ["CO"],
    )
    from_object = ef_scaled_euro(
        local,
        50,
        "Ubus Std 15 - 18 t",
        "V",
        ["CO"],
    )
    assert from_table.table.equals(from_object.table)


@requires_ef_table("brazil_cetesb")
def test_more_transform_validation() -> None:
    with pytest.raises(ValueError, match="unsupported fuel-correction"):
        fuel_correction("SO2", "IV")
    with pytest.raises(ValueError, match="unsupported euro_stage"):
        fuel_correction("CO", "VII")
    with pytest.raises(TypeError, match="numeric"):
        fuel_correction("CO", "IV", {"den": "x", "s": 1, "pah": 1, "cn": 1, "t95": 1})
    with pytest.raises(ValueError, match="finite"):
        fuel_correction("CO", "IV", {"den": float("nan"), "s": 1, "pah": 1, "cn": 1, "t95": 1})
    local = ef_brazil_cetesb(["CO"], "BUS_URBAN_D", 2015)
    with pytest.raises(ValueError, match="must have an"):
        ef_scaled_euro(local.table.drop(columns=["ef"]), 50, "Ubus Std 15 - 18 t", "V", ["CO"])
    with pytest.raises(ValueError, match="rows"):
        ef_scaled_euro(local.table.iloc[:0], 50, "Ubus Std 15 - 18 t", "V", ["CO"])
    with pytest.raises(ValueError, match="SDC"):
        ef_scaled_euro(local, 50, "Ubus Std 15 - 18 t", "V", ["CO"], SDC=0)
    with pytest.raises(ValueError, match="same length"):
        ef_scaled_euro(local, 50, "Ubus Std 15 - 18 t", ["V", "IV"], ["CO"])
