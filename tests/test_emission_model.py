"""emission_model tests."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from gtfs2emis_py import EmissionModelError, EmissionResult, emission_model, load_fleet
from gtfs2emis_py.ef import (
    WearError,
    ef_brazil_cetesb,
    ef_brazil_scaled_euro,
    ef_europe_emep,
    ef_usa_emfac,
    ef_usa_moves,
    emi_europe_emep_wear,
)
from tests._fixtures import requires_ef_table


def _transport(speeds: list[float], dists: list[float], shape: str = "SH1") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "shape_id": [shape] * len(speeds),
            "trip_id": [f"T{index}" for index in range(len(speeds))],
            "stop_sequence": list(range(len(speeds))),
            "speed_kmh": speeds,
            "dist_km": dists,
        },
        geometry=[LineString([(0, 0), (0.001, 0)])] * len(speeds),
        crs="EPSG:4326",
    )


def _transport_table(result):
    """Return the transport table held by a result."""
    return getattr(result.tp_model, "table", result.tp_model)


def _cetesb_fleet(model_year: int = 2015, composition: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type_name_br": ["BUS_URBAN_D"],
            "model_year": [model_year],
            "fuel": ["D"],
            "fleet_composition": [composition],
        }
    )


@requires_ef_table("brazil_cetesb")
def test_cetesb_average_fleet() -> None:
    result = emission_model(
        _transport([20.0, 30.0], [1.0, 2.0]),
        "ef_brazil_cetesb",
        _cetesb_fleet(),
        ["CO", "NOx"],
    )
    factor = ef_brazil_cetesb(["CO", "NOx"], "BUS_URBAN_D", 2015).table.set_index("pollutant")["ef"]
    co = result.emi[result.emi["pollutant"] == "CO"].sort_values("segment_id")["value"]
    nox = result.emi[result.emi["pollutant"] == "NOx"].sort_values("segment_id")["value"]
    assert co.tolist() == pytest.approx([factor["CO"] * 1.0, factor["CO"] * 2.0])
    assert nox.tolist() == pytest.approx([factor["NOx"] * 1.0, factor["NOx"] * 2.0])
    assert set(result.emi["unit"]) == {"g"}
    assert result.metadata["ef_model"] == "ef_brazil_cetesb"
    assert result.units == {"CO": "g", "NOx": "g"}


def test_emep_speed_dependent_factors() -> None:
    fleet = pd.DataFrame(
        {
            "veh_type": ["Ubus Std 15 - 18 t"],
            "euro": ["V"],
            "fuel": ["D"],
            "tech": ["SCR"],
            "fleet_composition": [1.0],
        }
    )
    result = emission_model(
        _transport([10.0, 50.0], [1.0, 1.0]),
        "ef_europe_emep",
        fleet,
        ["CO", "NOx"],
    )
    expected = ef_europe_emep(
        [10.0, 50.0], "Ubus Std 15 - 18 t", "V", ["CO", "NOx"], fuel="D", tech="SCR"
    ).table
    for pollutant in ("CO", "NOx"):
        actual = (
            result.emi[result.emi["pollutant"] == pollutant]
            .sort_values("segment_id")["value"]
            .tolist()
        )
        wanted = (
            expected[expected["pollutant"] == pollutant].sort_values("speed_kmh")["ef"].tolist()
        )
        assert actual == pytest.approx(wanted)


@requires_ef_table("usa_emfac")
def test_emfac_uses_reference_year() -> None:
    result = emission_model(
        _transport([10.0], [1.0]),
        "ef_usa_emfac",
        pd.DataFrame({"model_year": [2015], "fuel": ["D"], "fleet_composition": [1.0]}),
        ["CO"],
        reference_year=2020,
    )
    expected = ef_usa_emfac(["CO"], 2020, "D", model_year=2015, speed=[10.0]).table["ef"].iloc[0]
    assert result.emi["value"].iloc[0] == pytest.approx(expected)


@requires_ef_table("usa_emfac")
def test_per_vehicle_fleet_groups_by_shape() -> None:
    transport = pd.concat(
        [_transport([10.0], [1.0], shape="SH1"), _transport([10.0], [2.0], shape="SH2")],
        ignore_index=True,
    )
    fleet = pd.DataFrame(
        {
            "shape_id": ["SH1", "SH2"],
            "model_year": [2010, 2018],
            "fuel": ["D", "D"],
            "fleet_composition": [1.0, 1.0],
        }
    )
    result = emission_model(transport, "ef_usa_emfac", fleet, ["CO"], reference_year=2020)
    assert len(result.emi) == 2
    assert result.emi["shape_id"].tolist() == ["SH1", "SH2"]
    expected = [
        ef_usa_emfac(["CO"], 2020, "D", model_year=year, speed=[10.0]).table["ef"].iloc[0]
        for year in (2010, 2018)
    ]
    assert result.emi["value"].tolist() == pytest.approx([expected[0], expected[1] * 2.0])


@requires_ef_table("usa_moves")
def test_moves_uses_reference_year() -> None:
    result = emission_model(
        _transport([10.0], [1.0]),
        "ef_usa_moves",
        pd.DataFrame({"model_year": [2015], "fuel": ["D"], "fleet_composition": [1.0]}),
        ["CO"],
        reference_year=2020,
    )
    expected = ef_usa_moves(["CO"], 2015, reference_year=2020, speed=[10.0]).table["ef"].iloc[0]
    assert result.emi["value"].iloc[0] == pytest.approx(expected)


@requires_ef_table("brazil_cetesb")
def test_scaled_composite_uses_local_and_emep_factors() -> None:
    fleet = pd.DataFrame(
        {
            "type_name_br": ["BUS_URBAN_D"],
            "type_name_eu": ["Ubus Std 15 - 18 t"],
            "euro": ["V"],
            "tech": ["SCR"],
            "model_year": [2015],
            "fuel": ["D"],
            "fleet_composition": [1.0],
        }
    )
    result = emission_model(_transport([10.0], [1.0]), "ef_brazil_scaled_euro", fleet, ["CO"])
    expected = (
        ef_brazil_scaled_euro(["CO"], [10.0], "BUS_URBAN_D", 2015, "Ubus Std 15 - 18 t", "V")
        .table["ef"]
        .iloc[0]
    )
    assert result.emi["value"].iloc[0] == pytest.approx(expected)


@requires_ef_table("usa_emfac")
def test_per_vehicle_group_keys_compare_as_text() -> None:
    transport = _transport([10.0], [1.0], shape="1708")
    fleet = pd.DataFrame(
        {
            "shape_id": [1708],
            "model_year": [2015],
            "fuel": ["D"],
            "fleet_composition": [1.0],
        }
    )
    result = emission_model(transport, "ef_usa_emfac", fleet, ["CO"], reference_year=2020)
    assert result.emi["shape_id"].tolist() == ["1708"]


@requires_ef_table("brazil_cetesb")
def test_duplicate_speed_expands_to_each_segment() -> None:
    result = emission_model(
        _transport([10.0, 10.0], [1.0, 3.0]),
        "ef_brazil_cetesb",
        _cetesb_fleet(),
        ["CO"],
    )
    co = result.emi[result.emi["pollutant"] == "CO"].sort_values("segment_id")["value"]
    factor = ef_brazil_cetesb(["CO"], "BUS_URBAN_D", 2015).table["ef"].iloc[0]
    assert co.tolist() == pytest.approx([factor, factor * 3.0])


@requires_ef_table("brazil_cetesb")
def test_zero_distance_segments_are_omitted() -> None:
    result = emission_model(
        _transport([20.0, 0.0], [1.0, 0.0]),
        "ef_brazil_cetesb",
        _cetesb_fleet(),
        ["CO"],
    )
    assert result.emi["segment_id"].tolist() == [0]


@requires_ef_table("brazil_cetesb")
def test_result_helpers_are_copies() -> None:
    result = emission_model(_transport([20.0], [1.0]), "ef_brazil_cetesb", _cetesb_fleet(), ["CO"])
    long = result.to_long()
    long.loc[0, "value"] = -1
    assert result.emi["value"].iloc[0] != -1
    wide = result.to_frame()
    assert wide.iloc[0]["CO_hot_exhaust"] == pytest.approx(result.emi["value"].iloc[0])
    assert result.to_dataframe().equals(wide)
    assert result.to_pandas().equals(wide)
    assert result.emis_to_dt().equals(result.emi)


@requires_ef_table("brazil_cetesb")
def test_ncores_accepts_full_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    result = emission_model(
        _transport([20.0], [1.0]),
        "ef_brazil_cetesb",
        _cetesb_fleet(),
        ["CO"],
        ncores=2,
    )
    assert not result.emi.empty


def test_errors() -> None:
    transport = _transport([20.0], [1.0])
    with pytest.raises(EmissionModelError, match="unknown process"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], process="noise")
    with pytest.raises(EmissionModelError, match="unknown ef_model"):
        emission_model(transport, "nope", _cetesb_fleet(), ["CO"])
    with pytest.raises(EmissionModelError, match="EMEP-only"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], process="tyre")
    with pytest.raises(ValueError, match="resume"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], resume=True)
    with pytest.raises(ValueError, match="speed_kmh"):
        emission_model(
            transport.drop(columns=["speed_kmh"]), "ef_brazil_cetesb", _cetesb_fleet(), ["CO"]
        )
    with pytest.raises(ValueError, match="ncores"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], ncores=999)


@requires_ef_table("brazil_cetesb")
def test_fleet_can_be_a_validated_fleet_table() -> None:
    fleet = load_fleet(_cetesb_fleet(), ef_model="ef_brazil_cetesb")
    result = emission_model(_transport([20.0], [1.0]), "ef_brazil_cetesb", fleet, ["CO"])
    assert not result.emi.empty


def _emep_fleet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "veh_type": ["Ubus Std 15 - 18 t"],
            "euro": ["V"],
            "fuel": ["D"],
            "tech": ["SCR"],
            "fleet_composition": [1.0],
        }
    )


def test_wear_processes_match_the_wear_port() -> None:
    result = emission_model(
        _transport([30.0, 60.0], [1.0, 2.0]),
        "ef_europe_emep",
        _emep_fleet(),
        ["TSP", "PM10"],
        process=("tyre", "brake"),
    )
    assert set(result.emi["process"]) == {"tyre", "brake"}
    assert result.metadata["process"] == ["tyre", "brake"]
    assert set(result.emi["unit"]) == {"g"}
    for process_name in ("tyre", "brake"):
        expected = emi_europe_emep_wear(
            [1.0, 2.0],
            [30.0, 60.0],
            ["TSP", "PM10"],
            ["Ubus Std 15 - 18 t"],
            [1.0],
            process=process_name,
        )
        actual = result.emi[result.emi["process"] == process_name]
        expected = expected.rename(columns={"segment": "segment_id"})
        merged = actual.merge(
            expected,
            on=["segment_id", "pollutant"],
            suffixes=("_actual", "_expected"),
        )
        assert len(merged) == len(actual)
        assert merged["value"].tolist() == pytest.approx(merged["emi_g"].tolist())


def test_road_wear_rejects_pm1() -> None:
    with pytest.raises(WearError, match="road"):
        emission_model(
            _transport([30.0], [1.0]),
            "ef_europe_emep",
            _emep_fleet(),
            ["PM1.0"],
            process="road",
        )


def test_wear_wide_frame_keeps_process_columns() -> None:
    result = emission_model(
        _transport([30.0], [1.0]),
        "ef_europe_emep",
        _emep_fleet(),
        ["TSP"],
        process="tyre",
    )
    wide = result.to_frame()
    assert "TSP_tyre" in wide.columns


@requires_ef_table("brazil_cetesb")
def test_output_path_writes_an_emission_manifest(tmp_path) -> None:
    result = emission_model(
        _transport([20.0], [1.0]),
        "ef_brazil_cetesb",
        _cetesb_fleet(),
        ["CO"],
        output_path=tmp_path,
    )
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "shapes" / "SH1.parquet").exists()
    assert result.metadata["output_path"] == str(tmp_path)


def _emep_fleet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "veh_type": ["Ubus Std 15 - 18 t"],
            "euro": ["V"],
            "fuel": ["D"],
            "tech": ["SCR"],
            "fleet_composition": [1.0],
        }
    )


def test_emep_energy_emits_megajoules() -> None:
    result = emission_model(
        _transport([10.0], [2.0]),
        "ef_europe_emep",
        _emep_fleet(),
        ["EC"],
    )
    expected = (
        ef_europe_emep([10.0], "Ubus Std 15 - 18 t", "V", ["EC"], fuel="D", tech="SCR")
        .table["ef"]
        .iloc[0]
    )
    assert result.emi["value"].iloc[0] == pytest.approx(expected * 2.0)
    assert set(result.emi["unit"]) == {"MJ"}
    assert result.units == {"EC": "MJ"}


@requires_ef_table("usa_moves")
def test_moves_energy_emits_kilojoules() -> None:
    result = emission_model(
        _transport([10.0], [2.0]),
        "ef_usa_moves",
        pd.DataFrame({"model_year": [2015], "fuel": ["D"], "fleet_composition": [1.0]}),
        ["EC"],
        reference_year=2020,
    )
    expected = (
        ef_usa_moves(["EC"], 2015, reference_year=2020, speed=[10.0], fuel="D").table["ef"].iloc[0]
    )
    assert result.emi["value"].iloc[0] == pytest.approx(expected * 2.0)
    assert set(result.emi["unit"]) == {"kJ"}
    assert result.units == {"EC": "kJ"}


@requires_ef_table("brazil_cetesb")
def test_cetesb_energy_normalised_pollutant_is_rejected() -> None:
    with pytest.raises(EmissionModelError, match="energy-consumption conversion"):
        emission_model(
            _transport([10.0], [2.0]),
            "ef_brazil_cetesb",
            _cetesb_fleet(),
            ["gCO2/KWH"],
        )


def test_mixed_hot_exhaust_and_tyre_rows_are_grams() -> None:
    result = emission_model(
        _transport([10.0], [2.0]),
        "ef_europe_emep",
        _emep_fleet(),
        ["PM10"],
        process=("hot_exhaust", "tyre"),
    )
    assert set(result.emi["unit"]) == {"g"}
    assert set(result.units.values()) == {"g"}


def test_emission_model_does_not_accept_quiet() -> None:
    import inspect

    assert "quiet" not in inspect.signature(emission_model).parameters


def test_invalid_parallel_and_context_parameters() -> None:
    transport = _transport([20.0], [1.0])
    with pytest.raises(TypeError, match="parallel"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], parallel="yes")
    with pytest.raises(TypeError, match="reference_year"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), ["CO"], reference_year=True)
    with pytest.raises(EmissionModelError, match="heightfile"):
        emission_model(
            transport,
            "ef_brazil_cetesb",
            _cetesb_fleet(),
            ["CO"],
            heightfile="dem.tif",
        )
    with pytest.raises(TypeError, match="tp_model"):
        emission_model(object(), "ef_brazil_cetesb", _cetesb_fleet(), ["CO"])
    with pytest.raises(ValueError, match="pollutant must not be empty"):
        emission_model(transport, "ef_brazil_cetesb", _cetesb_fleet(), [])


@requires_ef_table("brazil_cetesb")
def test_from_long_rejects_missing_columns() -> None:
    result = emission_model(_transport([20.0], [1.0]), "ef_brazil_cetesb", _cetesb_fleet(), ["CO"])
    with pytest.raises(ValueError, match="missing column"):
        EmissionResult.from_long(result.emi.drop(columns=["unit"]), _transport_table(result))
