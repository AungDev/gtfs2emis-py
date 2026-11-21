"""Fleet schema and validation tests.

The three shipped fleet tables exercise both modes: Detroit (average, EMFAC),
Dublin (average, EMEP) and Curitiba (per-shape counts, CETESB). The invalid
cases are built in memory so the table of failures stays next to its assertions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gtfs2emis_py.fleet import (
    FleetError,
    FleetTable,
    UnknownFleetModelError,
    load_fleet,
)

DATA = Path(__file__).resolve().parent / "data"


def _average(**overrides: object) -> pd.DataFrame:
    row = {
        "veh_type": "Ubus Std 15 - 18 t",
        "euro": "V",
        "fuel": "D",
        "tech": "SCR",
        "fleet_composition": 1.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_detroit_average_mode() -> None:
    fleet = load_fleet(DATA / "usa_det_fleet.txt", ef_model="ef_usa_emfac")
    assert isinstance(fleet, FleetTable)
    assert fleet.mode == "average"
    assert fleet.group_columns == ()
    assert fleet.n_rows == 9
    assert fleet.table["fleet_composition"].sum() == pytest.approx(1.0)
    assert "model_year" in fleet.table.columns
    assert "count" in fleet.table.columns


def test_dublin_average_mode() -> None:
    fleet = load_fleet(DATA / "irl_dub_fleet.txt", ef_model="ef_europe_emep")
    assert fleet.mode == "average"
    assert fleet.table["fleet_composition"].sum() == pytest.approx(1.0)
    assert set(fleet.table["euro"]) <= {"III", "IV", "V", "VI"}


def test_curitiba_per_vehicle_mode_derives_composition_per_shape() -> None:
    fleet = load_fleet(DATA / "bra_cur_fleet.txt", ef_model="ef_brazil_cetesb")
    assert fleet.mode == "per_vehicle"
    assert fleet.group_columns == ("shape_id",)
    sums = fleet.table.groupby("shape_id")["fleet_composition"].sum()
    assert not sums.empty
    assert (sums - 1.0).abs().max() == pytest.approx(0.0, abs=1e-9)
    assert (fleet.table["fleet_composition"] > 0).all()


def test_mode_can_be_asserted() -> None:
    assert load_fleet(_average(), ef_model="ef_europe_emep", mode="average").mode == "average"


def test_per_vehicle_duplicate_counts_are_summed() -> None:
    source = pd.DataFrame(
        {
            "shape_id": ["S1", "S1"],
            "year": [2010, 2010],
            "type_name_br": ["BUS_URBAN_D", "BUS_URBAN_D"],
            "fuel": ["D", "D"],
            "N": [6, 2],
        }
    )
    fleet = load_fleet(source, ef_model="ef_brazil_cetesb")
    assert fleet.n_rows == 1
    assert fleet.table.loc[0, "count"] == 8
    assert fleet.table.loc[0, "fleet_composition"] == pytest.approx(1.0)


def test_dataframe_input_is_copied() -> None:
    source = _average()
    fleet = load_fleet(source, ef_model="ef_europe_emep")
    fleet.table.loc[0, "fleet_composition"] = 0.5
    assert source.loc[0, "fleet_composition"] == 1.0


def test_aliases_normalise_year_and_counts() -> None:
    source = pd.DataFrame(
        {
            "year": [2010, 2011],
            "N": [3, 1],
            "fuel": ["D", "D"],
        }
    )
    fleet = load_fleet(source, ef_model="ef_usa_moves")
    assert list(fleet.table.columns[:1]) == ["model_year"]
    assert fleet.table["fleet_composition"].tolist() == [0.75, 0.25]


def test_unknown_model_lists_the_specs() -> None:
    with pytest.raises(UnknownFleetModelError) as exc_info:
        load_fleet(_average(), ef_model="ef_not_a_model")
    message = str(exc_info.value)
    assert "ef_brazil_cetesb" in message and "ef_europe_emep" in message


def test_missing_required_column_names_it() -> None:
    source = _average()
    source = source.drop(columns=["tech"])
    with pytest.raises(FleetError, match="tech"):
        load_fleet(source, ef_model="ef_europe_emep")


def test_unknown_category_names_value_and_allowed_set() -> None:
    with pytest.raises(FleetError) as exc_info:
        load_fleet(_average(euro="VII"), ef_model="ef_europe_emep")
    message = str(exc_info.value)
    assert "VII" in message
    assert "Conventional" in message


def test_null_category_is_rejected() -> None:
    with pytest.raises(FleetError, match="null"):
        load_fleet(_average(tech=None), ef_model="ef_europe_emep")


def test_composition_out_of_range_is_rejected() -> None:
    with pytest.raises(FleetError, match=r"\[0, 1\]"):
        load_fleet(_average(fleet_composition=1.5), ef_model="ef_europe_emep")


def test_composition_must_sum_to_one() -> None:
    source = pd.concat(
        [
            _average(fleet_composition=0.4, euro="V"),
            _average(fleet_composition=0.4, euro="IV"),
        ]
    )
    with pytest.raises(FleetError, match="sum to 1"):
        load_fleet(source, ef_model="ef_europe_emep")


def test_duplicate_keys_are_rejected() -> None:
    source = pd.concat(
        [_average(fleet_composition=0.5), _average(fleet_composition=0.5)],
        ignore_index=True,
    )
    with pytest.raises(FleetError, match="duplicate"):
        load_fleet(source, ef_model="ef_europe_emep")


def test_non_positive_counts_are_rejected() -> None:
    source = pd.DataFrame({"year": [2010, 2011], "N": [1, 0], "fuel": ["D", "D"]})
    with pytest.raises(FleetError, match="positive"):
        load_fleet(source, ef_model="ef_usa_emfac")


def test_alias_and_canonical_column_together_are_rejected() -> None:
    source = pd.DataFrame(
        {"year": [2010], "model_year": [2010], "fuel": ["D"], "fleet_composition": [1.0]}
    )
    with pytest.raises(FleetError, match="alias"):
        load_fleet(source, ef_model="ef_usa_emfac")


def test_mode_contradiction_is_rejected() -> None:
    source = _average(shape_id="SH1")
    with pytest.raises(FleetError, match="average"):
        load_fleet(source, ef_model="ef_europe_emep", mode="average")


def test_two_assignment_columns_are_rejected() -> None:
    source = _average(shape_id="SH1", route_id="R1")
    with pytest.raises(FleetError, match="assignment column"):
        load_fleet(source, ef_model="ef_europe_emep")


def test_fractional_model_year_is_rejected() -> None:
    source = pd.DataFrame({"year": [2015.5], "N": [1], "fuel": ["D"]})
    with pytest.raises(FleetError, match="fractional years"):
        load_fleet(source, ef_model="ef_usa_emfac")


def test_year_out_of_range_is_rejected() -> None:
    source = pd.DataFrame({"year": [1800], "N": [1], "fuel": ["D"]})
    with pytest.raises(FleetError, match="1900"):
        load_fleet(source, ef_model="ef_usa_moves")


def test_non_path_or_dataframe_is_rejected() -> None:
    with pytest.raises(TypeError, match="path or DataFrame"):
        load_fleet(42, ef_model="ef_usa_moves")  # type: ignore[arg-type]


def test_emep_coach_segments_are_accepted() -> None:
    for segment in ("Coaches Std <=18 t", "Coaches Artic >18 t"):
        fleet = load_fleet(_average(veh_type=segment), ef_model="ef_europe_emep")
        assert fleet.n_rows == 1


@pytest.mark.parametrize(("alias", "canonical"), [("Diesel", "D"), ("Gasoline", "G")])
def test_us_fuel_aliases_are_normalised(alias: str, canonical: str) -> None:
    source = pd.DataFrame({"model_year": [2015], "fuel": [alias], "fleet_composition": [1.0]})
    fleet = load_fleet(source, ef_model="ef_usa_emfac")
    assert fleet.table["fuel"].tolist() == [canonical]
