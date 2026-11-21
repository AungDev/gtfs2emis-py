"""multiply_ef tests."""

from __future__ import annotations

import pandas as pd
import pytest

from gtfs2emis_py.multiply import MultiplyError, multiply_ef


def _constant_ef() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pollutant": ["CO", "CO", "NOx", "NOx"],
            "vehicle_index": [0, 1, 0, 1],
            "ef": [10.0, 20.0, 1.0, 2.0],
            "unit": ["g/km"] * 4,
        }
    )


def test_hand_computed_single_segment() -> None:
    result = multiply_ef([0.5, 0.5], 2.0, _constant_ef())
    # CO: (0.5*10 + 0.5*20) * 2 km = 30 g
    assert len(result) == 2
    co = result[result["pollutant"] == "CO"]["emi_g"].iloc[0]
    nox = result[result["pollutant"] == "NOx"]["emi_g"].iloc[0]
    assert co == pytest.approx(30.0)
    assert nox == pytest.approx(3.0)


def test_constant_broadcasts_across_segments() -> None:
    result = multiply_ef([1.0, 0.0], [1.0, 2.0, 3.0], _constant_ef())
    co = result[result["pollutant"] == "CO"].sort_values("segment")["emi_g"].tolist()
    assert co == pytest.approx([10.0, 20.0, 30.0])


def test_per_segment_matches_constant_when_constant() -> None:
    constant = multiply_ef([0.5, 0.5], [1.0, 2.0], _constant_ef())
    per_segment = _constant_ef()
    per_segment = pd.concat(
        [per_segment.assign(segment=segment) for segment in (0, 1)], ignore_index=True
    )
    result = multiply_ef([0.5, 0.5], [1.0, 2.0], per_segment)
    assert result["emi_g"].tolist() == pytest.approx(constant["emi_g"].tolist())


def test_aggregate_false_keeps_vehicle_rows() -> None:
    result = multiply_ef([0.5, 0.5], 2.0, _constant_ef(), aggregate=False)
    assert len(result) == 4
    assert list(result.columns) == ["segment", "pollutant", "vehicle_index", "emi_g"]
    co = result[result["pollutant"] == "CO"].sort_values("vehicle_index")["emi_g"]
    assert co.tolist() == pytest.approx([10.0, 20.0])


def test_zero_distance_is_zero_not_missing() -> None:
    result = multiply_ef([1.0, 0.0], 0.0, _constant_ef())
    assert (result["emi_g"] == 0).all()


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="fleet_composition"):
        multiply_ef([1.2], 1.0, _constant_ef())
    with pytest.raises(ValueError, match="dist_km"):
        multiply_ef([1.0], -1.0, _constant_ef())
    with pytest.raises(MultiplyError, match="vehicle_index"):
        multiply_ef([1.0], 1.0, _constant_ef())
    with pytest.raises(MultiplyError, match="missing column"):
        multiply_ef([1.0], 1.0, pd.DataFrame({"pollutant": ["CO"], "ef": [1.0]}))
    with pytest.raises(MultiplyError, match="duplicate"):
        multiply_ef(
            [1.0],
            1.0,
            pd.DataFrame(
                {
                    "pollutant": ["CO", "CO"],
                    "vehicle_index": [0, 0],
                    "ef": [1.0, 2.0],
                    "unit": ["g/km", "g/km"],
                }
            ),
        )
    with pytest.raises(MultiplyError, match="segment"):
        multiply_ef([1.0, 1.0], [1.0], _constant_ef().assign(segment=5))


def test_more_invalid_inputs_are_named() -> None:
    with pytest.raises(ValueError, match="fleet_composition must not be empty"):
        multiply_ef([], 1.0, _constant_ef())
    with pytest.raises(ValueError, match="fleet_composition values"):
        multiply_ef([float("nan")], 1.0, _constant_ef())
    with pytest.raises(ValueError, match="dist_km must not be empty"):
        multiply_ef([1.0], [], _constant_ef())
    with pytest.raises(ValueError, match="dist_km must be finite"):
        multiply_ef([1.0], float("inf"), _constant_ef())
    with pytest.raises(TypeError, match="ef must be"):
        multiply_ef([1.0], 1.0, [{"pollutant": "CO"}])
    with pytest.raises(MultiplyError, match="ef is empty"):
        multiply_ef(
            [1.0],
            1.0,
            pd.DataFrame({"pollutant": [], "vehicle_index": [], "ef": [], "unit": []}),
        )
    with pytest.raises(MultiplyError, match="finite"):
        multiply_ef([1.0], 1.0, _constant_ef().assign(ef=[1.0, 1.0, float("nan"), 1.0]))
    with pytest.raises(MultiplyError, match="non-negative"):
        multiply_ef([1.0], 1.0, _constant_ef().assign(ef=[1.0, 1.0, -1.0, 1.0]))
    with pytest.raises(MultiplyError, match="vehicle_index must be integers"):
        multiply_ef([1.0], 1.0, _constant_ef().assign(vehicle_index=[0.5, 0.5, 0, 0]))
    with pytest.raises(MultiplyError, match="segment must be integers"):
        multiply_ef([1.0, 1.0], [1.0], _constant_ef().assign(segment=0.5))
    duplicated = pd.concat([_constant_ef().assign(segment=0)] * 2, ignore_index=True)
    with pytest.raises(MultiplyError, match="duplicate"):
        multiply_ef([0.5, 0.5], [1.0], duplicated)
    incomplete = _constant_ef().assign(segment=0)
    with pytest.raises(MultiplyError, match="one row per segment"):
        multiply_ef([0.5, 0.5], [1.0, 2.0], incomplete)


def test_non_mass_per_distance_factor_is_rejected() -> None:
    for unit in ["g/kWh", "#/kWh", "MJ/km", "kJ/km", "g"]:
        frame = _constant_ef().assign(unit=unit)
        with pytest.raises(MultiplyError, match="unsupported emission-factor unit"):
            multiply_ef([1.0, 0.0], 1.0, frame)


def test_mass_per_distance_units_are_converted_to_grams() -> None:
    frame = _constant_ef().iloc[[0]].copy()
    frame["unit"] = "kg/km"
    result = multiply_ef([1.0], 2.0, frame)
    assert result["emi_g"].iloc[0] == pytest.approx(20_000.0)
    frame["unit"] = "g/m"
    assert multiply_ef([1.0], 2.0, frame)["emi_g"].iloc[0] == pytest.approx(20_000.0)
