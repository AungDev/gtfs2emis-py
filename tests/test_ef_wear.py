"""Wear emission tests."""

from __future__ import annotations

import numpy as np
import pytest

from gtfs2emis_py.ef.wear import WearError, emi_europe_emep_wear


def _value(frame, pollutant: str, process: str, vehicle: int = 0) -> float:
    row = frame[
        (frame["pollutant"] == pollutant)
        & (frame["process"] == process)
        & (frame["vehicle_index"] == vehicle)
    ]
    return float(row["emi_g"].iloc[0])


def test_tyre_known_value() -> None:
    frame = emi_europe_emep_wear(
        1.0, 30.0, ["TSP", "PM10"], ["Ubus Std 15 - 18 t"], [1.0], load=0.5, process="tyre"
    )
    # lcf = 1.41 + 1.38*0.5 = 2.10; EF = 0.5*2*2.10*0.0107 = 0.02247
    # sc(30) = 1.39
    assert _value(frame, "TSP", "tyre") == pytest.approx(1.39 * 0.02247)
    assert _value(frame, "PM10", "tyre") == pytest.approx(1.39 * 0.02247 * 0.6)


def test_brake_known_value() -> None:
    frame = emi_europe_emep_wear(
        1.0, 30.0, "TSP", ["Ubus Std 15 - 18 t"], [1.0], load=0.5, process="brake"
    )
    # lcf = 1 + 0.79*0.5 = 1.395; EF = 1.956*1.395*0.0075; sc(30) = 1.67
    assert _value(frame, "TSP", "brake") == pytest.approx(1.67 * 1.956 * 1.395 * 0.0075)


def test_road_known_value_and_no_speed_term() -> None:
    slow = emi_europe_emep_wear(1.0, 10.0, "PM10", ["Ubus Std 15 - 18 t"], [1.0], process="road")
    fast = emi_europe_emep_wear(1.0, 90.0, "PM10", ["Ubus Std 15 - 18 t"], [1.0], process="road")
    assert _value(slow, "PM10", "road") == pytest.approx(0.0760 * 0.5)
    assert _value(slow, "PM10", "road") == pytest.approx(_value(fast, "PM10", "road"))


def test_fleet_composition_weights_the_axles() -> None:
    weighted = emi_europe_emep_wear(
        1.0,
        30.0,
        "TSP",
        ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"],
        [0.5, 0.5],
        load=0.5,
        process="tyre",
    )
    # effective axles = 0.5*2 + 0.5*3 = 2.5
    expected = 1.39 * (0.5 * 2.5 * (1.41 + 1.38 * 0.5) * 0.0107)
    assert weighted["emi_g"].sum() == pytest.approx(expected)


def test_tyre_speed_boundaries() -> None:
    speeds = [39.9, 40.0, 90.0, 90.1]
    frame = emi_europe_emep_wear(
        [1.0, 1.0, 1.0, 1.0], speeds, "TSP", ["Ubus Std 15 - 18 t"], [1.0], process="tyre"
    )
    ef = 0.5 * 2 * (1.41 + 1.38 * 0.5) * 0.0107
    expected = [1.39, -0.00974 * 40 + 1.78, -0.00974 * 90 + 1.78, 0.902]
    by_segment = frame[frame["vehicle_index"] == 0].sort_values("segment")["emi_g"]
    assert by_segment.tolist() == pytest.approx([value * ef for value in expected])


def test_brake_speed_boundaries() -> None:
    speeds = [39.9, 40.0, 90.0, 90.1, 92.0, 95.0, 95.1]
    frame = emi_europe_emep_wear(
        [1.0] * len(speeds), speeds, "TSP", ["Ubus Std 15 - 18 t"], [1.0], process="brake"
    )
    ef = 1.956 * (1 + 0.79 * 0.5) * 0.0075
    expected = [
        1.67,
        -0.027 * 40 + 2.75,
        -0.027 * 90 + 2.75,
        0.185,
        0.185,
        0.185,
        0.185,
    ]
    by_segment = frame[frame["vehicle_index"] == 0].sort_values("segment")["emi_g"]
    assert by_segment.tolist() == pytest.approx([value * ef for value in expected])


def test_road_rejects_fine_particles() -> None:
    with pytest.raises(WearError, match="road wear has no factor"):
        emi_europe_emep_wear(1.0, 30.0, ["PM1.0"], ["Ubus Std 15 - 18 t"], [1.0], process="road")


def test_load_can_vary_per_segment() -> None:
    frame = emi_europe_emep_wear(
        [1.0, 1.0],
        30.0,
        "TSP",
        ["Ubus Std 15 - 18 t"],
        [1.0],
        load=[0.0, 1.0],
        process="tyre",
    )
    by_segment = frame.sort_values("segment")["emi_g"].tolist()
    empty = 1.39 * (0.5 * 2 * 1.41 * 0.0107)
    full = 1.39 * (0.5 * 2 * (1.41 + 1.38) * 0.0107)
    assert by_segment == pytest.approx([empty, full])


def test_shape_is_segment_x_pollutant_x_process_x_vehicle() -> None:
    frame = emi_europe_emep_wear(
        [1.0, 2.0],
        [10.0, 50.0],
        ["TSP", "PM10"],
        ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"],
        [0.6, 0.4],
        process=["tyre", "brake"],
    )
    assert len(frame) == 2 * 2 * 2 * 2
    assert list(frame.columns) == [
        "segment",
        "pollutant",
        "process",
        "veh_type",
        "vehicle_index",
        "emi_g",
    ]
    assert frame["emi_g"].ge(0).all()


def test_invalid_lengths_and_values() -> None:
    with pytest.raises(WearError, match="speed_kmh"):
        emi_europe_emep_wear([1.0, 2.0], [10.0, 20.0, 30.0], "TSP", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="speed_kmh"):
        emi_europe_emep_wear(1.0, 0.0, "TSP", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="load"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0], load=1.5)
    with pytest.raises(WearError, match="process"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0], process="magic")
    with pytest.raises(ValueError, match="fleet_composition"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.5])
    with pytest.raises(WearError, match="fleet_composition"):
        emi_europe_emep_wear(
            1.0,
            30.0,
            "TSP",
            ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"],
            [1.0],
        )
    assert np.isfinite(
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0])["emi_g"]
    ).all()


def test_wear_validation_branches() -> None:
    with pytest.raises(ValueError, match="dist_km must not be empty"):
        emi_europe_emep_wear([], 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        emi_europe_emep_wear([float("nan")], 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="between 1 and 130"):
        emi_europe_emep_wear(1.0, 0.5, "TSP", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0], load=2.0)
    with pytest.raises(WearError, match="unsupported process"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [1.0], process="noise")
    with pytest.raises(WearError, match="unsupported pollutant"):
        emi_europe_emep_wear(1.0, 30.0, "CO", "Ubus Std 15 - 18 t", [1.0])
    with pytest.raises(ValueError, match="fleet_composition must not be empty"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "Ubus Std 15 - 18 t", [])
    with pytest.raises(WearError, match="veh_type must have length"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", ["Ubus Std 15 - 18 t"] * 2, [1.0])
    with pytest.raises(WearError, match="unsupported veh_type"):
        emi_europe_emep_wear(1.0, 30.0, "TSP", "BUS_MAGIC", [1.0])
