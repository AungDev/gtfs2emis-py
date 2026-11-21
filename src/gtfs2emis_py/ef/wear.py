"""EMEP/EEA tyre, brake and road wear emissions.

Port of `emi_europe_emep_wear()`. The reference returns emissions in grams, not
g/km: distance is multiplied in, fleet composition weights the vehicles, and the
speed/load corrections are applied before the mass fraction. This module keeps
that contract.

Road wear has no PM1.0/PM0.1 factors in the reference; asking for them is an
error naming the pair, not a zero.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pandas as pd

__all__ = ["PROCESSES", "WearError", "emi_europe_emep_wear"]

PROCESSES: tuple[str, ...] = ("tyre", "brake", "road")
_POLLUTANTS = ("TSP", "PM10", "PM2.5", "PM1.0", "PM0.1")
_ROAD_POLLUTANTS = ("TSP", "PM10", "PM2.5")
_VEHICLES = (
    "Ubus Midi <=15 t",
    "Ubus Std 15 - 18 t",
    "Ubus Artic >18 t",
    "Coaches Std <=18 t",
    "Coaches Artic >18 t",
)
_AXLES = {
    "Ubus Midi <=15 t": 2,
    "Ubus Std 15 - 18 t": 2,
    "Ubus Artic >18 t": 3,
    "Coaches Std <=18 t": 2,
    "Coaches Artic >18 t": 3,
}
_MASS_FRACTION = {
    "tyre": {"TSP": 1.000, "PM10": 0.600, "PM2.5": 0.420, "PM1.0": 0.060, "PM0.1": 0.048},
    "brake": {"TSP": 1.000, "PM10": 0.980, "PM2.5": 0.390, "PM1.0": 0.100, "PM0.1": 0.080},
    "road": {"TSP": 1.000, "PM10": 0.500, "PM2.5": 0.270},
}
_TYRE_PC_EF = 0.0107  # g/km, EMEP/EEA 2019
_BRAKE_PC_EF = 0.0075  # g/km, Table 3-6
_ROAD_EF = 0.0760  # g/km, urban bus


class WearError(ValueError):
    """A process/species or shape request the reference does not support."""


def _as_tuple(value: object, name: str) -> tuple:
    """Normalize a scalar or sequence to a tuple."""
    if isinstance(value, str | numbers.Real) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, Sequence | np.ndarray | pd.Series):
        return tuple(cast(Iterable[object], value))
    raise TypeError(f"{name} must be a scalar or a sequence, got {type(value).__name__}")


def _broadcast(value: object, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar to n values or require length n."""
    array = np.asarray(_as_tuple(value, name), dtype=float)
    if array.size == 1:
        return np.repeat(array, n)
    if array.size != n:
        raise WearError(f"{name} must have length 1 or {n}; it has {array.size}")
    return array


def _speed_correction(speed: np.ndarray, process: str) -> np.ndarray:
    """Apply the R wear speed correction for tyre, brake or road."""
    if process == "tyre":
        return np.where(
            speed < 40.0,
            1.39,
            np.where(speed <= 90.0, -0.00974 * speed + 1.78, 0.902),
        )
    return np.where(
        speed < 40.0,
        1.67,
        np.where(speed <= 90.0, -0.0270 * speed + 2.75, 0.185),
    )


def emi_europe_emep_wear(
    dist_km: float | Sequence[float],
    speed_kmh: float | Sequence[float],
    pollutant: str | Sequence[str],
    veh_type: str | Sequence[str],
    fleet_composition: Sequence[float],
    load: float | Sequence[float] = 0.5,
    process: str | Sequence[str] = "tyre",
) -> pd.DataFrame:
    """Compute tyre, brake and road wear emissions in grams.

    Args:
        dist_km: Segment length(s) in km.
        speed_kmh: Mean speed(s) in km/h, 1-130; length 1 or ``len(dist_km)``.
        pollutant: TSP, PM10, PM2.5, PM1.0 or PM0.1 (road: no PM1.0/PM0.1).
        veh_type: EMEP segment(s); length 1 or ``len(fleet_composition)``.
        fleet_composition: Share of each vehicle, 0-1.
        load: Load factor, 0-1; length 1 or ``len(dist_km)``.
        process: ``tyre``, ``brake`` and/or ``road``.

    Returns:
        A tidy frame: ``segment``, ``pollutant``, ``process``, ``veh_type``,
        ``vehicle_index``, ``emi_g``, sorted by segment/process/pollutant/vehicle.

    Raises:
        WearError: For an unsupported process/species pair or mismatched lengths.
        ValueError: For empty input or values outside the documented ranges.
    """
    distance = np.asarray(_as_tuple(dist_km, "dist_km"), dtype=float)
    if distance.size == 0:
        raise ValueError("dist_km must not be empty")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("dist_km must be finite and non-negative")
    n = distance.size
    speed = _broadcast(speed_kmh, n, "speed_kmh")
    if np.any(speed < 1) or np.any(speed > 130):
        raise ValueError("speed_kmh must be between 1 and 130")
    load_factor = _broadcast(load, n, "load")
    if np.any(load_factor < 0) or np.any(load_factor > 1):
        raise ValueError("load must be between 0 and 1")

    processes = tuple(str(value) for value in _as_tuple(process, "process"))
    unknown_processes = sorted(set(processes) - set(PROCESSES))
    if unknown_processes:
        raise WearError(f"unsupported process(es) {unknown_processes}; allowed: {PROCESSES}")
    pollutants = tuple(str(value) for value in _as_tuple(pollutant, "pollutant"))
    unknown_pollutants = sorted(set(pollutants) - set(_POLLUTANTS))
    if unknown_pollutants:
        raise WearError(f"unsupported pollutant(s) {unknown_pollutants}; allowed: {_POLLUTANTS}")
    if "road" in processes:
        forbidden = sorted(set(pollutants) - set(_ROAD_POLLUTANTS))
        if forbidden:
            raise WearError(f"road wear has no factor for {forbidden}; allowed: {_ROAD_POLLUTANTS}")

    composition = np.asarray([float(value) for value in fleet_composition], dtype=float)
    if composition.size == 0:
        raise ValueError("fleet_composition must not be empty")
    if np.any(composition < 0) or np.any(composition > 1):
        raise ValueError("fleet_composition values must be between 0 and 1")
    vehicles = tuple(str(value) for value in _as_tuple(veh_type, "veh_type"))
    if len(vehicles) == 1:
        vehicles = vehicles * composition.size
    if len(vehicles) != composition.size:
        raise WearError(
            "veh_type must have length 1 or the length of fleet_composition "
            f"({composition.size}); it has {len(vehicles)}"
        )
    unknown_vehicles = sorted(set(vehicles) - set(_VEHICLES))
    if unknown_vehicles:
        raise WearError(f"unsupported veh_type(s) {unknown_vehicles}")
    axles = np.asarray([_AXLES[vehicle] for vehicle in vehicles], dtype=float)

    frames: list[pd.DataFrame] = []
    for process_name in processes:
        speed_correction = _speed_correction(speed, process_name)
        for pollutant_name in pollutants:
            fraction = _MASS_FRACTION[process_name][pollutant_name]
            if process_name == "tyre":
                lcf = 1.41 + 1.38 * load_factor
                ef = 0.5 * axles[None, :] * lcf[:, None] * _TYRE_PC_EF
                emission = (
                    distance[:, None]
                    * speed_correction[:, None]
                    * ef
                    * composition[None, :]
                    * fraction
                )
            elif process_name == "brake":
                lcf = 1.00 + 0.79 * load_factor
                ef = 1.956 * lcf * _BRAKE_PC_EF
                emission = (
                    distance[:, None]
                    * speed_correction[:, None]
                    * ef[:, None]
                    * composition[None, :]
                    * fraction
                )
            else:
                emission = distance[:, None] * _ROAD_EF * fraction * composition[None, :]
            frames.append(
                pd.DataFrame(
                    {
                        "segment": np.repeat(np.arange(n), len(vehicles)),
                        "pollutant": pollutant_name,
                        "process": process_name,
                        "veh_type": np.tile(np.asarray(vehicles), n),
                        "vehicle_index": np.tile(np.arange(len(vehicles)), n),
                        "emi_g": emission.reshape(-1),
                    }
                )
            )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["segment", "process", "pollutant", "vehicle_index"], kind="stable"
    ).reset_index(drop=True)
