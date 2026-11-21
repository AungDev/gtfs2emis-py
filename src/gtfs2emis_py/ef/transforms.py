"""EF transforms: fuel correction, speed scaling and the CETESB composite.

These are not source models: they take a local factor and reshape
it, so they live outside the four-model class and are dispatched by name.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from gtfs2emis_py.ef.base import (
    EmissionFactors,
    MissingEmissionFactorError,
    as_tuple,
)
from gtfs2emis_py.ef.brazil_cetesb import ef_brazil_cetesb
from gtfs2emis_py.ef.europe_emep import ef_europe_emep
from gtfs2emis_py.pollutants import pollutant_unit

__all__ = ["ef_brazil_scaled_euro", "ef_scaled_euro", "fuel_correction"]

_EURO_STAGES = ("PRE", "I", "II", "III", "IV", "V", "VI")
_BASE_FUEL_YEAR = {
    "PRE": 1996,
    "I": 1996,
    "II": 2000,
    "III": 2005,
    "IV": 2005,
    "V": 2005,
    "VI": 2005,
}
_DIESEL = {
    1996: {"den": 840.0, "s": 400.0, "pah": 9.0, "cn": 51.0, "t95": 350.0},
    2000: {"den": 840.0, "s": 300.0, "pah": 7.0, "cn": 53.0, "t95": 330.0},
}
_DEFAULT_IMPROVED = {"den": 835.0, "s": 40.0, "pah": 5.0, "cn": 53.0, "t95": 320.0}
_FUEL_KEYS = ("den", "s", "pah", "cn", "t95")
_CORRECTED = frozenset({"CO", "VOC", "NOx", "PM10"})


def _fuel_factor(pollutant: str, fuel: Mapping[str, float]) -> float:
    """Evaluate the EMEP fuel-correction factor for one pollutant."""
    den = fuel["den"]
    s = fuel["s"]
    pah = fuel["pah"]
    cn = fuel["cn"]
    t95 = fuel["t95"]
    if pollutant == "CO":
        return 2.24407 - 0.0011 * den + 0.00007 * pah - 0.00768 * cn - 0.00087 * t95
    if pollutant == "VOC":
        return 1.61466 - 0.00123 * den + 0.00133 * pah - 0.00181 * cn - 0.00068 * t95
    if pollutant == "NOx":
        return -1.75444 + 0.00906 * den - 0.0163 * pah + 0.00493 * cn + 0.00266 * t95
    if pollutant == "PM10":
        return (0.06959 + 0.00006 * den + 0.00065 * pah - 0.00001 * cn) * (
            1 - 0.0086 * (450 - s) / 100
        )
    raise ValueError(f"unsupported pollutant {pollutant!r}; expected {sorted(_CORRECTED)}")


def fuel_correction(
    pollutant: str | Sequence[str],
    euro_stage: str | Sequence[str],
    improved_fuel: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return the EMEP fuel-correction ratio per pollutant and Euro stage.

    Args:
        pollutant: CO, VOC, NOx or PM10.
        euro_stage: PRE, I, II, III, IV, V or VI.
        improved_fuel: Densities/sulphur/PAH/cetane/T95 for the improved fuel;
            defaults to the reference's ``den=835, s=40, pah=5, cn=53, t95=320``.

    Returns:
        A tidy frame with ``euro_stage``, ``pollutant`` and ``fcorr``.
    """
    pollutants = [str(value) for value in as_tuple(pollutant, "pollutant")]
    stages = [str(value) for value in as_tuple(euro_stage, "euro_stage")]
    unknown_pollutants = sorted(set(pollutants) - _CORRECTED)
    if unknown_pollutants:
        raise ValueError(
            f"unsupported fuel-correction pollutant(s) {unknown_pollutants}; "
            f"expected {sorted(_CORRECTED)}"
        )
    unknown_stages = sorted(set(stages) - set(_EURO_STAGES))
    if unknown_stages:
        raise ValueError(f"unsupported euro_stage(s) {unknown_stages}")
    improved = dict(_DEFAULT_IMPROVED if improved_fuel is None else improved_fuel)
    if set(improved) != set(_FUEL_KEYS):
        raise ValueError(f"improved_fuel must have exactly the keys {_FUEL_KEYS}")
    for key, value in improved.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"improved_fuel[{key!r}] must be numeric")
        improved[key] = float(value)
        if not math.isfinite(improved[key]):
            raise ValueError(f"improved_fuel[{key!r}] must be finite")
    fuels = {**_DIESEL, 2005: improved}
    rows = []
    for stage in stages:
        base = fuels[_BASE_FUEL_YEAR[stage]]
        for name in pollutants:
            rows.append(
                {
                    "euro_stage": stage,
                    "pollutant": name,
                    "fcorr": _fuel_factor(name, improved) / _fuel_factor(name, base),
                }
            )
    return pd.DataFrame(rows, columns=["euro_stage", "pollutant", "fcorr"])


def _local_values(
    ef_local: EmissionFactors | pd.DataFrame | Sequence[float],
    pollutants: list[str],
    n_vehicles: int,
) -> np.ndarray:
    """Validate and return the local EF vector in pollutant-major order."""
    expected = len(pollutants) * n_vehicles
    if isinstance(ef_local, pd.DataFrame):
        if "ef" not in ef_local.columns:
            raise ValueError("ef_local DataFrame must have an 'ef' column")
        values = ef_local["ef"].to_numpy(dtype=float)
        if len(ef_local) != expected:
            raise ValueError(
                "ef_local DataFrame must hold one factor per pollutant x vehicle "
                f"in pollutant-major, vehicle-minor order ({expected} rows); "
                f"it has {len(ef_local)}"
            )
        return values
    if isinstance(ef_local, EmissionFactors):
        table = ef_local.table
        values = table["ef"].to_numpy(dtype=float)
        if len(table) != expected:
            raise ValueError(
                f"ef_local must hold one factor per pollutant x vehicle "
                f"({expected} rows); it has {len(table)}"
            )
        return values
    if isinstance(ef_local, Sequence | np.ndarray | pd.Series):
        values = np.asarray(ef_local, dtype=float)
        if values.size != expected:
            raise ValueError(
                f"ef_local must have {expected} values (pollutant-major, vehicle-minor); "
                f"it has {values.size}"
            )
        return values
    raise TypeError(
        "ef_local must be an EmissionFactors, a DataFrame with an 'ef' column, "
        "or a sequence of numbers"
    )


def ef_scaled_euro(
    ef_local: EmissionFactors | pd.DataFrame | Sequence[float],
    speed: float | Sequence[float],
    veh_type: str | Sequence[str],
    euro: str | Sequence[str],
    pollutant: str | Sequence[str],
    fuel: str | Sequence[str] = "D",
    tech: str | Sequence[str] = "SCR",
    SDC: float = 19.0,
    slope: float | Sequence[float] = 0.0,
    load: float | Sequence[float] = 0.5,
    fcorr: float | Sequence[float] = 1.0,
) -> EmissionFactors:
    """Scale local EFs by the EMEP speed curve: ``EF_local * EF(V) / EF(SDC)``.

    ``EF(SDC)`` is evaluated at the same ``slope`` and ``load`` as the
    numerator, matching R's ``ef_scaled_euro()`` call. ``EC`` is rejected: it is
    an energy factor, not g/km.

    Args:
        ef_local: Local factors: an :class:`EmissionFactors`, a DataFrame with an
            ``ef`` column (one row per pollutant x vehicle), or a flat sequence.
            Rows/values are in pollutant-major, vehicle-minor order, matching the
            reference's positional order.
        speed: Speed(s) in km/h.
        veh_type: EMEP segment(s), aligned with the local vehicle order.
        euro: Euro standard(s).
        pollutant: Pollutant names, in the local-factor order.
        fuel, tech, slope, load, fcorr: Passed to :func:`ef_europe_emep`.
        SDC: Reference urban speed in km/h; 19 is the EMEP default.

    Returns:
        An :class:`EmissionFactors` in g/km with one row per
        ``(pollutant, vehicle, speed)``.
    """
    pollutants = [str(value) for value in as_tuple(pollutant, "pollutant")]
    vehicles = [str(value) for value in as_tuple(veh_type, "veh_type")]
    if len(vehicles) != len([str(v) for v in as_tuple(euro, "euro")]):
        raise ValueError("veh_type and euro must have the same length")
    if not math.isfinite(float(SDC)) or float(SDC) <= 0:
        raise ValueError("SDC must be finite and > 0")
    excluded = sorted(set(pollutants) & {"EC"})
    if excluded:
        raise ValueError(f"EC cannot be scaled to a local g/km factor: {excluded}")
    n_vehicles = len(vehicles)
    n_speeds = len(as_tuple(speed, "speed"))
    local = _local_values(ef_local, pollutants, n_vehicles)

    reference = ef_europe_emep(
        SDC,
        vehicles,
        euro,
        pollutants,
        fuel=fuel,
        tech=tech,
        slope=slope,
        load=load,
        fcorr=fcorr,
    )
    at_speed = ef_europe_emep(
        speed,
        vehicles,
        euro,
        pollutants,
        fuel=fuel,
        tech=tech,
        slope=slope,
        load=load,
        fcorr=fcorr,
    )
    ratio = _matrix(reference, pollutants, n_vehicles, 1)  # (1, pairs)
    values = _matrix(at_speed, pollutants, n_vehicles, n_speeds)  # (n_speed, pairs)
    factor = local / ratio.reshape(-1)
    scaled = values * factor.reshape(1, -1)

    frames: list[pd.DataFrame] = []
    for index, name in enumerate(pollutants):
        block = at_speed.table[at_speed.table["pollutant"] == name].copy()
        block["ef"] = scaled[:, index * n_vehicles : (index + 1) * n_vehicles].reshape(-1)
        frames.append(block)
    table = pd.concat(frames, ignore_index=True)
    table["unit"] = table["pollutant"].map(lambda value: pollutant_unit(value))
    return EmissionFactors(table=table, ef_model="ef_scaled_euro", process="hot_exhaust")


def _matrix(
    factors: EmissionFactors,
    pollutants: list[str],
    n_vehicles: int,
    n_speeds: int,
) -> np.ndarray:
    """Return an (n_speed, pollutant-major x vehicle) matrix in input order.

    ``ef_europe_emep`` emits each pollutant block speed-major and vehicle-minor,
    so a reshape is enough — no pivot, and duplicate speeds keep their order.
    """
    blocks: list[np.ndarray] = []
    for name in pollutants:
        block = factors.table[factors.table["pollutant"] == name]
        values = block["ef"].to_numpy(dtype=float)
        if values.size != n_speeds * n_vehicles:
            raise ValueError(
                f"internal EMEP grid for {name!r} has {values.size} rows; "
                f"expected {n_speeds * n_vehicles}"
            )
        blocks.append(values.reshape(n_speeds, n_vehicles))
    return np.concatenate(blocks, axis=1)


def ef_brazil_scaled_euro(
    pollutant: str | Sequence[str],
    speed: float | Sequence[float],
    veh_type: str | Sequence[str],
    model_year: int | Sequence[int],
    type_name_eu: str | Sequence[str],
    euro: str | Sequence[str],
    fuel: str | Sequence[str] = "D",
    tech: str | Sequence[str] = "SCR",
    SDC: float = 19.0,
    slope: float | Sequence[float] = 0.0,
    load: float | Sequence[float] = 0.5,
    fcorr: float | Sequence[float] = 1.0,
) -> EmissionFactors:
    """CETESB local factors scaled by the EMEP speed curve (the composite model).

    ``veh_type`` are CETESB classes (BUS_URBAN_D, ...) and ``type_name_eu`` the
    matching EMEP segments, position by position.
    """
    pollutants = [str(value) for value in as_tuple(pollutant, "pollutant")]
    br_vehicles = [str(value) for value in as_tuple(veh_type, "veh_type")]
    emep_vehicles = [str(value) for value in as_tuple(type_name_eu, "type_name_eu")]
    if len(br_vehicles) != len(emep_vehicles):
        raise ValueError("veh_type and type_name_eu must have the same length")
    local = ef_brazil_cetesb(pollutants, br_vehicles, model_year)
    values: list[float] = []
    for name in pollutants:
        for vehicle in br_vehicles:
            rows = local.table[
                (local.table["pollutant"] == name) & (local.table["veh_type"] == vehicle)
            ]
            if len(rows) != 1:
                raise MissingEmissionFactorError(
                    f"expected one local CETESB factor for {name}/{vehicle}; found {len(rows)}"
                )
            values.append(float(rows["ef"].iloc[0]))
    scaled = ef_scaled_euro(
        values,
        speed,
        emep_vehicles,
        euro,
        pollutants,
        fuel=fuel,
        tech=tech,
        SDC=SDC,
        slope=slope,
        load=load,
        fcorr=fcorr,
    )
    return EmissionFactors(
        table=scaled.table,
        ef_model="ef_brazil_scaled_euro",
        process="hot_exhaust",
    )
