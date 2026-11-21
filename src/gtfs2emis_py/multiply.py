"""EF x distance x fleet composition.

One arithmetic path for a constant factor and a per-segment factor: the caller
supplies a tidy frame (``pollutant``, ``vehicle_index``, ``ef``) and gets grams
back, either per vehicle or aggregated per pollutant. No segments x fleet x
pollutant cross-product is materialised.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pandas as pd

__all__ = ["MultiplyError", "multiply_ef"]


class MultiplyError(ValueError):
    """The EF frame or fleet composition is not usable."""


def _as_tuple(value: object, name: str) -> tuple:
    """Normalize a scalar or sequence to a tuple."""
    if isinstance(value, str | numbers.Real) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, Sequence | np.ndarray | pd.Series):
        return tuple(cast(Iterable[object], value))
    raise TypeError(f"{name} must be a scalar or a sequence, got {type(value).__name__}")


_MASS_UNITS = {
    "g/km": 1.0,
    "kg/km": 1000.0,
    "g/mi": 1.0 / 1.609344,
    "g/m": 1000.0,
}
_MASS_UNIT_RE = re.compile(r"^(?P<mass>g|kg)/(?P<distance>km|mi|m)$")
_MASS_FACTORS = {"g": 1.0, "kg": 1000.0}
_DISTANCE_KM = {"km": 1.0, "mi": 1.609344, "m": 0.001}


def _factor_to_grams(unit: str) -> float:
    """Return the multiplier that turns a mass-per-distance factor into g/km.

    Only mass-per-distance units are accepted. Energy-per-distance (``MJ/km``,
    ``kJ/km``), energy-per-energy (``g/kWh``) and count-per-energy (``#/kWh``)
    factors need a value-unit conversion and are rejected here.
    """
    text = str(unit).strip()
    if text in _MASS_UNITS:
        return _MASS_UNITS[text]
    match = _MASS_UNIT_RE.fullmatch(text)
    if match is None:
        raise MultiplyError(
            f"unsupported emission-factor unit {unit!r}; multiply_ef accepts mass per "
            "distance (g/km, kg/km, g/mi, g/m) only"
        )
    return _MASS_FACTORS[match.group("mass")] / _DISTANCE_KM[match.group("distance")]


def multiply_ef(
    fleet_composition: Sequence[float],
    dist_km: float | Sequence[float],
    ef: pd.DataFrame,
    *,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Multiply emission factors by distance and fleet composition.

    Args:
        fleet_composition: Share of each vehicle, 0-1, indexed by
            ``vehicle_index``.
        dist_km: Segment length(s) in km.
        ef: Frame with ``pollutant``, ``vehicle_index``, ``ef`` and ``unit``.
            The unit must be mass per distance (``g/km``, ``kg/km``,
            ``g/mi`` or ``g/m``); it is converted to g/km before arithmetic.
            A constant factor omits ``segment``; a per-segment factor includes
            it, one row per segment x pollutant x vehicle.
        aggregate: Sum over vehicles per pollutant when True; keep the vehicle
            rows when False.

    Returns:
        ``segment, pollutant, emi_g`` when aggregated, plus ``vehicle_index``
        otherwise, sorted by segment and pollutant (and vehicle).

    Raises:
        MultiplyError: For a missing column, unsupported factor unit, duplicate
            factor, incomplete segment grid, or a composition shorter than the
            vehicle indices.
        ValueError: For negative/non-finite distance or composition outside
            [0, 1].
    """
    composition = np.asarray([float(value) for value in fleet_composition], dtype=float)
    if composition.size == 0:
        raise ValueError("fleet_composition must not be empty")
    if np.any(~np.isfinite(composition)) or np.any(composition < 0) or np.any(composition > 1):
        raise ValueError("fleet_composition values must be finite and between 0 and 1")

    distance = np.asarray(_as_tuple(dist_km, "dist_km"), dtype=float)
    if distance.size == 0:
        raise ValueError("dist_km must not be empty")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("dist_km must be finite and non-negative")
    n_segments = distance.size

    if not isinstance(ef, pd.DataFrame):
        raise TypeError("ef must be a pandas DataFrame")
    missing = {"pollutant", "vehicle_index", "ef", "unit"} - set(ef.columns)
    if missing:
        raise MultiplyError(f"ef is missing column(s) {sorted(missing)}")
    if ef.empty:
        raise MultiplyError("ef is empty")
    if ef["ef"].isna().any() or not np.isfinite(ef["ef"].to_numpy(dtype=float)).all():
        raise MultiplyError("ef values must be finite")
    if (ef["ef"].to_numpy(dtype=float) < 0).any():
        raise MultiplyError("ef values must be non-negative")
    # ``ef`` is a factor, not an emission. Normalise its denominator to km and
    # its mass unit to grams once, before any distance multiplication.
    ef = ef.copy()
    ef["ef"] = ef["ef"].to_numpy(dtype=float) * np.asarray(
        [_factor_to_grams(unit) for unit in ef["unit"]], dtype=float
    )
    vehicles = ef["vehicle_index"].to_numpy()
    if not np.issubdtype(vehicles.dtype, np.integer):
        if not np.all(np.equal(np.mod(vehicles, 1), 0)):
            raise MultiplyError("vehicle_index must be integers")
        vehicles = vehicles.astype(int)
    if vehicles.min() < 0 or vehicles.max() >= composition.size:
        raise MultiplyError(f"vehicle_index must be between 0 and {composition.size - 1}")

    if "segment" in ef.columns:
        segments = ef["segment"].to_numpy()
        if not np.issubdtype(segments.dtype, np.integer):
            if not np.all(np.equal(np.mod(segments, 1), 0)):
                raise MultiplyError("segment must be integers")
            segments = segments.astype(int)
        if segments.min() < 0 or segments.max() >= n_segments:
            raise MultiplyError(f"segment must be between 0 and {n_segments - 1}")
        if ef.duplicated(["segment", "pollutant", "vehicle_index"]).any():
            raise MultiplyError("ef has duplicate pollutant/vehicle/segment rows")
        expected = n_segments * ef[["pollutant", "vehicle_index"]].drop_duplicates().shape[0]
        if len(ef) != expected:
            raise MultiplyError(
                f"ef must have one row per segment x pollutant x vehicle "
                f"({expected}); it has {len(ef)}"
            )
        frame = ef.copy()
        frame["segment"] = segments
        frame["emi_g"] = (
            frame["ef"].to_numpy(dtype=float)
            * composition[frame["vehicle_index"].to_numpy(dtype=int)]
            * distance[segments]
        )
    else:
        pairs = ef[["pollutant", "vehicle_index", "ef"]].copy()
        if pairs.duplicated(["pollutant", "vehicle_index"]).any():
            raise MultiplyError("ef has duplicate pollutant/vehicle rows")
        pair_count = len(pairs)
        frame = pd.DataFrame(
            {
                "segment": np.repeat(np.arange(n_segments), pair_count),
                "pollutant": np.tile(pairs["pollutant"].to_numpy(), n_segments),
                "vehicle_index": np.tile(pairs["vehicle_index"].to_numpy(dtype=int), n_segments),
                "ef": np.tile(pairs["ef"].to_numpy(dtype=float), n_segments),
            }
        )
        frame["emi_g"] = (
            frame["ef"]
            * composition[frame["vehicle_index"].to_numpy(dtype=int)]
            * distance[frame["segment"].to_numpy(dtype=int)]
        )

    if aggregate:
        result = frame.groupby(["segment", "pollutant"], as_index=False, sort=True)["emi_g"].sum()
        return result.sort_values(["segment", "pollutant"], kind="stable").reset_index(drop=True)
    result = frame[["segment", "pollutant", "vehicle_index", "emi_g"]]
    return result.sort_values(["segment", "pollutant", "vehicle_index"], kind="stable").reset_index(
        drop=True
    )
