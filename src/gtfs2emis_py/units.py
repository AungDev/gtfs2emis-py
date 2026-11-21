"""Canonical units and boundary checks.

The package computes on plain ``float64`` values in documented canonical units:
``km`` (distance), ``km/h`` (speed), ``g/km`` (mass emission factor), ``g``
(mass), ``m`` (spatial resolution) and ``s`` (time). No unit library is involved
anywhere — ``pint`` is not a dependency and must not become one.

This module is the boundary those values pass through. A public function that
takes a physical quantity accepts a number plus a unit name, converts it to the
canonical unit here, and then works in floats. Everything downstream may assume
the canonical unit and never converts again.

What the checks are, and are not:

- They reject non-finite values, booleans (``True`` is an ``int`` in Python and
  would silently become ``1.0`` km), negative distances/speeds/factors, and
  unknown unit names with a message listing the accepted ones.
- They cannot catch a *wrong* unit of the right dimension — 30 passed as km/h
  when the caller meant mph is indistinguishable from a valid speed. Every entry
  point therefore documents its unit, and the range checks that catch the gross
  cases live with the functions that own them.
"""

from __future__ import annotations

import math
from typing import Final, Literal

__all__ = [
    "CANONICAL_DISTANCE",
    "CANONICAL_EMISSION_FACTOR",
    "CANONICAL_LENGTH",
    "CANONICAL_MASS",
    "CANONICAL_SPEED",
    "CANONICAL_TIME",
    "UnitError",
    "as_g",
    "as_g_per_km",
    "as_km",
    "as_kmh",
]

DistanceUnit = Literal["km", "m", "mi", "ft"]
SpeedUnit = Literal["km/h", "m/s", "mph"]
EmissionFactorUnit = Literal["g/km", "g/mi", "g/m", "kg/km"]
MassUnit = Literal["g", "kg", "mg", "t"]

CANONICAL_DISTANCE: Final = "km"
CANONICAL_SPEED: Final = "km/h"
CANONICAL_EMISSION_FACTOR: Final = "g/km"
CANONICAL_MASS: Final = "g"
CANONICAL_LENGTH: Final = "m"
CANONICAL_TIME: Final = "s"

_KM_PER_MILE: Final = 1.609344
_KM_PER_FOOT: Final = 0.0003048

_DISTANCE_TO_KM: Final[dict[str, float]] = {
    "km": 1.0,
    "m": 0.001,
    "mi": _KM_PER_MILE,
    "ft": _KM_PER_FOOT,
}

_SPEED_TO_KMH: Final[dict[str, float]] = {
    "km/h": 1.0,
    "m/s": 3.6,
    "mph": _KM_PER_MILE,
}

# Mass of pollutant per distance. ``g/m`` and ``kg/km`` are the same quantity,
# which is why the registry can accept both without inventing a conversion.
_EF_TO_G_PER_KM: Final[dict[str, float]] = {
    "g/km": 1.0,
    "g/mi": 1.0 / _KM_PER_MILE,
    "g/m": 1000.0,
    "kg/km": 1000.0,
}

_MASS_TO_G: Final[dict[str, float]] = {
    "g": 1.0,
    "kg": 1000.0,
    "mg": 0.001,
    "t": 1_000_000.0,
}


class UnitError(ValueError):
    """Raised when a value cannot be interpreted in the requested unit.

    Covers a non-numeric or boolean input, a non-finite value, a negative
    distance/speed/factor, and an unknown unit name.
    """


def _to_canonical(
    value: float,
    unit: str,
    factors: dict[str, float],
    *,
    quantity: str,
) -> float:
    """Convert ``value`` from ``unit`` to the canonical unit of ``quantity``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UnitError(f"{quantity} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise UnitError(f"{quantity} must be finite, got {value!r}")
    if number < 0:
        raise UnitError(f"{quantity} must be non-negative, got {value!r}")
    try:
        factor = factors[unit]
    except KeyError:
        raise UnitError(
            f"unsupported {quantity} unit {unit!r}; expected one of {sorted(factors)}"
        ) from None
    return number * factor


def as_km(value: float, unit: DistanceUnit = "km") -> float:
    """Return a distance in kilometres.

    Args:
        value: Non-negative, finite distance.
        unit: One of ``km``, ``m``, ``mi``, ``ft``.

    Raises:
        UnitError: For a non-finite or negative value, or an unknown unit.
    """
    return _to_canonical(value, unit, _DISTANCE_TO_KM, quantity="distance")


def as_kmh(value: float, unit: SpeedUnit = "km/h") -> float:
    """Return a speed in kilometres per hour.

    Args:
        value: Non-negative, finite speed.
        unit: One of ``km/h``, ``m/s``, ``mph``.

    Raises:
        UnitError: For a non-finite or negative value, or an unknown unit.
    """
    return _to_canonical(value, unit, _SPEED_TO_KMH, quantity="speed")


def as_g_per_km(value: float, unit: EmissionFactorUnit = "g/km") -> float:
    """Return a mass emission factor in grams per kilometre.

    Args:
        value: Non-negative, finite factor.
        unit: One of ``g/km``, ``g/mi``, ``g/m``, ``kg/km``.

    Raises:
        UnitError: For a non-finite or negative value, or an unknown unit.
    """
    return _to_canonical(value, unit, _EF_TO_G_PER_KM, quantity="emission factor")


def as_g(value: float, unit: MassUnit = "g") -> float:
    """Return a mass in grams.

    Args:
        value: Non-negative, finite mass.
        unit: One of ``g``, ``kg``, ``mg``, ``t``.

    Raises:
        UnitError: For a non-finite or negative value, or an unknown unit.
    """
    return _to_canonical(value, unit, _MASS_TO_G, quantity="mass")
