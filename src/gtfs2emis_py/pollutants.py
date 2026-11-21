"""Pollutant registry: canonical unit and kind per pollutant.

An emission factor is not always ``g/km``. R's ``gtfs2emis`` tags every factor
``g/km`` unconditionally (``R/ef_europe_emep.R:352``), which turns EMEP's energy
consumption into grams and MOVES' kilojoules into grams — the defect this
registry fixes. Each pollutant declares the unit its factor is actually stored
in, and the kind of quantity that unit measures.

The registry is deliberately small. It holds what the four EF families produce
that the port can name with a verified unit:

- **Mass** pollutants in ``g/km``.
- **Energy** consumption (``EC``) in ``MJ/km`` for EMEP and ``kJ/km`` for MOVES.
  Two models, two units, one pollutant — hence ``Pollutant.units_by_model``.
- **Particle number** (``SPN25``) in ``#/kWh``.

CETESB's ``FC`` (fuel consumption), ``KML`` (kilometres per litre) and ``FS``
(fuel sales) are *not* registered: R declares them ``g/km`` and the shipped
table does not say otherwise. Their units are settled when the CETESB port
The CETESB loader reads the table, and adding them here would invent a
conversion. ``volume`` is therefore declared in :data:`PollutantKind` and used
by no entry yet; the vocabulary is fixed ahead of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Pollutant",
    "PollutantKind",
    "UnknownPollutantError",
    "emission_value_unit",
    "get_pollutant",
    "list_pollutants",
    "pollutant_unit",
]

PollutantKind = Literal["mass", "energy", "count", "volume"]

# Mass pollutants, all in g/km. Names follow the R tables where they exist; the
# aliases cover the spellings a caller is likely to use.
_MASS_POLLUTANTS: tuple[str, ...] = (
    "CO",
    "CO2",
    "CH4",
    "N2O",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "HONO",
    "NH4",
    "NO3",
    "SO2",
    "SOX",
    "ROG",
    "TOG",
    "PM10",
    "PM2.5",
    "PM1.0",
    "PM0.1",
    "TSP",
    "BC",
    "OC",
    "VOC",
    "NMHC",
    "THC",
    "HC",
    "FC",
    "RCHO",
    "ETOH",
)


@dataclass(frozen=True)
class Pollutant:
    """A pollutant's canonical emission-factor unit and quantity kind.

    Attributes:
        name: Canonical name, as used in output tables.
        unit: The unit a factor for this pollutant is stored in by default.
        kind: What the unit measures — ``mass``, ``energy``, ``count`` or
            ``volume``.
        units_by_model: Model-specific overrides, keyed by the ``ef_model`` name
            (e.g. ``EC`` is ``MJ/km`` for EMEP and ``kJ/km`` for MOVES). The
            dataclass is frozen so the attribute is never reassigned; the
            mapping itself is a plain ``dict``, so callers must not mutate it.
            :func:`get_pollutant` hands out a copy for exactly that reason.
    """

    name: str
    unit: str
    kind: PollutantKind
    units_by_model: dict[str, str] = field(default_factory=dict)

    def unit_for(self, model: str | None = None) -> str:
        """Return the factor unit for ``model``, falling back to the canonical one.

        A model without an explicit override uses ``unit``: only the models that
        disagree with the default are listed, which keeps the registry honest
        about what has actually been verified.
        """
        if model is None:
            return self.unit
        return self.units_by_model.get(model, self.unit)


class UnknownPollutantError(ValueError):
    """Raised when a pollutant name is not in the registry.

    The message lists the supported names, because the alternative — a silent
    ``None`` — turns a typo into a missing column three functions later.
    """


_POLLUTANTS: dict[str, Pollutant] = {
    **{name: Pollutant(name=name, unit="g/km", kind="mass") for name in _MASS_POLLUTANTS},
    "EC": Pollutant(
        name="EC",
        unit="MJ/km",
        kind="energy",
        units_by_model={"ef_europe_emep": "MJ/km", "ef_usa_moves": "kJ/km"},
    ),
    "SPN25": Pollutant(name="SPN25", unit="#/kWh", kind="count"),
    # EMEP reports fuel consumption in g/km (its roxygen states the unit); the
    # CETESB table's FC has no unit and is rejected before this registry is
    # consulted, so the default here cannot leak into the CETESB path.
    "FC": Pollutant(name="FC", unit="g/km", kind="mass", units_by_model={"ef_europe_emep": "g/km"}),
    # CETESB reports two energy-normalised metrics whose names state the unit
    # (`g/kWh`); they are mass per energy, not g/km.
    "gCO2/KWH": Pollutant(name="gCO2/KWH", unit="g/kWh", kind="mass"),
    "gD/KWH": Pollutant(name="gD/KWH", unit="g/kWh", kind="mass"),
}

# Lower-cased spelling (and alias) -> canonical name. Case-insensitive matching
# is what lets "nox", "NOX" and "NOx" be the same pollutant.
_ALIASES: dict[str, str] = {
    **{name.casefold(): name for name in _POLLUTANTS},
    "nox": "NOx",
    "nmvoc": "VOC",
    "pm25": "PM2.5",
    "pm2_5": "PM2.5",
    "pm1": "PM1.0",
    "pm0.1": "PM0.1",
    "spn": "SPN25",
}


def list_pollutants() -> tuple[str, ...]:
    """Return every registered canonical pollutant name, sorted."""
    return tuple(sorted(_POLLUTANTS))


def _copy_of(pollutant: Pollutant) -> Pollutant:
    """Return a copy whose ``units_by_model`` mapping the caller owns.

    ``frozen=True`` stops attribute reassignment; it does not stop a caller from
    assigning into ``pollutant.units_by_model``. Handing out the module-level
    instance would therefore let one caller change every later lookup in the
    process.
    """
    return Pollutant(
        name=pollutant.name,
        unit=pollutant.unit,
        kind=pollutant.kind,
        units_by_model=dict(pollutant.units_by_model),
    )


def get_pollutant(name: str, *, model: str | None = None) -> Pollutant:
    """Look up a pollutant by name or alias.

    Args:
        name: Pollutant name, case-insensitive (``"nox"``, ``"NOx"``).
        model: Optional ``ef_model`` name; when given, ``unit_for(model)`` is
            applied before returning, so ``get_pollutant("EC", model="ef_usa_moves")``
            carries ``kJ/km``.

    Returns:
        A copy of the registry entry, with ``unit`` already resolved for
        ``model``. Mutating the returned object's ``units_by_model`` mapping does
        not change the registry, so repeated lookups are stable.

    Raises:
        UnknownPollutantError: If ``name`` is not registered; the message lists
            the supported names.
    """
    canonical = _ALIASES.get(name.strip().casefold())
    if canonical is None:
        raise UnknownPollutantError(
            f"unknown pollutant {name!r}; registered pollutants are {', '.join(list_pollutants())}"
        )
    pollutant = _POLLUTANTS[canonical]
    if model is None:
        return _copy_of(pollutant)
    return Pollutant(
        name=pollutant.name,
        unit=pollutant.unit_for(model),
        kind=pollutant.kind,
        units_by_model=dict(pollutant.units_by_model),
    )


def pollutant_unit(name: str, *, model: str | None = None) -> str:
    """Return the factor unit for a pollutant, optionally for a specific model."""
    return get_pollutant(name, model=model).unit


def emission_value_unit(name: str, *, model: str | None = None) -> str:
    """Return the unit of an emitted *value* (not its emission factor).

    A factor carries a denominator (``g/km``, ``MJ/km``, ``g/kWh``); the value
    obtained by applying it over a distance is grams, megajoules or kilojoules.
    Mass pollutants therefore emit in ``g``; EMEP energy emits in ``MJ`` and
    MOVES energy in ``kJ``. Count and volume values use ``#`` and ``L`` unless a
    model-specific conversion is added by a later task.
    """
    pollutant = get_pollutant(name, model=model)
    if pollutant.kind == "mass":
        return "g"
    if pollutant.kind == "energy":
        return "kJ" if model == "ef_usa_moves" else "MJ"
    if pollutant.kind == "count":
        return "#"
    return "L"
