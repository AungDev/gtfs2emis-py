"""Shared EF result type and lookup errors."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

__all__ = [
    "EmissionFactors",
    "MissingEmissionFactorError",
    "UnsupportedPollutantError",
    "as_tuple",
]

Process = Literal["hot_exhaust", "tyre", "brake", "road"]


def as_tuple(value: object, name: str) -> tuple:
    """Return a scalar as a one-tuple and any vector-like input as a tuple.

    Accepts Python sequences plus NumPy arrays and pandas Series, which are not
    ``collections.abc.Sequence`` but are what callers actually pass.
    """
    if isinstance(value, str | numbers.Real) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, Sequence | np.ndarray | pd.Series):
        return tuple(cast(Iterable[object], value))
    raise TypeError(f"{name} must be a scalar or a sequence, got {type(value).__name__}")


@dataclass(frozen=True)
class EmissionFactors:
    """A tidy emission-factor table for one EF model.

    Attributes:
        table: One row per ``(pollutant, veh_type, model_year)`` with columns
            ``pollutant``, ``veh_type``, ``model_year``, ``ef`` and ``unit``.
            ``ef`` is a canonical float; ``unit`` is the pollutant's own unit
            (``g/km`` for mass pollutants, ``g/kWh`` for the CETESB energy
            metrics, and so on).
        ef_model: The model that produced the factors (e.g.
            ``ef_brazil_cetesb``).
        process: The emission process; hot exhaust unless the caller asked for
            a wear process.
    """

    table: pd.DataFrame
    ef_model: str
    process: Process = "hot_exhaust"

    def __len__(self) -> int:
        """Number of factor rows."""
        return len(self.table)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the factor table."""
        return self.table.copy(deep=True)

    @property
    def units(self) -> dict[str, str]:
        """Pollutant -> unit mapping for the factors in this result."""
        return dict(self.table.groupby("pollutant")["unit"].first())


class MissingEmissionFactorError(ValueError):
    """A requested ``(pollutant, veh_type, model_year)`` has no factor.

    The message lists the offending keys; a silent zero would become an
    emission total three functions later.
    """


class UnsupportedPollutantError(ValueError):
    """The model's table has the pollutant but its unit is not established."""
