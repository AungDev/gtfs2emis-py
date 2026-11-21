"""EMFAC2017 emission factors (California).

Port of `ef_usa_emfac()` from gtfs2emis v0.1.2. EMFAC is **binned**, not
interpolated: a speed belongs to the first bin whose upper bound is >= the
speed, with the bound itself in the lower bin (speed 16.0 is in ``9-16``;
16.1 is in ``17-24``). That is R's
``findInterval(speed, upper_limits, left.open = TRUE)``.

The shipped table covers model years 2005-2019 and reference years
2010-2017/2019/2020, even though the reference validates 1982-2020 and
2010-2020. A year inside the documented range but absent from the table is a
:class:`MissingEmissionFactorError`, not a silent fallback.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np
import pandas as pd

from gtfs2emis_py.ef.base import (
    EmissionFactors,
    MissingEmissionFactorError,
    as_tuple,
)
from gtfs2emis_py.ef.data import load_ef_table
from gtfs2emis_py.pollutants import pollutant_unit

__all__ = ["ef_usa_emfac"]

_VEH_TYPE = "BUS_URBAN_D"


def ef_usa_emfac(
    pollutant: str | Sequence[str],
    reference_year: int = 2020,
    fuel: str | Sequence[str] = "D",
    *,
    model_year: int | Sequence[int],
    speed: float | Sequence[float],
) -> EmissionFactors:
    """Look up EMFAC2017 hot-exhaust emission factors.

    Args:
        pollutant: CH4, CO, CO2, N2O, NOx, PM10, PM25, ROG, SOX or TOG.
        reference_year: Calendar year, 2010-2020, scalar.
        fuel: D, CNG or G; length 1 or one per model year.
        model_year: Model year, 1982-2020; length 1 or one per fuel.
        speed: Speed in km/h, 1-113; each value selects a bin.

    Returns:
        An :class:`EmissionFactors` with one row per
        ``(pollutant, model_year, fuel, speed)`` and the selected bin bounds.

    Raises:
        MissingEmissionFactorError: For an absent table cell or bin.
        ValueError: For an out-of-range year or speed, or a bad vocabulary.
        TypeError: For a non-numeric or non-string input.
    """
    pollutants = [str(item) for item in as_tuple(pollutant, "pollutant")]
    fuels = [str(item) for item in as_tuple(fuel, "fuel")]
    if not pollutants or not fuels:
        raise ValueError("pollutant and fuel must not be empty")
    if isinstance(reference_year, bool) or not isinstance(reference_year, numbers.Integral):
        raise TypeError("reference_year must be an int")
    reference = int(reference_year)
    if not 2010 <= reference <= 2020:
        raise ValueError(f"reference_year {reference} is outside 2010-2020")

    raw_years = as_tuple(model_year, "model_year")
    years: list[int] = []
    for value in raw_years:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"model_year must be an int, got {type(value).__name__}")
        year = int(value)
        if not 1982 <= year <= 2020:
            raise ValueError(f"model_year {year} is outside 1982-2020")
        years.append(year)
    if not years:
        raise ValueError("model_year must not be empty")

    raw_speeds = as_tuple(speed, "speed")
    speeds: list[float] = []
    for value in raw_speeds:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"speed must be numeric, got {type(value).__name__}")
        number = float(value)
        if not 1 <= number <= 113:
            raise ValueError(f"speed {number} is outside 1-113 km/h")
        speeds.append(number)
    if not speeds:
        raise ValueError("speed must not be empty")

    n_vehicles = max(len(years), len(fuels))
    if len(years) == 1:
        years = years * n_vehicles
    elif len(years) != n_vehicles:
        raise ValueError(
            f"model_year must have length 1 or {n_vehicles}, the number of "
            f"vehicles; it has length {len(years)}"
        )
    if len(fuels) == 1:
        fuels = fuels * n_vehicles
    elif len(fuels) != n_vehicles:
        raise ValueError(
            f"fuel must have length 1 or {n_vehicles}, the number of vehicles; "
            f"it has length {len(fuels)}"
        )

    table = load_ef_table("usa_emfac")
    _require(pollutants, set(table["pollutant"].unique()), "pollutant")
    _require(fuels, set(table["fuel"].unique()), "fuel")

    prefiltered = table[
        (table["reference_year"] == reference)
        & (table["pollutant"].isin(pollutants))
        & (table["fuel"].isin(fuels))
        & (table["model_year"].isin(years))
    ]
    if prefiltered.empty:
        raise MissingEmissionFactorError(
            f"no EMFAC factor for reference_year={reference}, "
            f"model_year={sorted(set(years))}, fuel={sorted(set(fuels))}, "
            f"pollutant={pollutants}"
        )
    limits = np.sort(prefiltered["upper_speed_interval"].unique().astype(float))
    bins = np.searchsorted(limits, np.asarray(speeds, dtype=float), side="left")
    if np.any(bins >= len(limits)):
        raise MissingEmissionFactorError(
            f"EMFAC bins end at {limits[-1]} km/h; speed(s) "
            f"{[s for s, b in zip(speeds, bins, strict=True) if b >= len(limits)]} "
            "have no bin"
        )

    frames: list[pd.DataFrame] = []
    for pollutant_name in pollutants:
        for index in range(n_vehicles):
            year = years[index]
            fuel_name = fuels[index]
            rows = table[
                (table["reference_year"] == reference)
                & (table["pollutant"] == pollutant_name)
                & (table["model_year"] == year)
                & (table["fuel"] == fuel_name)
            ].sort_values("upper_speed_interval")
            if rows.empty:
                raise MissingEmissionFactorError(
                    f"no EMFAC factor for pollutant={pollutant_name}, "
                    f"model_year={year}, fuel={fuel_name}, "
                    f"reference_year={reference}"
                )
            row_limits = rows["upper_speed_interval"].to_numpy(dtype=float)
            if not np.array_equal(row_limits, limits):
                raise MissingEmissionFactorError(
                    f"EMFAC bins for pollutant={pollutant_name}, model_year={year}, "
                    f"fuel={fuel_name} are {row_limits.tolist()}, expected "
                    f"{limits.tolist()}"
                )
            frames.append(
                pd.DataFrame(
                    {
                        "pollutant": pollutant_name,
                        "veh_type": _VEH_TYPE,
                        "model_year": year,
                        "fuel": fuel_name,
                        "speed_kmh": np.asarray(speeds, dtype=float),
                        "bin_lower_kmh": rows["lower_speed_interval"].to_numpy(dtype=float)[bins],
                        "bin_upper_kmh": row_limits[bins],
                        "ef": rows["ef"].to_numpy(dtype=float)[bins],
                        "unit": pollutant_unit(pollutant_name),
                    }
                )
            )
    return EmissionFactors(
        table=pd.concat(frames, ignore_index=True),
        ef_model="ef_usa_emfac",
        process="hot_exhaust",
    )


def _require(values: Sequence[str], allowed: set[str], name: str) -> None:
    """Raise a named ValueError when a requested value is absent."""
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported EMFAC {name}(s) {unknown}; allowed: {sorted(allowed)}")
