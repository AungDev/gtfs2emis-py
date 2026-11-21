"""CETESB emission factors (Brazil).

The table ships as ``brazil_cetesb.parquet``; each row is a
``(pollutant, model_year)`` pair and each bus class is a numeric column
(``BUS_URBAN_D``, ``BUS_MICRO_D``, ``BUS_COACH_D``, ``BUS_ARTIC_D``). CETESB has
no speed dependence, so this function has no speed argument.

Not every pollutant in the table is ported. ``FC``, ``FS`` and ``KML`` carry no
unit beyond R's blanket ``g/km`` tag, and the table states
nothing else; registering them would invent a conversion. They raise
:class:`UnsupportedPollutantError`, and the 183 nulls in the ``BUS_*`` columns
become :class:`MissingEmissionFactorError` rather than zeros.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import pandas as pd

from gtfs2emis_py.ef.base import (
    EmissionFactors,
    MissingEmissionFactorError,
    UnsupportedPollutantError,
    as_tuple,
)
from gtfs2emis_py.ef.data import load_ef_table
from gtfs2emis_py.pollutants import get_pollutant

__all__ = ["CETESB_VEHICLES", "ef_brazil_cetesb"]

CETESB_VEHICLES: tuple[str, ...] = (
    "BUS_URBAN_D",
    "BUS_MICRO_D",
    "BUS_COACH_D",
    "BUS_ARTIC_D",
)

# Pollutants present in the shipped table whose unit the data does not state.
_UNIT_UNVERIFIED: frozenset[str] = frozenset({"FC", "FS", "KML"})

_MODEL_YEAR_MIN = 1960
_MODEL_YEAR_MAX = 2020


def ef_brazil_cetesb(
    pollutant: str | Sequence[str],
    veh_type: str | Sequence[str],
    model_year: int | Sequence[int],
) -> EmissionFactors:
    """Look up CETESB hot-exhaust emission factors.

    Args:
        pollutant: One pollutant name or a sequence (e.g. ``"CO"``,
            ``["CO", "NOx"]``). Case-insensitive through the pollutant registry.
        veh_type: One bus class or a sequence from :data:`CETESB_VEHICLES`.
        model_year: One model year or a sequence, within 1960-2020.

    Returns:
        An :class:`EmissionFactors` with one row per requested combination,
        sorted by ``pollutant``, ``veh_type`` and ``model_year``.

    Raises:
        UnknownPollutantError: If a name is not in the pollutant registry.
        UnsupportedPollutantError: For ``FC``/``FS``/``KML``, whose unit the
            table does not state.
        MissingEmissionFactorError: When the table has no value for a requested
            combination (including the reference data's nulls).
        ValueError: For an unknown bus class or a model year outside the range.
        TypeError: For an input that is neither a scalar nor a sequence.
    """
    pollutants = tuple(str(value).strip() for value in as_tuple(pollutant, "pollutant"))
    vehicles = tuple(str(value).strip() for value in as_tuple(veh_type, "veh_type"))
    years = tuple(as_tuple(model_year, "model_year"))
    if not pollutants or not vehicles or not years:
        raise ValueError("pollutant, veh_type and model_year must not be empty")

    unknown_vehicles = sorted(set(vehicles) - set(CETESB_VEHICLES))
    if unknown_vehicles:
        raise ValueError(
            f"unsupported CETESB veh_type(s) {unknown_vehicles}; "
            f"expected a subset of {list(CETESB_VEHICLES)}"
        )
    validated_years: list[int] = []
    for year in years:
        if isinstance(year, bool) or not isinstance(year, numbers.Integral):
            raise TypeError(f"model_year must be an int, got {type(year).__name__}")
        year_int = int(year)
        if not _MODEL_YEAR_MIN <= year_int <= _MODEL_YEAR_MAX:
            raise ValueError(
                f"model_year {year_int} is outside the CETESB table's range "
                f"{_MODEL_YEAR_MIN}-{_MODEL_YEAR_MAX}"
            )
        validated_years.append(year_int)
    years = tuple(dict.fromkeys(validated_years))

    unverified = sorted(set(pollutants) & _UNIT_UNVERIFIED)
    if unverified:
        raise UnsupportedPollutantError(
            f"CETESB pollutant(s) {unverified} are not ported: the shipped table "
            "states no unit beyond R's blanket g/km"
        )
    # Resolve names first, so an unknown name fails before any lookup.
    units = {name: get_pollutant(name).unit for name in pollutants}

    database = load_ef_table("brazil_cetesb")
    available = set(database["pollutant"].dropna().astype(str))
    missing_pollutants = sorted(set(pollutants) - available)
    if missing_pollutants:
        raise MissingEmissionFactorError(
            f"CETESB table has no pollutant(s) {missing_pollutants}; available: {sorted(available)}"
        )

    selected = database[
        database["pollutant"].astype(str).isin(pollutants) & database["model_year"].isin(years)
    ]
    long = selected.melt(
        id_vars=["pollutant", "model_year"],
        value_vars=list(vehicles),
        var_name="veh_type",
        value_name="ef",
    )
    grid = pd.MultiIndex.from_product(
        [pollutants, vehicles, years],
        names=["pollutant", "veh_type", "model_year"],
    ).to_frame(index=False)
    merged = grid.merge(
        long,
        on=["pollutant", "veh_type", "model_year"],
        how="left",
        validate="one_to_one",
    )
    missing = merged[merged["ef"].isna()]
    if not missing.empty:
        keys = missing[["pollutant", "veh_type", "model_year"]].head(3).to_dict("records")
        raise MissingEmissionFactorError(
            f"CETESB has no emission factor for {len(missing)} requested "
            f"combination(s), e.g. {keys}"
        )
    merged["unit"] = merged["pollutant"].map(units).astype("string")
    merged["ef"] = merged["ef"].astype("float64")
    merged["model_year"] = merged["model_year"].astype("int32")
    merged = merged.sort_values(["pollutant", "veh_type", "model_year"], kind="stable").reset_index(
        drop=True
    )
    return EmissionFactors(
        table=merged[["pollutant", "veh_type", "model_year", "ef", "unit"]],
        ef_model="ef_brazil_cetesb",
        process="hot_exhaust",
    )
