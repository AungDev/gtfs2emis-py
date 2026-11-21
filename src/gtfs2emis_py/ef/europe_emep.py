"""EMEP/EEA emission factors (Europe).

Port of `ef_europe_emep()` from gtfs2emis v0.1.2 (R). The reference selects a
row per ``(Pol, Fuel, Segment, Technology, Euro, Slope, Load)``, clamps the
speed to that row's ``Vmin``/``Vmax``, evaluates one of twenty ``Function_ID``
polynomial families, and multiplies by the row's ``k`` and the caller's
``fcorr``. Speed-dependent rows are joined on slope and load; rows tabulated at a
single slope/load are reused with a warning.

Two reference behaviours are reproduced rather than "fixed":

- a missing cell aborts the whole call (no silent zero);
- a key with two rows from different editions (the shipped table has 525 such
  keys) also aborts, because the reference's join has no edition preference.
  Both cases raise :class:`MissingEmissionFactorError`; the message says which.

``SPN23 [#/kWh]`` is deliberately rejected: R supports it, but its count-per-
energy unit is not mass-per-distance and there is no documented conversion.
``EC`` is returned in ``MJ/km`` rather than ``g/km``.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from gtfs2emis_py.ef.base import (
    EmissionFactors,
    MissingEmissionFactorError,
    as_tuple,
)
from gtfs2emis_py.ef.data import load_ef_table
from gtfs2emis_py.pollutants import pollutant_unit

__all__ = ["EmepFallbackWarning", "ef_europe_emep"]

_SLOPES = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
_LOADS = (0.0, 0.5, 1.0)
_NO_TECH_EURO = frozenset({"Conventional", "I", "II", "III"})
_FC_CO2 = frozenset({"FC", "CO2"})
_EXCLUDED_POLLUTANTS = frozenset({"SPN23 [#/kWh]"})


class EmepFallbackWarning(UserWarning):
    """The reference would also have printed a message (technology/slope)."""


def _as_strings(value: object, name: str) -> list[str]:
    """Coerce and validate one or more non-empty strings."""
    values = [str(item) for item in as_tuple(value, name)]
    if not values or any(item == "" for item in values):
        raise ValueError(f"{name} must not be empty")
    return values


def _as_floats(value: object, name: str) -> list[float]:
    """Coerce and validate finite numeric values."""
    raw = as_tuple(value, name)
    result: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise TypeError(f"{name} must be numeric, got {type(item).__name__}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite, got {item!r}")
        result.append(number)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _recycle(values: list[str], n: int, name: str) -> list[str]:
    """Broadcast a one-element list to n items or require length n."""
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(
            f"{name} must have length 1 or {n}, the number of vehicles; it has length {len(values)}"
        )
    return values


def _resolve_tech(euro: str, tech: str, pollutant: str) -> str:
    """Apply the reference's five sequential technology fallbacks."""
    if euro in _NO_TECH_EURO and tech != "-":
        warnings.warn(
            f"{pollutant} emission factor not found for technology {tech} and "
            f"Euro {euro}; the reference uses '-'",
            EmepFallbackWarning,
            stacklevel=3,
        )
        tech = "-"
    if euro in {"V", "VI"} and pollutant in _FC_CO2 and tech == "-":
        warnings.warn(
            f"{pollutant} emission factor not found for technology '-' and "
            f"Euro {euro}; the reference uses 'SCR'",
            EmepFallbackWarning,
            stacklevel=3,
        )
        tech = "SCR"
    if euro not in {"V", "VI"} and pollutant in _FC_CO2 and tech != "-":
        warnings.warn(
            f"{pollutant} emission factor not found for technology {tech} and "
            f"Euro {euro}; the reference uses '-'",
            EmepFallbackWarning,
            stacklevel=3,
        )
        tech = "-"
    if euro in {"V", "VI"} and pollutant in _FC_CO2 and tech in {"-", "DPF+SCR"}:
        tech = "SCR"
    if euro == "VI" and pollutant not in _FC_CO2 and tech != "DPF+SCR":
        warnings.warn(
            f"{pollutant} emission factor not found for technology {tech} and "
            f"Euro VI; the reference uses 'DPF+SCR'",
            EmepFallbackWarning,
            stacklevel=3,
        )
        tech = "DPF+SCR"
    return tech


_FORMULAS: dict[str, Callable[[pd.DataFrame, np.ndarray, float], np.ndarray]] = {}


def _formula(*names: str) -> Callable:
    """Decorator registering a formula under one or more Function_ID names."""

    def register(fn: Callable) -> Callable:
        """Register the decorated formula for each Function_ID alias."""
        for name in names:
            _FORMULAS[name] = fn
        return fn

    return register


@_formula("1", "2019_1")
def _f_2019(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP 2019 family 1 formula."""
    return (
        (c["Alpha"] * v**2 + c["Beta"] * v + c["Gamma"] + c["Delta"] / v)
        / (c["Epsilon"] * v**2 + c["Zita"] * v + c["Hta"])
        * (1 - c["RF"])
        * c["k"]
        * fcorr
    )


@_formula("2013_3", "2007_2")
def _f_exp_power(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP exponential-power formula."""
    return (c["Alpha"] * (c["Beta"] ** v) * (v ** c["Gamma"])) * c["k"] * fcorr


@_formula("2013_12")
def _f_logistic_a(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP logistic formula (variant A)."""
    return (c["Alpha"] / (1 + c["Beta"] * np.exp(-c["Gamma"] * v))) * c["k"] * fcorr


@_formula("2013_9")
def _f_inverse_power(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP inverse-power formula."""
    return (1 / (c["Alpha"] + c["Beta"] * (v ** c["Gamma"]))) * c["k"] * fcorr


@_formula("2013_11")
def _f_exp_power_delta(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP exponential-power formula with a Delta exponent."""
    return (c["Alpha"] - c["Beta"] * np.exp(-c["Gamma"] * (v ** c["Delta"]))) * c["k"] * fcorr


@_formula("2013_4", "2007_3")
def _f_power_sum(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP sum-of-powers formula."""
    return ((c["Alpha"] * (v ** c["Beta"])) + (c["Gamma"] * (v ** c["Delta"]))) * c["k"] * fcorr


@_formula("2013_16", "2007_11")
def _f_exp_log(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP exponential-log formula."""
    return np.exp(c["Alpha"] + c["Beta"] / v + c["Gamma"] * np.log(v)) * c["k"] * fcorr


@_formula("2013_10")
def _f_inverse_linear(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP inverse-linear formula."""
    return (1 / (c["Alpha"] + c["Beta"] * v)) * c["k"] * fcorr


@_formula("2013_1")
def _f_cubic(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP cubic formula."""
    return (c["Alpha"] * v**3 + c["Beta"] * v**2 + c["Gamma"] * v + c["Delta"]) * c["k"] * fcorr


@_formula("2007_1")
def _f_two_exp(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP two-exponential formula."""
    return (
        (c["Epsilon"] + c["Alpha"] * np.exp(-c["Beta"] * v) + c["Gamma"] * np.exp(-c["Delta"] * v))
        * c["k"]
        * fcorr
    )


@_formula("2007_4")
def _f_logistic_b(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP logistic formula (variant B)."""
    return (
        (
            c["Alpha"]
            + c["Beta"] / (1 + np.exp(-c["Gamma"] + c["Delta"] * np.log(v) + c["Epsilon"] * v))
        )
        * c["k"]
        * fcorr
    )


@_formula("2007_5")
def _f_saturating(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP saturating-growth formula."""
    return (
        (
            c["Alpha"]
            + c["Beta"] * v
            + ((c["Gamma"] - c["Beta"]) * (1 - np.exp(-c["Delta"] * v)) / c["Delta"])
        )
        * c["k"]
        * fcorr
    )


@_formula("2007_6")
def _f_exp_growth(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP exponential-growth formula."""
    return (c["Gamma"] + c["Alpha"] * np.exp(c["Beta"] * v)) * c["k"] * fcorr


@_formula("2007_7")
def _f_cubic_2007(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the 2007 EMEP cubic formula."""
    return (c["Alpha"] * v**3 + c["Beta"] * v**2 + c["Gamma"] * v + c["Delta"]) * c["k"] * fcorr


@_formula("2007_9")
def _f_inverse_quadratic(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP inverse-quadratic formula."""
    return 1 / (c["Gamma"] * v**2 + c["Beta"] * v + c["Alpha"]) * c["k"] * fcorr


@_formula("2007_10")
def _f_logistic_2007(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the 2007 EMEP logistic formula."""
    return (c["Alpha"] / (1 + c["Beta"] * np.exp(-c["Gamma"] * v))) * c["k"] * fcorr


@_formula("2007_8")
def _f_exp_decay(c: pd.DataFrame, v: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the EMEP exponential-decay formula."""
    return (c["Gamma"] + c["Alpha"] * np.exp(-c["Beta"] * v)) * c["k"] * fcorr


def _evaluate(c: pd.DataFrame, speed: np.ndarray, fcorr: float) -> np.ndarray:
    """Evaluate the rows' Function_ID families for their clamped speeds."""
    families = c["Function_ID"].to_numpy()
    out = np.empty(len(c), dtype=float)
    for family in pd.unique(families):
        mask = families == family
        fn = _FORMULAS.get(str(family))
        if fn is None:
            raise MissingEmissionFactorError(
                f"unknown EMEP Function_ID {family!r}; the port implements {sorted(_FORMULAS)}"
            )
        rows = c.loc[mask]
        out[mask] = fn(rows, speed[mask], fcorr)
    return out


def ef_europe_emep(
    speed: float | Sequence[float],
    veh_type: str | Sequence[str],
    euro: str | Sequence[str],
    pollutant: str | Sequence[str],
    fuel: str | Sequence[str] = "D",
    tech: str | Sequence[str] = "SCR",
    slope: float | Sequence[float] = 0.0,
    load: float | Sequence[float] = 0.5,
    fcorr: float | Sequence[float] = 1.0,
    on_missing_slope: str = "error",
) -> EmissionFactors:
    """Look up EMEP/EEA hot-exhaust emission factors.

    Args:
        speed: Speed in km/h, 1-130, rounded like R's ``round()``.
        veh_type: One of the five EMEP bus segments.
        euro: Euro standard, Conventional through VI, or EEV/PRE.
        pollutant: EMEP pollutant; ``SPN23 [#/kWh]`` is deliberately rejected
            even though R supports it.
        fuel: D, DHD, DHE, CNG or BD.
        tech: SCR, EGR, DPF+SCR or ``-``. Fallbacks mirror the reference and warn.
        slope: One of -0.06 ... 0.06, length 1 or ``len(speed)``.
        load: One of 0, 0.5, 1, length 1 or ``len(speed)``.
        fcorr: Fuel-composition correction, non-negative, length 1 or one per
            vehicle.
        on_missing_slope: ``"error"`` (R parity) or ``"nearest"``. A single-row
            family with no requested slope/load raises by default; ``"nearest"``
            reuses its only row with an :class:`EmepFallbackWarning`.

    Returns:
        An :class:`EmissionFactors` whose table has one row per
        ``(pollutant, vehicle, speed)`` with the matching slope and load.
        ``EC`` is in ``MJ/km``; every other pollutant in ``g/km``.

    Raises:
        MissingEmissionFactorError: For a missing or ambiguous table cell.
        ValueError: For an invalid vocabulary value, grid value or length.
        TypeError: For a non-numeric or non-string input.
    """
    speeds = _as_floats(speed, "speed")
    if any(not 1 <= value <= 130 for value in speeds):
        raise ValueError("speed must be between 1 and 130 km/h")
    speeds = [int(round(value)) for value in speeds]
    speed_array = np.asarray(speeds, dtype=float)

    vehicles = _as_strings(veh_type, "veh_type")
    euros = _as_strings(euro, "euro")
    fuels = _as_strings(fuel, "fuel")
    techs = _as_strings(tech, "tech")
    pollutants = _as_strings(pollutant, "pollutant")
    n_veh = max(len(vehicles), len(euros), len(fuels), len(techs))
    vehicles = _recycle(vehicles, n_veh, "veh_type")
    euros = _recycle(euros, n_veh, "euro")
    fuels = _recycle(fuels, n_veh, "fuel")
    techs = _recycle(techs, n_veh, "tech")
    fcorrs = _as_floats(fcorr, "fcorr")
    if any(value < 0 for value in fcorrs):
        raise ValueError("fcorr must be non-negative")
    if len(fcorrs) == 1:
        fcorrs = fcorrs * n_veh
    elif len(fcorrs) != n_veh:
        raise ValueError(
            f"fcorr must have length 1 or {n_veh}, the number of vehicles; "
            f"it has length {len(fcorrs)}"
        )

    slope_values = [round(value, 2) for value in _as_floats(slope, "slope")]
    load_values = [round(value, 2) for value in _as_floats(load, "load")]
    if len(slope_values) == 1:
        slope_values *= len(speeds)
    elif len(slope_values) != len(speeds):
        raise ValueError("slope and speed must have the same length")
    if len(load_values) == 1:
        load_values *= len(speeds)
    elif len(load_values) != len(speeds):
        raise ValueError("load and speed must have the same length")
    for value in slope_values:
        if not any(math.isclose(value, allowed, abs_tol=1e-9) for allowed in _SLOPES):
            raise ValueError(f"slope {value} is not one of {list(_SLOPES)}")
    for value in load_values:
        if not any(math.isclose(value, allowed, abs_tol=1e-9) for allowed in _LOADS):
            raise ValueError(f"load {value} is not one of {list(_LOADS)}")

    if on_missing_slope not in {"error", "nearest"}:
        raise ValueError("on_missing_slope must be 'error' or 'nearest'")
    table = load_ef_table("europe_emep")
    allowed_pollutants = set(table["Pol"].unique()) - _EXCLUDED_POLLUTANTS
    allowed_vehicles = set(table["Segment"].unique())
    allowed_euros = set(table["Euro"].unique())
    allowed_fuels = set(table["Fuel"].unique())
    allowed_techs = set(table["Technology"].unique())
    _require(pollutants, allowed_pollutants, "pollutant")
    _require(vehicles, allowed_vehicles, "veh_type")
    _require(euros, allowed_euros, "euro")
    _require(fuels, allowed_fuels, "fuel")
    _require(techs, allowed_techs, "tech")

    key_frame = pd.DataFrame({"Slope": slope_values, "Load": load_values, "speed": speed_array})
    unique = key_frame.drop_duplicates(ignore_index=True)
    mapped = key_frame.merge(
        unique.reset_index().rename(columns={"index": "_u"}),
        on=["Slope", "Load", "speed"],
        how="left",
        validate="many_to_one",
    )
    group = mapped["_u"].to_numpy(dtype=int)

    frames: list[pd.DataFrame] = []
    for pollutant_name in pollutants:
        matrix = np.empty((len(speeds), n_veh), dtype=float)
        for vehicle_index in range(n_veh):
            tech_name = _resolve_tech(euros[vehicle_index], techs[vehicle_index], pollutant_name)
            matrix[:, vehicle_index] = _lookup_one(
                table,
                pollutant=pollutant_name,
                fuel=fuels[vehicle_index],
                segment=vehicles[vehicle_index],
                tech=tech_name,
                euro=euros[vehicle_index],
                unique=unique,
                group=group,
                fcorr=fcorrs[vehicle_index],
                on_missing_slope=on_missing_slope,
            )
        unit = pollutant_unit(pollutant_name, model="ef_europe_emep")
        speed_column = np.repeat(speed_array, n_veh)
        vehicle_index_column = np.tile(np.arange(n_veh), len(speeds))
        frames.append(
            pd.DataFrame(
                {
                    "pollutant": pollutant_name,
                    "vehicle_index": vehicle_index_column,
                    "veh_type": np.asarray(vehicles)[vehicle_index_column],
                    "euro": np.asarray(euros)[vehicle_index_column],
                    "fuel": np.asarray(fuels)[vehicle_index_column],
                    "tech": np.asarray(techs)[vehicle_index_column],
                    "slope": np.repeat(np.asarray(slope_values), n_veh),
                    "load": np.repeat(np.asarray(load_values), n_veh),
                    "speed_kmh": speed_column,
                    "ef": matrix.reshape(-1),
                    "unit": unit,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return EmissionFactors(table=result, ef_model="ef_europe_emep", process="hot_exhaust")


def _require(values: Sequence[str], allowed: set[str], name: str) -> None:
    """Raise a named ValueError when a requested value is absent."""
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported EMEP {name}(s) {unknown}; allowed: {sorted(allowed)}")


def _lookup_one(
    table: pd.DataFrame,
    *,
    pollutant: str,
    fuel: str,
    segment: str,
    tech: str,
    euro: str,
    unique: pd.DataFrame,
    group: np.ndarray,
    fcorr: float,
    on_missing_slope: str,
) -> np.ndarray:
    """Return the single matching factor row or raise/fallback per policy."""
    rows = table[
        (table["Pol"] == pollutant)
        & (table["Fuel"] == fuel)
        & (table["Segment"] == segment)
        & (table["Technology"] == tech)
        & (table["Euro"] == euro)
    ]
    keys = f"Pol={pollutant}, Fuel={fuel}, Segment={segment}, Technology={tech}, Euro={euro}"
    if rows.empty:
        raise MissingEmissionFactorError(f"no EMEP emission factor for {keys}")
    if len(rows) == 1:
        row = rows.iloc[0]
        if np.any(unique["Slope"].to_numpy() != row["Slope"]) or np.any(
            unique["Load"].to_numpy() != row["Load"]
        ):
            if on_missing_slope == "error":
                raise MissingEmissionFactorError(
                    f"{pollutant} factors for {segment} are tabulated only at "
                    f"slope {row['Slope']} and load {row['Load']}; pass "
                    "on_missing_slope='nearest' to reuse that row"
                )
            warnings.warn(
                f"{pollutant} factors for {segment} are tabulated only at "
                f"slope {row['Slope']} and load {row['Load']}; that row is used "
                "for every requested slope and load",
                EmepFallbackWarning,
                stacklevel=3,
            )
        selected = rows.iloc[np.zeros(len(unique), dtype=int)].reset_index(drop=True)
        selected["speed"] = unique["speed"].to_numpy(dtype=float)
    else:
        if rows.duplicated(["Slope", "Load"]).any():
            families = sorted(rows["Function_ID"].astype(str).unique())
            raise MissingEmissionFactorError(
                f"ambiguous EMEP emission factor for {keys}: the table has "
                f"{len(rows)} rows for the same slope/load ({', '.join(families)}); "
                "the reference aborts here because it has no edition preference"
            )
        selected = unique.merge(
            rows,
            on=["Slope", "Load"],
            how="left",
            validate="many_to_one",
        )
        if len(selected) != len(unique) or selected["Function_ID"].isna().any():
            missing = selected[selected["Function_ID"].isna()][["Slope", "Load"]]
            raise MissingEmissionFactorError(
                f"no EMEP emission factor for {keys} at "
                f"{missing.drop_duplicates().to_dict('records')}"
            )
    clamped = np.clip(
        selected["speed"].to_numpy(dtype=float),
        selected["Vmin"].to_numpy(dtype=float),
        selected["Vmax"].to_numpy(dtype=float),
    )
    out = _evaluate(selected, clamped, fcorr)
    return out[group]
