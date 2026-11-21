"""Fleet composition schema and validation.

``emission_model()`` consumes fleet data in one of two modes, and the emission
factor models disagree about which columns they need. This module is the single
place that reads, normalises and validates that table, so the EF ports and the
emission pipeline can assume a documented schema.

Two modes:

- **average** — one row per vehicle-characteristics combination, with the share
  of the fleet it represents (``fleet_composition``). Most transport agencies
  publish this shape.
- **per-vehicle** — rows carry a route or shape assignment (``shape_id`` or
  ``route_id``) and a count per combination; the share is computed within each
  assignment group.

In both modes the composition must be a share in ``[0, 1]`` and must sum to 1
within ``tolerance`` per group. The model's required columns and category
vocabularies live in :data:`FLEET_SPECS`; a value outside a vocabulary raises
rather than reaching an EF lookup that would silently return nothing.

Column aliases are normalised on read: ``year`` -> ``model_year`` and
``N``/``total`` -> ``count``. The table is not otherwise rewritten; R column
names are kept so the EF ports read the same words as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

__all__ = [
    "FLEET_SPECS",
    "FleetError",
    "FleetSpec",
    "FleetTable",
    "UnknownFleetModelError",
    "load_fleet",
]

FleetMode = Literal["average", "per_vehicle"]

# Composition must sum to 1 within this tolerance unless the caller passes a
# different one. 1e-6 is far below any real fleet's rounding error (the shipped
# fixtures are exact to ~15 decimals).
_DEFAULT_TOLERANCE = 1e-6

# Columns a run may have to group by; a table with either is per-vehicle mode.
_ASSIGNMENT_COLUMNS: tuple[str, ...] = ("shape_id", "route_id")

# GTFS-adjacent input spellings -> canonical column names. The shipped fixtures
# use both spellings (`usa_det_fleet.txt` has `year` and `N`, `bra_cur_fleet.txt`
# has `year` and `total`), so normalising at read time is what makes one schema
# out of them.
_ALIASES: dict[str, str] = {
    "year": "model_year",
    "N": "count",
    "total": "count",
}


class FleetError(ValueError):
    """A fleet table cannot be used by the emission model.

    Messages name the offending columns, keys or values, because a bare "invalid
    fleet" is the error a caller cannot act on.
    """


class UnknownFleetModelError(FleetError):
    """Raised when ``ef_model`` has no fleet specification."""


@dataclass(frozen=True)
class FleetSpec:
    """What one EF model needs from a fleet table.

    Attributes:
        required: Columns that must be present after alias normalisation.
        categories: Allowed values per categorical column. A column absent from
            this mapping is not checked.
        year_column: The model-year column, or ``None`` when the model has no
            year input.
    """

    required: tuple[str, ...]
    categories: dict[str, frozenset[str]]
    year_column: str | None = "model_year"


_CETESB_VEH = frozenset({"BUS_URBAN_D", "BUS_MICRO_D", "BUS_COACH_D", "BUS_ARTIC_D"})
_EMEP_VEH = frozenset(
    {
        "Ubus Midi <=15 t",
        "Ubus Std 15 - 18 t",
        "Ubus Artic >18 t",
        "Coaches Std <=18 t",
        "Coaches Artic >18 t",
    }
)
_EMEP_EURO = frozenset({"Conventional", "I", "II", "III", "IV", "V", "VI", "EEV"})
_EMEP_FUEL = frozenset({"D", "DHD", "DHE", "CNG", "BD"})
_EMEP_TECH = frozenset({"-", "SCR", "EGR", "DPF+SCR"})
_US_FUEL = frozenset({"D", "Diesel", "G", "Gasoline", "CNG"})
_US_FUEL_ALIASES = {"Diesel": "D", "Gasoline": "G"}

FLEET_SPECS: dict[str, FleetSpec] = {
    # CETESB keys on the Brazilian bus class (`type_name_br`); `veh_type` is the
    # EMEP crosswalk and is optional here.
    "ef_brazil_cetesb": FleetSpec(
        required=("type_name_br", "model_year", "fuel"),
        categories={"type_name_br": _CETESB_VEH, "fuel": frozenset({"D"})},
    ),
    "ef_europe_emep": FleetSpec(
        required=("veh_type", "euro", "fuel", "tech"),
        categories={
            "veh_type": _EMEP_VEH,
            "euro": _EMEP_EURO,
            "fuel": _EMEP_FUEL,
            "tech": _EMEP_TECH,
        },
    ),
    # EMFAC and MOVES key on model year, fuel and (for MOVES) a reference year;
    # the reference year is an emission-model argument, not a fleet column.
    "ef_usa_emfac": FleetSpec(
        required=("model_year", "fuel"),
        categories={"fuel": _US_FUEL},
    ),
    "ef_usa_moves": FleetSpec(
        required=("model_year", "fuel"),
        categories={"fuel": _US_FUEL},
    ),
}


@dataclass(frozen=True)
class FleetTable:
    """A validated fleet table.

    Attributes:
        table: The normalised table, with ``model_year``/``count`` names and a
            ``fleet_composition`` column (derived from counts when absent).
        mode: ``average`` or ``per_vehicle``.
        ef_model: The EF model the requirements were checked against.
        group_columns: Columns whose rows share one composition (empty for
            average mode, the assignment column for per-vehicle mode).
        composition_column: Name of the share column (always
            ``fleet_composition`` today; carried so a caller does not hard-code
            it twice).
    """

    table: pd.DataFrame
    mode: FleetMode
    ef_model: str
    group_columns: tuple[str, ...]
    composition_column: str = "fleet_composition"

    @property
    def n_rows(self) -> int:
        """Number of vehicle-characteristics rows."""
        return len(self.table)


def _normalise_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with aliased column names and stripped strings."""
    renamed = table.rename(
        columns={old: new for old, new in _ALIASES.items() if old in table.columns}
    )
    collisions = [
        (old, new) for old, new in _ALIASES.items() if old in table.columns and new in table.columns
    ]
    if collisions:
        pairs = ", ".join(f"{old!r}/{new!r}" for old, new in collisions)
        raise FleetError(f"fleet table has both an alias and its canonical name: {pairs}")
    string_columns = renamed.select_dtypes(include="object").columns
    renamed[string_columns] = renamed[string_columns].apply(
        lambda column: column.str.strip() if column.dtype == "object" else column
    )
    if "fuel" in renamed.columns:
        renamed["fuel"] = renamed["fuel"].replace(_US_FUEL_ALIASES)
    return renamed


def _resolve_mode(table: pd.DataFrame, requested: str) -> tuple[FleetMode, str | None]:
    """Return the mode and the assignment column (``None`` for average)."""
    present = [column for column in _ASSIGNMENT_COLUMNS if column in table.columns]
    if len(present) > 1:
        raise FleetError(
            "fleet table has more than one assignment column "
            f"({present}); keep `shape_id` or `route_id`, not both"
        )
    detected: FleetMode = "per_vehicle" if present else "average"
    if requested != "auto" and requested != detected:
        raise FleetError(
            f"fleet table looks like {detected!r} mode but {requested!r} was requested"
        )
    return detected, present[0] if present else None


def _check_required(table: pd.DataFrame, spec: FleetSpec, model: str) -> None:
    """Validate required columns and a composition/count source."""
    missing = [column for column in spec.required if column not in table.columns]
    if missing:
        raise FleetError(
            f"fleet table for {model} is missing required column(s) {missing}; "
            f"it has {sorted(table.columns)}"
        )
    if "fleet_composition" not in table.columns and "count" not in table.columns:
        raise FleetError(
            "fleet table needs `fleet_composition` or a count column "
            "(`count`/`N`/`total`) to derive it"
        )


def _check_categories(table: pd.DataFrame, spec: FleetSpec, model: str) -> None:
    """Validate categorical values and reject nulls."""
    for column, allowed in spec.categories.items():
        if column not in table.columns:
            continue
        values = table[column].dropna().astype(str)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise FleetError(
                f"fleet table column {column!r} has unsupported value(s) {unknown}; "
                f"{model} accepts {sorted(allowed)}"
            )
        if table[column].isna().any():
            rows = table.index[table[column].isna()].tolist()[:3]
            raise FleetError(f"fleet table column {column!r} has null value(s) at rows {rows}")


def _check_years(table: pd.DataFrame, spec: FleetSpec, model: str) -> None:
    """Validate integer model years and cast them to int32."""
    if spec.year_column is None or spec.year_column not in table.columns:
        return
    years = pd.to_numeric(table[spec.year_column], errors="coerce")
    non_integer = years.notna() & (years % 1 != 0)
    bad = years.isna() | non_integer | (years < 1900) | (years > 2100)
    if bad.any():
        rows = table.index[bad].tolist()[:3]
        values = table.loc[bad, spec.year_column].tolist()[:3]
        raise FleetError(
            f"fleet table {spec.year_column!r} must be an integer year between "
            f"1900 and 2100 (fractional years are rejected, not truncated); "
            f"invalid value(s) {values} at rows {rows}. The EF model applies its own range."
        )
    table[spec.year_column] = years.astype("int32")


def _composition(table: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    """Add ``fleet_composition``, deriving it from counts when absent."""
    if "fleet_composition" in table.columns:
        composition = pd.to_numeric(table["fleet_composition"], errors="coerce")
        if composition.isna().any():
            rows = table.index[composition.isna()].tolist()[:3]
            raise FleetError(f"`fleet_composition` has non-numeric value(s) at rows {rows}")
    else:
        counts = pd.to_numeric(table["count"], errors="coerce")
        if counts.isna().any() or (counts <= 0).any():
            rows = table.index[counts.isna() | (counts <= 0)].tolist()[:3]
            raise FleetError(f"fleet counts must be positive numbers; invalid rows {rows}")
        if group_columns:
            totals = counts.groupby([table[column] for column in group_columns]).transform("sum")
            composition = counts / totals
        else:
            composition = counts / counts.sum()
    out = table.copy()
    out["fleet_composition"] = composition.astype("float64")
    return out


def _check_composition(
    table: pd.DataFrame, group_columns: tuple[str, ...], tolerance: float
) -> None:
    """Validate share bounds and per-group sums."""
    values = table["fleet_composition"]
    bad = (values < 0) | (values > 1)
    if bad.any():
        rows = table.index[bad].tolist()[:3]
        raise FleetError(
            f"`fleet_composition` must be in [0, 1]; invalid rows {rows} "
            f"({values[bad].tolist()[:3]})"
        )
    if group_columns:
        grouped = table.groupby(list(group_columns), dropna=False)["fleet_composition"].sum()
    else:
        grouped = pd.Series({0: values.sum()})
    off = grouped[(grouped - 1.0).abs() > tolerance]
    if not off.empty:
        examples = [(index, round(float(value), 12)) for index, value in off.items()][:3]
        raise FleetError(
            "fleet composition must sum to 1 per group within "
            f"{tolerance}; offending group(s): {examples}"
        )


def _check_duplicates(table: pd.DataFrame, spec: FleetSpec, group_columns: tuple[str, ...]) -> None:
    """Reject two rows describing the same vehicle characteristics.

    Used in average mode, where two shares for one characteristics tuple are
    ambiguous. Per-vehicle counts are additive and are aggregated instead (see
    :func:`_aggregate_counts`).
    """
    measure_columns = {"fleet_composition", "count"}
    key_columns = [column for column in table.columns if column not in measure_columns]
    duplicated = table.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        keys = table.loc[duplicated, key_columns].drop_duplicates().head(3).to_dict("records")
        raise FleetError(f"fleet table has duplicate key(s) on {key_columns}: {keys}")


def _aggregate_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Sum counts for rows that share a characteristics tuple.

    The shipped Curitiba table has the same ``(shape_id, model_year, euro,
    type_name_br, veh_type, fuel)`` twice with counts 6 and 2 — a raw-input
    artefact, not a composition error. Counts are additive, so they are summed;
    a per-vehicle table that also carries shares is rejected instead, because two
    shares for one key cannot be added.
    """
    measure_columns = {"fleet_composition", "count"}
    key_columns = [column for column in table.columns if column not in measure_columns]
    if "fleet_composition" in table.columns:
        duplicated = table.duplicated(subset=key_columns, keep=False)
        if duplicated.any():
            keys = table.loc[duplicated, key_columns].drop_duplicates().head(3).to_dict("records")
            raise FleetError(
                "per-vehicle fleet table has duplicate key(s) with shares, which "
                f"cannot be aggregated: {keys}"
            )
        return table
    return table.groupby(key_columns, as_index=False, dropna=False)["count"].sum()


def load_fleet(
    source: str | Path | pd.DataFrame,
    *,
    ef_model: str,
    mode: Literal["average", "per_vehicle", "auto"] = "auto",
    tolerance: float = _DEFAULT_TOLERANCE,
) -> FleetTable:
    """Read and validate a fleet table for one EF model.

    Args:
        source: CSV path, or an already-loaded DataFrame.
        ef_model: One of :data:`FLEET_SPECS` (``ef_brazil_cetesb``,
            ``ef_europe_emep``, ``ef_usa_emfac``, ``ef_usa_moves``).
        mode: ``auto`` (default) infers the mode from a ``shape_id``/``route_id``
            column; ``average`` or ``per_vehicle`` asserts it.
        tolerance: Allowed deviation from a composition sum of 1 per group.

    Returns:
        A validated :class:`FleetTable`; the table is a copy, so later mutation
        of the input does not affect it.

    Raises:
        UnknownFleetModelError: If ``ef_model`` has no fleet specification.
        FleetError: For a missing column, unknown category, invalid year
            (including a fractional year), composition outside ``[0, 1]`` or
            not summing to 1, duplicate keys, or a mode that contradicts the
            table.
        TypeError: If ``source`` is not a path or DataFrame.
    """
    if isinstance(source, str | Path):
        table = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        table = source.copy()
    else:
        raise TypeError(f"fleet source must be a path or DataFrame, got {type(source).__name__}")

    if ef_model not in FLEET_SPECS:
        raise UnknownFleetModelError(
            f"unknown ef_model {ef_model!r}; fleet specs exist for {sorted(FLEET_SPECS)}"
        )
    spec = FLEET_SPECS[ef_model]

    table = _normalise_columns(table)
    resolved_mode, assignment = _resolve_mode(table, mode)
    _check_required(table, spec, ef_model)
    _check_categories(table, spec, ef_model)
    _check_years(table, spec, ef_model)

    group_columns: tuple[str, ...] = (assignment,) if assignment else ()
    if resolved_mode == "per_vehicle":
        table = _aggregate_counts(table)
    else:
        _check_duplicates(table, spec, group_columns)
    table = _composition(table, group_columns)
    _check_composition(table, group_columns, tolerance)

    sort_columns = [column for column in (*group_columns, *spec.required) if column in table]
    table = table.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return FleetTable(
        table=table,
        mode=resolved_mode,
        ef_model=ef_model,
        group_columns=group_columns,
    )
