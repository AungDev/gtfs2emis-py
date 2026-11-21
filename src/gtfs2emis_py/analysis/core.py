"""Analysis helpers on EmissionResult.

`emis_summary()` rolls emissions up by pollutant/process, time or vehicle
variables; `emis_grid()` allocates segment emissions to a polygon grid by the
share of each segment's length that falls inside a cell (conserving mass).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

__all__ = ["emis_grid", "emis_summary"]

_TIME_SECONDS = {"day": 86400, "hour": 3600, "minute": 60}
_BY = {"pollutant", "time", "vehicle"}
_EMI_VEHICLE_VARS = {"vehicle_type", "fleet_row"}


def _as_list(value: object, name: str) -> list[str]:
    """Coerce a string or sequence into a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence | np.ndarray | pd.Series):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a string or a sequence")


def emis_summary(
    result: Any,
    *,
    by: str | Sequence[str] = "pollutant",
    veh_vars: str | Sequence[str] = (),
    segment_vars: str | Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate emissions by pollutant, time and/or vehicle variables.

    Args:
        result: An :class:`EmissionResult`.
        by: Any of ``pollutant``, ``time`` and ``vehicle``. Time adds
            ``hour_of_day = (service_day_seconds // 3600) % 24`` (R parity) and
            ``service_hour = service_day_seconds // 3600`` (never wrapped).
        veh_vars: Extra vehicle columns from the emission table
            (``vehicle_type``, ``fleet_row``).
        segment_vars: Extra transport columns joined by ``segment_id``.
            Columns already present in the emission table (``shape_id``,
            ``trip_id``, ``stop_sequence``) are used directly; the service-day
            column is joined only when ``by`` includes ``time``.
    Returns:
        A frame with the selected dimensions, ``value`` in grams, and ``unit``
        when one pollutant has a single unit.

    Raises:
        ValueError: For an unknown ``by`` value or variable, or a missing
            transport time column.
    """
    emi = getattr(result, "emi", None)
    if not isinstance(emi, pd.DataFrame):
        raise TypeError("result must expose a pandas DataFrame in .emi")
    tp_model = getattr(result, "tp_model", None)
    table = getattr(tp_model, "table", tp_model)
    if not isinstance(table, gpd.GeoDataFrame):
        raise TypeError("result.tp_model must be a TransportModel or a GeoDataFrame")

    by_values = _as_list(by, "by")
    unknown_by = sorted(set(by_values) - _BY)
    if unknown_by:
        raise ValueError(f"unknown by value(s) {unknown_by}; accepted: {sorted(_BY)}")
    vehicle_vars = _as_list(veh_vars, "veh_vars")
    segment_vars_list = _as_list(segment_vars, "segment_vars")
    unknown_vehicle = sorted(set(vehicle_vars) - _EMI_VEHICLE_VARS)
    if unknown_vehicle:
        raise ValueError(f"unknown veh_vars {unknown_vehicle}")
    available_segments = set(table.columns)
    unknown_segments = sorted(set(segment_vars_list) - available_segments)
    if unknown_segments:
        raise ValueError(f"unknown segment_vars {unknown_segments}")

    dimensions = ["pollutant", "process"]
    if "vehicle" in by_values:
        dimensions.append("vehicle_type")
    if "time" in by_values:
        if "from_service_day_seconds" not in table.columns:
            raise ValueError("transport table has no from_service_day_seconds")
        dimensions.extend(["hour_of_day", "service_hour"])
    dimensions.extend(vehicle_vars)
    dimensions.extend(segment_vars_list)
    dimensions = list(dict.fromkeys(dimensions))

    # Only join columns that the emission table does not already carry. pandas
    # would suffix an overlapping name (_x/_y) and the later groupby would
    # still look for the unsuffixed name, which is the cause of R3-F23.
    join_columns = [*segment_vars_list]
    if "time" in by_values:
        join_columns.insert(0, "from_service_day_seconds")
    join_columns = list(dict.fromkeys(join_columns))
    missing = sorted(set(join_columns) - set(table.columns))
    if missing:
        raise ValueError(f"transport table is missing column(s) {missing}")
    missing_ids = sorted(set(emi["segment_id"]) - set(table.index))
    if missing_ids:
        raise ValueError(
            f"some segment_id values are not present in the transport table: {missing_ids[:3]}"
        )
    needed_join = [column for column in join_columns if column not in emi.columns]
    if needed_join:
        joined = emi.merge(
            table[needed_join],
            left_on="segment_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
    else:
        joined = emi.copy()
    if (
        "from_service_day_seconds" in joined.columns
        and joined["from_service_day_seconds"].isna().any()
    ):
        raise ValueError("some segment_id values are not present in the transport table")

    if "time" in by_values:
        seconds = joined["from_service_day_seconds"].astype("int64")
        joined["hour_of_day"] = (seconds // 3600) % 24
        joined["service_hour"] = seconds // 3600

    grouped = joined.groupby(dimensions, as_index=False, sort=True)["value"].sum()
    units = joined.groupby("pollutant", sort=True)["unit"].first()
    grouped["unit"] = grouped["pollutant"].map(units)
    return grouped.sort_values(dimensions, kind="stable").reset_index(drop=True)


def emis_grid(
    result: Any,
    grid: gpd.GeoDataFrame,
    *,
    time_resolution: Literal["day", "hour", "minute"] | None = None,
    aggregate: bool = True,
    on_uncovered: Literal["raise", "warn", "drop"] = "raise",
) -> gpd.GeoDataFrame:
    """Allocate segment emissions to a polygon grid by intersected length.

    Lengths are planar EPSG:4326 degree lengths for both the segment and each
    overlap; the ratio conserves mass but is not a geodesic allocation. Project
    the inputs to a metric CRS first if that distinction matters.


    Args:
        result: An :class:`EmissionResult`.
        grid: Polygon GeoDataFrame with a CRS.
        time_resolution: Optional bucket from the segment's
            ``from_service_day_seconds``: ``day``, ``hour`` or ``minute``.
        aggregate: Sum by grid cell and pollutant/process when True; return one
            row per segment/cell otherwise.
        on_uncovered: What to do when a segment is not fully covered by the
            grid. ``"raise"`` (default) preserves the conservation contract and
            reports the uncovered ratio; ``"warn"`` emits a ``UserWarning`` with
            the same ratio; ``"drop"`` explicitly accepts the loss.

    Returns:
        A GeoDataFrame with ``grid_id``, ``pollutant``, ``process``, optional
        ``time_bucket`` and ``value``. With ``aggregate=False`` it also carries
        ``segment_id`` and no geometry.

    Raises:
        TypeError: For a non-GeoDataFrame grid.
        ValueError: For a missing CRS, a CRS mismatch, an empty/invalid grid,
            a segment without a geometry, or uncovered mass under
            ``on_uncovered="raise"``.
    """
    if not isinstance(grid, gpd.GeoDataFrame):
        raise TypeError("grid must be a GeoDataFrame")
    if grid.crs is None:
        raise ValueError("grid has no CRS")
    if grid.empty:
        raise ValueError("grid is empty")
    if not grid.geometry.is_valid.all() or grid.geometry.is_empty.any():
        raise ValueError("grid contains invalid or empty geometries")
    if not grid.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("grid geometries must be Polygon or MultiPolygon")
    if time_resolution is not None and time_resolution not in _TIME_SECONDS:
        raise ValueError(f"time_resolution must be one of {sorted(_TIME_SECONDS)} or None")
    if on_uncovered not in {"raise", "warn", "drop"}:
        raise ValueError("on_uncovered must be 'raise', 'warn' or 'drop'")

    emi = getattr(result, "emi", None)
    tp_model = getattr(result, "tp_model", None)
    table = getattr(tp_model, "table", tp_model)
    if not isinstance(emi, pd.DataFrame) or not isinstance(table, gpd.GeoDataFrame):
        raise TypeError("result must expose .emi and a GeoDataFrame .tp_model")
    if table.crs is None:
        raise ValueError("transport table has no CRS")
    if table.crs != grid.crs:
        raise ValueError(f"CRS mismatch: transport is {table.crs}, grid is {grid.crs}")
    required = {"from_service_day_seconds", "geometry"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"transport table is missing column(s) {missing}")

    per_segment = emi.groupby(["segment_id", "pollutant", "process"], as_index=False, sort=True)[
        "value"
    ].sum()
    frame = grid.reset_index(drop=True).copy()
    frame["grid_id"] = np.arange(len(frame), dtype=int)
    sindex = frame.sindex
    allocations: list[dict[str, Any]] = []
    for row in per_segment.itertuples(index=False):
        if row.segment_id not in table.index:
            raise ValueError(f"segment_id {row.segment_id} is not in the transport table")
        geometry = table.at[row.segment_id, "geometry"]
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            raise ValueError(f"segment_id {row.segment_id} has no geometry")
        segment_length = float(geometry.length)
        if segment_length <= 0:
            continue
        candidates = sindex.query(geometry, predicate="intersects")
        covered = 0.0
        for position in candidates:
            cell = frame.geometry.iloc[int(position)]
            overlap = geometry.intersection(cell).length
            if overlap <= 0:
                continue
            covered += float(overlap)
            allocation = {
                "grid_id": int(position),
                "segment_id": row.segment_id,
                "pollutant": row.pollutant,
                "process": row.process,
                "value": float(row.value) * float(overlap) / segment_length,
                "coverage": float(overlap) / segment_length,
            }
            if time_resolution is not None:
                seconds = int(table.at[row.segment_id, "from_service_day_seconds"])
                allocation["time_bucket"] = seconds // _TIME_SECONDS[time_resolution]
            allocations.append(allocation)
        if covered > segment_length * (1 + 1e-9):
            raise ValueError("grid overlap exceeds the segment length")
        uncovered_fraction = max(0.0, 1.0 - covered / segment_length)
        if uncovered_fraction > 1e-9:
            uncovered_value = float(row.value) * uncovered_fraction
            message = (
                f"segment {row.segment_id} is only {covered / segment_length:.6f} "
                f"covered by the grid; {uncovered_value:.6g} g "
                f"({uncovered_fraction:.6f} of the segment) is outside it"
            )
            if on_uncovered == "raise":
                raise ValueError(message)
            if on_uncovered == "warn":
                warnings.warn(message, UserWarning, stacklevel=2)

    if not allocations:
        raise ValueError("no grid cell intersects any emitting segment")
    allocated = pd.DataFrame(allocations)
    if not aggregate:
        return allocated

    dimensions = ["grid_id", "pollutant", "process"]
    if time_resolution is not None:
        dimensions.append("time_bucket")
    totals = allocated.groupby(dimensions, as_index=False, sort=True)["value"].sum()
    out = frame.merge(totals, on="grid_id", how="inner")
    return (
        gpd.GeoDataFrame(out, geometry="geometry", crs=frame.crs)
        .sort_values(dimensions, kind="stable")
        .reset_index(drop=True)
    )
