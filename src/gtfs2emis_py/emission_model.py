"""Public emission-model entry point.

Combines the transport table, a validated fleet and one EF model into a typed
:class:`EmissionResult`. The hot-exhaust path is implemented for all five
models (CETESB, EMEP, EMFAC, MOVES and the CETESB-scaled composite), and the
EMEP wear processes are routed through the same entry point. ``output_path``
writes one Parquet file per shape plus a manifest.
"""

from __future__ import annotations

import numbers
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from gtfs2emis_py import __version__
from gtfs2emis_py.ef import (
    ef_brazil_cetesb,
    ef_brazil_scaled_euro,
    ef_europe_emep,
    ef_usa_emfac,
    ef_usa_moves,
    emi_europe_emep_wear,
)
from gtfs2emis_py.fleet import FleetTable, load_fleet
from gtfs2emis_py.gtfs.slope import slope_class_europe_emep
from gtfs2emis_py.multiply import multiply_ef
from gtfs2emis_py.pollutants import emission_value_unit

__all__ = [
    "EF_MODELS",
    "EmissionResult",
    "EmissionModelError",
    "emission_model",
]

EF_MODELS: tuple[str, ...] = (
    "ef_brazil_cetesb",
    "ef_europe_emep",
    "ef_usa_emfac",
    "ef_usa_moves",
    "ef_brazil_scaled_euro",
)
_WEAR = frozenset({"tyre", "brake", "road"})


class EmissionModelError(ValueError):
    """The request cannot be satisfied by this EF model or process set."""


@dataclass(frozen=True)
class EmissionResult:
    """Typed emission result.

    ``emi`` is the long emission table (segment × vehicle × pollutant ×
    process), ``ef`` the factors actually applied, ``tp_model`` the transport
    model held by reference, and ``units`` the pollutant -> unit map. The
    ``unit`` column is the unit of ``value`` (grams for mass pollutants,
    ``MJ``/``kJ`` for energy species), not the factor's unit.
    """

    emi: pd.DataFrame
    ef: pd.DataFrame
    tp_model: Any
    metadata: dict[str, Any]
    units: dict[str, str]

    def to_long(self) -> pd.DataFrame:
        """Return a copy of the long emission table."""
        return self.emi.copy(deep=True)

    def to_frame(self) -> pd.DataFrame:
        """Wide per-segment table: one row per segment, one column per pollutant."""
        frame = self.emi.copy(deep=True)
        wide = frame.pivot_table(
            index="segment_id",
            columns=["pollutant", "process"],
            values="value",
            aggfunc="sum",
        )
        wide.columns = [f"{pollutant}_{process}" for pollutant, process in wide.columns]
        return wide.reset_index()

    def to_dataframe(self) -> pd.DataFrame:
        """Alias of :meth:`to_frame`."""
        return self.to_frame()

    def to_pandas(self) -> pd.DataFrame:
        """Alias of :meth:`to_frame`."""
        return self.to_frame()

    def emis_to_dt(self) -> pd.DataFrame:
        """Alias of :meth:`to_long`, preserving R's name."""
        return self.to_long()

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        tp_model: Any,
        metadata: dict[str, Any] | None = None,
        units: dict[str, str] | None = None,
    ) -> EmissionResult:
        """Rebuild a result from a long table.

        Raises:
            ValueError: For a missing column or a pollutant carrying two units.
        """
        required = {
            "segment_id",
            "shape_id",
            "trip_id",
            "stop_sequence",
            "vehicle_type",
            "fleet_row",
            "pollutant",
            "process",
            "value",
            "unit",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"long emission table is missing column(s) {missing}")
        copied = frame.copy(deep=True)
        derived: dict[str, str] = {}
        for pollutant, group in copied.groupby("pollutant", sort=True):
            values = group["unit"].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"pollutant {pollutant!r} carries multiple units")
            derived[str(pollutant)] = str(values.iloc[0])
        if units is not None:
            derived.update({str(key): str(value) for key, value in units.items()})
        return cls(
            emi=copied,
            ef=pd.DataFrame(columns=["pollutant", "vehicle_index", "ef", "unit"]),
            tp_model=tp_model,
            metadata={} if metadata is None else dict(metadata),
            units=derived,
        )


def emission_model(
    tp_model: Any,
    ef_model: str,
    fleet_data: FleetTable | pd.DataFrame | str | Path,
    pollutant: str | Sequence[str],
    *,
    reference_year: int = 2020,
    process: str | Sequence[str] = "hot_exhaust",
    heightfile: str | Path | None = None,
    parallel: bool = True,
    ncores: int | None = None,
    output_path: str | Path | None = None,
    resume: bool = False,
) -> EmissionResult:
    """Estimate hot-exhaust or EMEP wear emissions for a transport model.

    Args:
        tp_model: A :class:`TransportModel` or its GeoDataFrame.
        ef_model: One of :data:`EF_MODELS`.
        fleet_data: A validated :class:`FleetTable`, or a table/path passed to
            :func:`load_fleet` with the matching model.
        pollutant: Pollutant name(s) supported by the model.
        reference_year: EMFAC/MOVES reference year.
        process: ``hot_exhaust``, ``tyre``, ``brake`` or ``road``. Wear
            processes are EMEP-only.
        heightfile: Optional DEM for EMEP slope classification.
        parallel, ncores: flags. The emission path consumes the
            already-built transport table; per-shape parallelism is applied in
            :func:`transport_model`.
        output_path: Optional directory for per-shape Parquet output and the
            emission manifest; when omitted the result is returned in memory.
        resume: Reuse complete per-shape files recorded in an existing
            manifest; requires ``output_path``.

    Returns:
        An :class:`EmissionResult`.

    Raises:
        EmissionModelError: For an unknown model, a wear process, a unit
            mismatch or an unsupported pollutant.
        ValueError: For invalid parameters.
    """
    if ef_model not in EF_MODELS:
        raise EmissionModelError(f"unknown ef_model {ef_model!r}; expected one of {EF_MODELS}")
    processes = _as_strings(process, "process")
    accepted = {"hot_exhaust", *_WEAR}
    unknown = sorted(set(processes) - accepted)
    if unknown:
        raise EmissionModelError(f"unknown process(es) {unknown}; accepted: {sorted(accepted)}")
    if set(processes).intersection(_WEAR) and ef_model != "ef_europe_emep":
        raise EmissionModelError(
            f"wear processes (tyre, brake, road) are EMEP-only; {ef_model} supports hot_exhaust"
        )
    if resume and output_path is None:
        raise ValueError("resume=True requires output_path")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a bool")
    if ncores is not None:
        if isinstance(ncores, bool) or not isinstance(ncores, int) or ncores < 1:
            raise ValueError("ncores must be a positive int or None")
        if parallel and ncores > (os.cpu_count() or 1):
            raise ValueError(f"ncores must be <= {os.cpu_count() or 1}")
    pollutants = _as_strings(pollutant, "pollutant")
    if not pollutants:
        raise ValueError("pollutant must not be empty")
    if isinstance(reference_year, bool) or not isinstance(reference_year, numbers.Integral):
        raise TypeError("reference_year must be an int")

    table = tp_model.table if hasattr(tp_model, "table") else tp_model
    if not isinstance(table, gpd.GeoDataFrame):
        raise TypeError("tp_model must be a TransportModel or a GeoDataFrame")
    required = {"speed_kmh", "dist_km"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"tp_model is missing column(s) {sorted(missing)}")
    if table.empty:
        raise EmissionModelError("tp_model has no rows")

    if isinstance(fleet_data, FleetTable):
        fleet = fleet_data
        if fleet.ef_model != ef_model:
            if not (ef_model == "ef_brazil_scaled_euro" and fleet.ef_model == "ef_brazil_cetesb"):
                raise EmissionModelError(
                    f"fleet was validated for {fleet.ef_model}, not {ef_model}"
                )
    else:
        fleet_model = "ef_brazil_cetesb" if ef_model == "ef_brazil_scaled_euro" else ef_model
        fleet = load_fleet(fleet_data, ef_model=fleet_model)

    if ef_model in {"ef_usa_emfac", "ef_usa_moves"} and not isinstance(reference_year, int):
        raise TypeError("reference_year must be an int")

    slope: Any = pd.Series(0.0, index=table.index)
    if heightfile is not None:
        if ef_model not in {"ef_europe_emep", "ef_brazil_scaled_euro"}:
            raise EmissionModelError("heightfile is only used by EMEP-based models")
        table = slope_class_europe_emep(table, heightfile, keep=True)
        slope = table["slope"]

    emi_parts: list[pd.DataFrame] = []
    ef_parts: list[pd.DataFrame] = []
    units: dict[str, str] = {}
    for group_value, group_fleet in _fleet_groups(fleet):
        subset = _subset_for_group(table, fleet, group_value)
        subset = subset[subset["dist_km"] > 0]
        if subset.empty:
            continue
        segment_ids = subset.index.to_numpy()
        dist = subset["dist_km"].to_numpy(dtype=float)
        speed = [float(value) for value in subset["speed_kmh"]]
        group_slope = slope.loc[subset.index].tolist() if heightfile is not None else 0.0
        if "hot_exhaust" in processes:
            for fleet_index, vehicle in group_fleet.iterrows():
                composition = float(vehicle[fleet.composition_column])
                if composition == 0:
                    continue
                ef_frame = _vehicle_factors(
                    vehicle,
                    ef_model=ef_model,
                    pollutants=pollutants,
                    speed=speed,
                    slope=group_slope,
                    reference_year=int(reference_year),
                    n_segments=len(subset),
                )
                ef_frame["vehicle_type"] = _vehicle_label(vehicle, ef_model)
                ef_frame["fleet_row"] = int(fleet_index)
                ef_parts.append(ef_frame)

                mass_mask = ef_frame["unit"].map(_is_mass_per_distance)
                mass_frame = ef_frame[mass_mask]
                energy_frame = ef_frame[~mass_mask]
                if not energy_frame.empty:
                    bad_units = sorted(set(energy_frame["unit"]) - {"MJ/km", "kJ/km"})
                    if bad_units or set(energy_frame["pollutant"]) != {"EC"}:
                        raise EmissionModelError(
                            f"{ef_model} pollutant(s) {sorted(set(energy_frame['pollutant']))} "
                            f"carry factor unit(s) {sorted(set(energy_frame['unit']))}; "
                            "this path requires an energy-consumption conversion, not a "
                            "distance multiplication"
                        )

                if not mass_frame.empty:
                    factor_columns = ["pollutant", "vehicle_index", "ef", "unit"]
                    if "segment" in mass_frame.columns:
                        factor_columns.append("segment")
                    factors = mass_frame[factor_columns].copy()
                    emissions = multiply_ef([composition], dist, factors, aggregate=False)
                    emissions = _decorate_emission_rows(
                        emissions,
                        subset,
                        segment_ids,
                        vehicle,
                        ef_model,
                        int(fleet_index),
                    )
                    emissions["unit"] = emissions["pollutant"].map(
                        lambda name: emission_value_unit(str(name), model=ef_model)
                    )
                    emi_parts.append(emissions)
                    units.update(
                        {
                            str(name): emission_value_unit(str(name), model=ef_model)
                            for name in mass_frame["pollutant"].unique()
                        }
                    )

                if not energy_frame.empty:
                    energy_rows = energy_frame.copy()
                    if "segment" not in energy_rows.columns:
                        energy_rows = pd.concat(
                            [energy_rows.assign(segment=segment) for segment in range(len(dist))],
                            ignore_index=True,
                        )
                    energy_rows["segment"] = energy_rows["segment"].to_numpy(dtype=int)
                    energy_rows["value"] = (
                        energy_rows["ef"].to_numpy(dtype=float)
                        * composition
                        * dist[energy_rows["segment"].to_numpy(dtype=int)]
                    )
                    energy_rows = _decorate_emission_rows(
                        energy_rows,
                        subset,
                        segment_ids,
                        vehicle,
                        ef_model,
                        int(fleet_index),
                        value_column="value",
                    )
                    energy_rows["unit"] = energy_rows["pollutant"].map(
                        lambda name: emission_value_unit(str(name), model=ef_model)
                    )
                    emi_parts.append(energy_rows)
                    units.update(
                        {
                            str(name): emission_value_unit(str(name), model=ef_model)
                            for name in energy_rows["pollutant"].unique()
                        }
                    )

        for process_name in [value for value in processes if value in _WEAR]:
            wear = emi_europe_emep_wear(
                dist,
                speed,
                pollutants,
                group_fleet["veh_type"].tolist(),
                group_fleet[fleet.composition_column].tolist(),
                load=0.5,
                process=process_name,
            )
            segments = wear["segment"].to_numpy(dtype=int)
            wear["segment_id"] = [segment_ids[index] for index in segments]
            wear["shape_id"] = subset["shape_id"].to_numpy()[segments]
            wear["trip_id"] = subset["trip_id"].to_numpy()[segments]
            wear["stop_sequence"] = subset["stop_sequence"].to_numpy()[segments]
            wear["fleet_row"] = [int(group_fleet.index[index]) for index in wear["vehicle_index"]]
            wear["unit"] = "g"
            wear = wear.rename(columns={"emi_g": "value"})
            emi_parts.append(
                wear[
                    [
                        "segment_id",
                        "shape_id",
                        "trip_id",
                        "stop_sequence",
                        "veh_type",
                        "fleet_row",
                        "pollutant",
                        "process",
                        "value",
                        "unit",
                    ]
                ].rename(columns={"veh_type": "vehicle_type"})
            )
            units.update({name: "g" for name in pollutants})

    if not emi_parts:
        raise EmissionModelError(
            "no fleet row with a positive composition matched the transport table"
        )
    emi = pd.concat(emi_parts, ignore_index=True)
    ef = (
        pd.concat(ef_parts, ignore_index=True)
        if ef_parts
        else pd.DataFrame(columns=["pollutant", "vehicle_index", "ef", "unit"])
    )
    metadata = {
        "ef_model": ef_model,
        "pollutant": pollutants,
        "reference_year": int(reference_year),
        "process": processes,
        "package_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    result = EmissionResult(emi=emi, ef=ef, tp_model=tp_model, metadata=metadata, units=units)
    if output_path is not None:
        from gtfs2emis_py.emission_writer import write_emission_output

        written = write_emission_output(result, output_path, resume=resume)
        metadata["output_path"] = str(written.path)
        metadata["emission_manifest"] = written.manifest
    return result


_MASS_UNIT_RE = re.compile(r"^(g|kg)/(km|mi|m)$")


def _is_mass_per_distance(unit: object) -> bool:
    """Return True for a mass-per-distance factor unit."""
    return isinstance(unit, str) and _MASS_UNIT_RE.fullmatch(unit.strip()) is not None


def _decorate_emission_rows(
    frame: pd.DataFrame,
    subset: gpd.GeoDataFrame,
    segment_ids: Any,
    vehicle: pd.Series,
    ef_model: str,
    fleet_index: int,
    *,
    value_column: str = "emi_g",
) -> pd.DataFrame:
    """Attach transport keys and identity columns to vehicle emission rows."""
    frame = frame.copy()
    segments = frame["segment"].to_numpy(dtype=int)
    frame["segment_id"] = [segment_ids[index] for index in segments]
    frame["shape_id"] = subset["shape_id"].to_numpy()[segments]
    frame["trip_id"] = subset["trip_id"].to_numpy()[segments]
    frame["stop_sequence"] = subset["stop_sequence"].to_numpy()[segments]
    frame["vehicle_type"] = _vehicle_label(vehicle, ef_model)
    frame["fleet_row"] = fleet_index
    frame["pollutant"] = frame["pollutant"].astype(str)
    frame["process"] = "hot_exhaust"
    if "unit" not in frame.columns:
        frame["unit"] = ""
    frame = frame.rename(columns={value_column: "value"})
    return frame[
        [
            "segment_id",
            "shape_id",
            "trip_id",
            "stop_sequence",
            "vehicle_type",
            "fleet_row",
            "pollutant",
            "process",
            "value",
            "unit",
        ]
    ]


def _vehicle_factors(
    vehicle: pd.Series,
    *,
    ef_model: str,
    pollutants: list[str],
    speed: Sequence[float],
    slope: Any,
    reference_year: int,
    n_segments: int,
) -> pd.DataFrame:
    """Build per-vehicle factors for one speed group."""
    if ef_model == "ef_brazil_cetesb":
        factors = ef_brazil_cetesb(
            pollutants, str(vehicle["type_name_br"]), int(vehicle["model_year"])
        )
        frame = factors.table[["pollutant", "ef", "unit"]].copy()
        frame["vehicle_index"] = 0
        return frame
    if ef_model == "ef_europe_emep":
        factors = ef_europe_emep(
            speed,
            str(vehicle["veh_type"]),
            str(vehicle["euro"]),
            pollutants,
            fuel=str(vehicle["fuel"]),
            tech=str(vehicle["tech"]),
            slope=slope,
            load=0.5,
        )
        frame = factors.table[["pollutant", "vehicle_index", "ef", "unit", "speed_kmh"]].copy()
        return _by_speed(frame, speed, integer_speed=True)
    if ef_model == "ef_usa_emfac":
        factors = ef_usa_emfac(
            pollutants,
            reference_year,
            str(vehicle["fuel"]),
            model_year=int(vehicle["model_year"]),
            speed=speed,
        )
        frame = factors.table[["pollutant", "ef", "unit", "speed_kmh"]].copy()
        frame["vehicle_index"] = 0
        return _by_speed(frame, speed)
    if ef_model == "ef_usa_moves":
        factors = ef_usa_moves(
            pollutants,
            int(vehicle["model_year"]),
            reference_year=reference_year,
            speed=speed,
            fuel=str(vehicle["fuel"]),
        )
        frame = factors.table[["pollutant", "ef", "unit", "speed_kmh"]].copy()
        frame["vehicle_index"] = 0
        return _by_speed(frame, speed)
    if ef_model == "ef_brazil_scaled_euro":
        missing = {"type_name_eu", "euro", "tech"} - set(vehicle.index)
        if missing:
            raise EmissionModelError(
                f"ef_brazil_scaled_euro needs fleet column(s) {sorted(missing)}"
            )
    factors = ef_brazil_scaled_euro(
        pollutants,
        speed,
        str(vehicle["type_name_br"]),
        int(vehicle["model_year"]),
        str(vehicle["type_name_eu"]),
        str(vehicle["euro"]),
        fuel=str(vehicle["fuel"]),
        tech=str(vehicle["tech"]),
        slope=slope,
        load=0.5,
    )
    frame = factors.table[["pollutant", "vehicle_index", "ef", "unit", "speed_kmh"]].copy()
    return _by_speed(frame, speed, integer_speed=True)


def _by_speed(
    frame: pd.DataFrame, speed: Sequence[float], *, integer_speed: bool = False
) -> pd.DataFrame:
    """Attach the segment index by matching the factor row's speed.

    A repeated speed in the input yields one factor row per matching segment,
    so a duplicate segment speed is never silently folded into one factor.
    """

    def key(value: float) -> float:
        """Return the speed bin key used to group equal speeds."""
        return float(int(round(value))) if integer_speed else round(float(value), 1)

    positions: dict[float, list[int]] = {}
    for index, value in enumerate(speed):
        positions.setdefault(key(value), []).append(index)
    frame = frame.assign(_key=frame["speed_kmh"].map(key))
    frame = frame.drop_duplicates(subset=["pollutant", "vehicle_index", "_key"], keep="first")
    rows = []
    for _, factor in frame.iterrows():
        for segment in positions.get(float(factor["_key"]), []):
            row = factor.to_dict()
            row["segment"] = segment
            rows.append(row)
    return pd.DataFrame(rows).drop(columns=["_key", "speed_kmh"])


def _fleet_groups(fleet: FleetTable) -> list[tuple[Any, pd.DataFrame]]:
    """Yield fleet groups for average or per-vehicle mode."""
    if not fleet.group_columns:
        return [(None, fleet.table)]
    column = fleet.group_columns[0]
    return list(fleet.table.groupby(column, sort=True))


def _subset_for_group(table: gpd.GeoDataFrame, fleet: FleetTable, value: Any) -> gpd.GeoDataFrame:
    """Select the transport rows belonging to one fleet group."""
    if not fleet.group_columns:
        return table
    column = fleet.group_columns[0]
    if column not in table.columns:
        raise EmissionModelError(
            f"fleet groups by {column!r}, which the transport table does not carry"
        )
    return table[table[column].astype(str) == str(value)]


def _vehicle_label(vehicle: pd.Series, ef_model: str) -> str:
    """Format a stable vehicle label for the output table."""
    if ef_model in {"ef_brazil_cetesb", "ef_brazil_scaled_euro"}:
        return str(vehicle["type_name_br"])
    if ef_model == "ef_europe_emep":
        return str(vehicle["veh_type"])
    return "BUS_URBAN_D"


def _as_strings(value: object, name: str) -> list[str]:
    """Coerce a string or sequence to a non-empty list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence | np.ndarray | pd.Series):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a string or a sequence")
