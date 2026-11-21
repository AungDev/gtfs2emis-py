"""Speed adjustment for the transport table.

R's ``gtfs2gps::adjust_speed()`` replaces — not clamps — speeds outside
``[min_speed, max_speed]`` with ``new_speed``, whose default is the arithmetic
distance-weighted mean of the valid speeds within each shape. Zero-distance
stop segments are exempt (a bus standing at a stop is not a slow outlier), and
timestamps are re-anchored on each trip's first service-day second afterwards.

This module reproduces that rule on the interval table the builder produces.
It does not touch the EMEP ``Vmin``/``Vmax`` clamp, which belongs to the EF
lookup and is a different rule.

Service-day seconds are never wrapped: a trip that crosses midnight
keeps accumulating past 86 400.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import timedelta
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd

__all__ = ["NoValidSpeedError", "adjust_speeds"]

_REQUIRED_COLUMNS = (
    "trip_id",
    "from_service_day_seconds",
    "to_service_day_seconds",
    "dist_km",
    "speed_kmh",
    "cumtime_s",
)


class NoValidSpeedError(ValueError):
    """No interval has a usable speed and ``new_speed`` was not supplied.

    R raises ``gtfs2gps_no_valid_speed_error`` for the same condition; an
    arbitrary fallback here would silently invent a fleet-wide speed.
    """


def _validate_parameters(
    min_speed: float,
    max_speed: float,
    new_speed: float | None,
    cumtime_origin: object = "first_stop",
    shape_start_km: object = None,
) -> None:
    """Apply the transport-model parameter rules (contract §2.1)."""
    if cumtime_origin not in ("first_stop", "shape_start"):
        raise ValueError(
            f"cumtime_origin must be 'first_stop' or 'shape_start', got {cumtime_origin!r}"
        )
    if cumtime_origin == "shape_start":
        if not isinstance(shape_start_km, Mapping):
            raise ValueError(
                "cumtime_origin='shape_start' requires the per-trip "
                "shape_start_km mapping produced by build_transport_table()"
            )
        if not shape_start_km:
            raise ValueError("shape_start_km must not be empty")
    if shape_start_km is not None and not isinstance(shape_start_km, Mapping):
        raise TypeError("shape_start_km must be a mapping of trip_id to kilometres")
    if not isinstance(min_speed, int | float) or isinstance(min_speed, bool):
        raise TypeError(f"min_speed must be a number, got {type(min_speed).__name__}")
    if not isinstance(max_speed, int | float) or isinstance(max_speed, bool):
        raise TypeError(f"max_speed must be a number, got {type(max_speed).__name__}")
    if not math.isfinite(min_speed) or min_speed < 2:
        raise ValueError(f"min_speed must be finite and >= 2 km/h, got {min_speed!r}")
    if not math.isfinite(max_speed) or max_speed < 3:
        raise ValueError(f"max_speed must be finite and >= 3 km/h, got {max_speed!r}")
    if min_speed >= max_speed:
        raise ValueError(
            f"min_speed ({min_speed!r}) must be smaller than max_speed ({max_speed!r})"
        )
    if new_speed is not None:
        if isinstance(new_speed, bool) or not isinstance(new_speed, int | float):
            raise TypeError(f"new_speed must be a number or None, got {type(new_speed).__name__}")
        if not math.isfinite(new_speed) or new_speed < 1:
            raise ValueError(f"new_speed must be finite and >= 1 km/h, got {new_speed!r}")


def _replacement_speed(
    distances: pd.Series, speeds: pd.Series, valid: pd.Series, new_speed: float | None
) -> float:
    """Return the explicit speed, or the distance-weighted mean of valid ones."""
    if new_speed is not None:
        return float(new_speed)
    if not bool(valid.any()):
        raise NoValidSpeedError(
            "no interval has a speed inside the requested range and new_speed "
            "is None; pass new_speed or widen [min_speed, max_speed]"
        )
    total_distance = float(distances[valid].sum())
    if total_distance <= 0:
        raise NoValidSpeedError(
            "the valid intervals have no measurable distance; pass new_speed explicitly"
        )
    # R's adjust_speed() uses weighted.mean(speed, dist), an arithmetic mean.
    # The harmonic sum(d) / sum(d / v) is a different statistic and is not the
    # reference. Valid intervals only; the reference excludes stopped buses.
    return float((speeds[valid] * distances[valid]).sum() / total_distance)


def _adjust_trip_times(frame: gpd.GeoDataFrame, trip_indices: pd.Index) -> None:
    """Recompute each trip's service-day seconds, cumtime and timestamps in place."""
    group = frame.loc[trip_indices]
    old_from = group["from_service_day_seconds"].astype("int64").to_numpy()
    distances = group["dist_km"].astype("float64").to_numpy()
    speeds = group["speed_kmh"].astype("float64").to_numpy()

    anchor = int(old_from[0])
    # R's adjust_speed() computes time = dist / speed as a float, cumtime =
    # cumsum(time), and rounds only when materialising timestamps. Rounding
    # each interval first and accumulating integers drifts from that by up to
    # a second, so accumulate once and round the cumulative value.
    with np.errstate(divide="ignore", invalid="ignore"):
        travel = np.where(
            distances > 0,
            distances * 1000.0 / np.where(speeds > 0, speeds / 3.6, np.nan),
            0.0,
        )
    travel = np.where(np.isnan(speeds), np.nan, travel)
    cumulative = np.cumsum(travel)
    rounded = np.rint(cumulative)
    new_from = np.concatenate(([0.0], rounded[:-1])) + anchor
    new_to = rounded + anchor

    # A shape with no valid speed has NaN travel time. R's NA semantics leave
    # its seconds, cumtime and timestamps undefined rather than claiming a
    # zero-length trip; nullable Int32 keeps that representable.
    frame["from_service_day_seconds"] = frame["from_service_day_seconds"].astype("Int32")
    frame["to_service_day_seconds"] = frame["to_service_day_seconds"].astype("Int32")
    frame.loc[trip_indices, "from_service_day_seconds"] = pd.Series(
        new_from, index=trip_indices, dtype="Int32"
    )
    frame.loc[trip_indices, "to_service_day_seconds"] = pd.Series(
        new_to, index=trip_indices, dtype="Int32"
    )
    frame.loc[trip_indices, "cumtime_s"] = cumulative.astype("float64")

    if "from_timestamp" not in frame.columns:
        return
    old_timestamp = group["from_timestamp"].iloc[0]
    if pd.isna(old_timestamp):
        return
    anchor_timestamp = old_timestamp.to_pydatetime() - timedelta(seconds=int(old_from[0]))
    from_values = [
        pd.NaT
        if not np.isfinite(value)
        else pd.Timestamp(anchor_timestamp + timedelta(seconds=int(value)))
        for value in new_from
    ]
    to_values = [
        pd.NaT
        if not np.isfinite(value)
        else pd.Timestamp(anchor_timestamp + timedelta(seconds=int(value)))
        for value in new_to
    ]
    frame.loc[trip_indices, "from_timestamp"] = pd.Series(
        from_values, index=trip_indices, dtype=frame["from_timestamp"].dtype
    )
    frame.loc[trip_indices, "to_timestamp"] = pd.Series(
        to_values, index=trip_indices, dtype=frame["to_timestamp"].dtype
    )


def adjust_speeds(
    table: gpd.GeoDataFrame,
    *,
    min_speed: float = 2.0,
    max_speed: float = 80.0,
    new_speed: float | None = None,
    cumtime_origin: Literal["first_stop", "shape_start"] = "first_stop",
    shape_start_km: Mapping[str, float] | None = None,
    shape_speed_kmh: Mapping[str, float] | None = None,
) -> gpd.GeoDataFrame:
    """Replace out-of-range and missing speeds, then re-anchor each trip.

    Args:
        table: A transport table from :func:`build_transport_table` (contract
            §4). It is not mutated; the adjusted copy is returned.
        min_speed: Lower valid bound in km/h, finite and >= 2 (contract §2.1).
        max_speed: Upper valid bound in km/h, finite and >= 3, above
            ``min_speed``.
        new_speed: Replacement speed in km/h, finite and >= 1. ``None`` means the
            arithmetic distance-weighted mean of the valid speeds within each
            shape; a shape with no valid speed leaves its intervals undefined.
        cumtime_origin: ``"first_stop"`` (default) starts ``cumtime_s`` at the
            first stop-to-stop interval. ``"shape_start"`` reproduces R's
            absolute values by adding the unobserved shape-start → first-stop
            travel time; it requires ``shape_start_km``.
        shape_start_km: Mapping of ``trip_id`` to the shape-start → first-stop
            distance in kilometres, normally ``table.attrs["shape_start_km"]``
            from :func:`build_transport_table`. Only read for
            ``cumtime_origin="shape_start"``.
        shape_speed_kmh: Optional mapping of ``shape_id`` to R's vertex-weighted
            replacement speed. When omitted, the interval-weighted replacement
            speed is used; ``transport_model`` supplies the exact one.

    Returns:
        A copy with adjusted ``speed_kmh``, recomputed per-interval duration,
        ``from_service_day_seconds``/``to_service_day_seconds``, ``cumtime_s``
        and, when the table carried them, ``from_timestamp``/``to_timestamp``.
        Zero-distance intervals have zero travel time; scheduled dwell is not
        carried into the adjusted timetable (R's ``cumsum(dist / speed)``).
        ``cumtime_origin="shape_start"`` changes only ``cumtime_s``: service-day
        seconds and timestamps stay anchored at the first stop.

    Raises:
        ValueError: For an out-of-range parameter, a missing/invalid
            ``cumtime_origin``, a missing ``shape_start_km`` entry, or a table
            missing a required column.
        NoValidSpeedError: When no valid speed exists and ``new_speed`` is None.
    """
    _validate_parameters(min_speed, max_speed, new_speed, cumtime_origin, shape_start_km)
    if shape_speed_kmh is not None and not isinstance(shape_speed_kmh, Mapping):
        raise TypeError("shape_speed_kmh must be a mapping of shape_id to km/h")
    missing = [column for column in _REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"transport table is missing required column(s) {missing}")
    frame = table.copy()
    if frame.empty:
        return frame

    distances = frame["dist_km"].astype("float64")
    speeds = frame["speed_kmh"].astype("float64")
    finite = speeds.notna() & np.isfinite(speeds)
    valid = (distances > 0) & finite & (speeds >= min_speed) & (speeds <= max_speed)
    replaceable = (distances > 0) & ~valid

    if new_speed is not None:
        frame.loc[replaceable, "speed_kmh"] = float(new_speed)
        replacement_by_shape: dict[object, float] = {}
    else:
        # R's gtfs2emis::transport_model() calls adjust_speed() once per shape
        # file. Its replacement speed is therefore a shape-level weighted.mean,
        # not one mean over the whole feed.
        if "shape_id" in frame.columns:
            groups = frame.groupby("shape_id", sort=False).groups.items()
        else:
            groups = [(None, frame.index)]
        replacements: dict[object, float] = {}
        for key, indices in groups:
            group_valid = valid.loc[indices]
            try:
                replacements[key] = _replacement_speed(
                    distances.loc[indices], speeds.loc[indices], group_valid, None
                )
            except NoValidSpeedError:
                # Match R's NA result: no usable speed leaves the interval
                # unreplaced and its cumtime undefined.
                replacements[key] = float("nan")
        replacement_by_shape = replacements
        if "shape_id" in frame.columns:
            frame.loc[replaceable, "speed_kmh"] = frame.loc[replaceable, "shape_id"].map(
                replacements
            )
        else:
            frame.loc[replaceable, "speed_kmh"] = replacements[None]

    for _trip_id, indices in frame.groupby("trip_id", sort=False).groups.items():
        _adjust_trip_times(frame, indices)

    if cumtime_origin == "shape_start":
        assert shape_start_km is not None
        leading_seconds: dict[str, float] = {}
        for trip_id, indices in frame.groupby("trip_id", sort=False).groups.items():
            key = str(trip_id)
            if key not in shape_start_km:
                raise ValueError(
                    f"shape_start_km has no entry for trip {key!r}; "
                    "pass the mapping produced by build_transport_table()"
                )
            start_distance = float(shape_start_km[key])
            if not math.isfinite(start_distance) or start_distance < 0:
                raise ValueError(
                    f"shape_start_km[{key!r}] must be finite and >= 0, got {start_distance!r}"
                )
            if new_speed is not None:
                leading_speed = float(new_speed)
            else:
                raw_key = frame.at[indices[0], "shape_id"] if "shape_id" in frame.columns else None
                shape_key = None if raw_key is None or pd.isna(raw_key) else str(raw_key)
                if (
                    shape_key is not None
                    and shape_speed_kmh is not None
                    and shape_key in shape_speed_kmh
                ):
                    leading_speed = float(shape_speed_kmh[shape_key])
                elif shape_key is not None and shape_key in replacement_by_shape:
                    leading_speed = float(replacement_by_shape[shape_key])
                elif None in replacement_by_shape:
                    leading_speed = float(replacement_by_shape[None])
                else:
                    raise ValueError(
                        f"no replacement speed for shape {shape_key!r} of trip {key!r}"
                    )
            if math.isnan(leading_speed):
                # R's weighted.mean() over no valid speeds is NaN; the leading
                # cumtime is undefined too, not an invented positive speed.
                leading_seconds[key] = float("nan")
                continue
            if not math.isfinite(leading_speed) or leading_speed <= 0:
                raise ValueError(
                    f"leading speed for trip {key!r} must be finite and > 0, got {leading_speed!r}"
                )
            leading_seconds[key] = (start_distance * 1000.0) / (leading_speed / 3.6)
        leading_by_trip = frame["trip_id"].astype(str).map(leading_seconds)
        frame["cumtime_s"] = frame["cumtime_s"] + leading_by_trip
        if "cumdist_km" in frame.columns:
            start_by_trip = (
                frame["trip_id"]
                .astype(str)
                .map({key: float(shape_start_km[key]) for key in leading_seconds})
            )
            frame["cumdist_km"] = frame["cumdist_km"] + start_by_trip
    return frame
