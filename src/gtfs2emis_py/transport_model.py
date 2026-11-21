"""Public transport-model entry point (contract §2/§9).

``transport_model()`` is the function a caller actually uses: read and
validate a feed, build the per-interval table, adjust speeds, optionally filter
by route type or service, and either return the table or write one GeoParquet
file per shape.

The manifest type here is the one written down in the contract. The on-disk
``manifest.json`` is the writer's schema; this function maps it into
``TransportManifest`` so callers do not handle a raw dict.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import gtfs_kit as gk
import pandas as pd

from gtfs2emis_py.gtfs.interpolation import build_transport_table, parse_service_date
from gtfs2emis_py.gtfs.parallel import ShapeProcessingError, ShapeTask, map_shapes
from gtfs2emis_py.gtfs.reader import GtfsFeed, read_gtfs
from gtfs2emis_py.gtfs.speed import adjust_speeds
from gtfs2emis_py.gtfs.writer import write_transport_output

__all__ = [
    "BUS_ROUTE_TYPES",
    "ShapeResult",
    "TransportManifest",
    "TransportModel",
    "TransportModelError",
    "transport_model",
]

BUS_ROUTE_TYPES: tuple[int, ...] = (3, *range(700, 800))
_FREQ_SUFFIX = re.compile(r"-freq-\d+$")


class TransportModelError(RuntimeError):
    """No shape produced rows, or the selection left nothing to model."""

    def __init__(self, message: str, manifest: TransportManifest | None = None) -> None:
        """Store the error message and optional manifest."""
        self.manifest = manifest
        super().__init__(message)


@dataclass(frozen=True)
class ShapeResult:
    """One shape's outcome in an on-disk run."""

    shape_id: str
    file: str | None
    rows: int
    status: str
    error: str | None


@dataclass(frozen=True)
class TransportManifest:
    """Run-level provenance and counters (contract §9.2).

    ``parameters`` carries the effective run settings, including parallel
    controls, output destination, resume flag and the input feed hash.
    """

    schema_version: int
    generated_at: str
    gtfs_source: str
    gtfs_sha256: str | None
    parameters: dict[str, Any]
    counts: dict[str, int]
    skipped: dict[str, int]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    shapes: tuple[ShapeResult, ...]


@dataclass(frozen=True)
class TransportModel:
    """In-memory result: the contract table plus its manifest."""

    table: gpd.GeoDataFrame
    manifest: TransportManifest


def _error_manifest(
    *,
    source: str,
    source_sha256: str | None,
    parameters: dict[str, Any],
    skipped: dict[str, int],
    errors: tuple[str, ...],
) -> TransportManifest:
    """Build the typed manifest attached to a fatal ``TransportModelError``."""
    return TransportManifest(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        gtfs_source=source,
        gtfs_sha256=source_sha256,
        parameters=parameters,
        counts={"shapes": 0, "trips": 0, "rows": 0},
        skipped=skipped,
        warnings=(),
        errors=errors,
        shapes=(),
    )


def transport_model(
    gtfs_data: str | Path | GtfsFeed,
    *,
    min_speed: float = 2.0,
    max_speed: float = 80.0,
    new_speed: float | None = None,
    cumtime_origin: Literal["first_stop", "shape_start"] = "first_stop",
    parallel: bool = True,
    ncores: int | None = None,
    spatial_resolution: float = 100.0,
    route_types: Sequence[int] = BUS_ROUTE_TYPES,
    service_date: str | date | None = None,
    service_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    resume: bool = False,
) -> TransportModel | TransportManifest:
    """Build the stop-to-stop transport table for a GTFS feed.

    Args:
        gtfs_data: A feed path or an already-read :class:`GtfsFeed`.
        min_speed: Lower valid speed in km/h, at least 2.
        max_speed: Upper valid speed in km/h, at least 3 and above ``min_speed``.
        new_speed: Replacement speed in km/h when given, at least 1.
        cumtime_origin: ``"first_stop"`` (default) starts ``cumtime_s`` at the
            first stop-to-stop interval. ``"shape_start"`` reproduces R's
            absolute values by adding the unobserved shape-start → first-stop
            travel time, at the shape's replacement speed. It changes only
            ``cumtime_s``, not service-day seconds or timestamps.
        parallel: switch. When true and ``ncores`` is None or > 1,
            shapes are built through :func:`map_shapes`; false or ``ncores=1``
            uses the serial feed-at-once path.
        ncores: Worker count, 1 to ``cpu_count()``; ``None`` means
            ``max(1, cpu_count() - 1)``. Ignored when ``parallel=False``.
        spatial_resolution: Maximum spherical edge length between densified
            vertices, in metres.
        route_types: GTFS route types to keep; bus (3) and 700-799 by default.
        service_date: Keep services active on this date and materialise
            timestamps. Accepts a ``date`` object or an ISO string in either
            extended (``YYYY-MM-DD``) or basic (``YYYYMMDD``) form; the builder
            and service filter use the same parser.
        service_ids: Keep exactly these services; mutually exclusive with
            ``service_date``.
        output_path: Existing directory for per-shape GeoParquet output.
        resume: Skip shapes whose file is complete and matches the manifest.

    Returns:
        ``TransportModel`` when ``output_path`` is None; otherwise
        ``TransportManifest`` after writing ``shapes/`` and ``manifest.json``.

    Raises:
        ValueError: For an invalid parameter, an empty route-type selection, a
            selection with no trips, ``resume`` without ``output_path``, or a
            missing output directory.
        TransportModelError: When no shape produced rows. Its ``manifest``
            carries the input source, parameters, zero counts and the error.
        ManifestError: When ``resume=True`` and the existing manifest cannot
            be resumed against the current request.
        TransportWriteError: After every shape has been attempted, if any
            shape failed to write. Its ``manifest`` is the written manifest
            with the failed shape entries; successful files remain on disk.
    """
    _validate_parameters(
        min_speed=min_speed,
        max_speed=max_speed,
        new_speed=new_speed,
        cumtime_origin=cumtime_origin,
        parallel=parallel,
        ncores=ncores,
        spatial_resolution=spatial_resolution,
        route_types=route_types,
        service_date=service_date,
        service_ids=service_ids,
        output_path=output_path,
        resume=resume,
    )
    feed = gtfs_data if isinstance(gtfs_data, GtfsFeed) else read_gtfs(gtfs_data)
    if os.environ.get("GTFS2EMIS_COVERAGE_SERIAL") == "1":
        # Coverage does not follow loky children; the CI coverage job runs the
        # same code serially so the published figure measures one program.
        parallel = False
    source = feed.source
    source_sha256 = _sha256(source)
    parameters = {
        "min_speed": float(min_speed),
        "max_speed": float(max_speed),
        "new_speed": None if new_speed is None else float(new_speed),
        "cumtime_origin": cumtime_origin,
        "spatial_resolution": float(spatial_resolution),
        "route_types": [int(value) for value in route_types],
        "service_date": None if service_date is None else str(service_date),
        "service_ids": None if service_ids is None else list(service_ids),
        "parallel": bool(parallel),
        "ncores": ncores,
        "output_path": None if output_path is None else str(output_path),
        "resume": bool(resume),
        "gtfs_sha256": source_sha256,
    }

    selected_trips, selection_stats = _select_trips(
        feed,
        route_types=[int(value) for value in route_types],
        service_date=service_date,
        service_ids=service_ids,
    )
    if not selected_trips:
        message = (
            "no trips survived route/service selection; "
            f"route_types={sorted(set(route_types))}, "
            f"service_date={service_date!r}, service_ids={service_ids!r}"
        )
        raise TransportModelError(
            message,
            manifest=_error_manifest(
                source=source,
                source_sha256=source_sha256,
                parameters=parameters,
                skipped=selection_stats,
                errors=(message,),
            ),
        )
    build_kwargs: dict[str, Any] = {
        "spatial_resolution": spatial_resolution,
        "service_date": service_date,
        "include_shape_start": cumtime_origin == "shape_start",
        "min_speed": min_speed,
        "max_speed": max_speed,
    }
    if parallel and (ncores is None or ncores > 1):
        table = _parallel_transport(
            feed,
            selected_trips=selected_trips,
            ncores=ncores,
            build_kwargs=build_kwargs,
        )
    else:
        table = build_transport_table(
            feed,
            trip_ids=sorted(selected_trips),
            **build_kwargs,
        )
    table = adjust_speeds(
        table,
        min_speed=min_speed,
        max_speed=max_speed,
        new_speed=new_speed,
        cumtime_origin=cumtime_origin,
        shape_start_km=table.attrs.get("shape_start_km"),
        shape_speed_kmh=table.attrs.get("shape_replacement_speed_kmh"),
    )
    skipped = dict(selection_stats)
    for key, value in table.attrs.get("build_stats", {}).items():
        skipped[key] = skipped.get(key, 0) + int(value)
    if table.empty:
        message = (
            "no trips survived route/service selection; "
            f"route_types={sorted(set(route_types))}, "
            f"service_date={service_date!r}, service_ids={service_ids!r}"
        )
        raise TransportModelError(
            message,
            manifest=_error_manifest(
                source=source,
                source_sha256=source_sha256,
                parameters=parameters,
                skipped=skipped,
                errors=(message,),
            ),
        )
    table = table.sort_values(["shape_id", "trip_id", "stop_sequence"], kind="stable").reset_index(
        drop=True
    )
    counts = {
        "shapes": int(table["shape_id"].nunique()),
        "trips": int(table["trip_id"].nunique()),
        "rows": int(len(table)),
    }
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    shape_errors = tuple(
        f"{failure.shape_id}: {failure.message}" for failure in table.attrs.get("failures", ())
    )
    manifest = TransportManifest(
        schema_version=1,
        generated_at=generated_at,
        gtfs_source=source,
        gtfs_sha256=source_sha256,
        parameters=parameters,
        counts=counts,
        skipped=skipped,
        warnings=tuple(table.attrs.get("warnings", ())),
        errors=shape_errors,
        shapes=tuple(
            ShapeResult(
                shape_id=str(shape_id),
                file=None,
                rows=int(len(group)),
                status="done",
                error=None,
            )
            for shape_id, group in table.groupby("shape_id", sort=True)
        ),
    )
    if output_path is None:
        return TransportModel(table=table, manifest=manifest)

    written = write_transport_output(
        table,
        output_path,
        resume=resume,
        parameters={"transport_model": parameters},
        source_sha256=source_sha256,
        skipped=skipped,
        warnings=tuple(table.attrs.get("warnings", ())),
    )
    shapes = tuple(
        ShapeResult(
            shape_id=entry["shape_id"],
            file=entry["file"],
            rows=entry["rows"],
            status=entry["status"],
            error=entry["error"],
        )
        for entry in written["shapes"]
    )
    failed = tuple(
        f"{entry['shape_id']}: {entry['error']}"
        for entry in written["shapes"]
        if entry["status"] == "failed"
    )
    return TransportManifest(
        schema_version=1,
        generated_at=generated_at,
        gtfs_source=source,
        gtfs_sha256=source_sha256,
        parameters=parameters,
        counts=counts,
        skipped=skipped,
        warnings=tuple(table.attrs.get("warnings", ())),
        errors=shape_errors + failed,
        shapes=shapes,
    )


def _validate_parameters(
    *,
    min_speed: object,
    max_speed: object,
    new_speed: object,
    cumtime_origin: object,
    parallel: object,
    ncores: object,
    spatial_resolution: object,
    route_types: object,
    service_date: object,
    service_ids: object,
    output_path: object,
    resume: object,
) -> None:
    """Validate every transport-model parameter before work starts."""
    for name, value, lower in (
        ("min_speed", min_speed, 2.0),
        ("max_speed", max_speed, 3.0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < lower:
            raise ValueError(f"{name} must be finite and >= {lower}")
    if float(min_speed) >= float(max_speed):  # type: ignore[arg-type]
        raise ValueError("min_speed must be below max_speed")
    if new_speed is not None:
        if isinstance(new_speed, bool) or not isinstance(new_speed, (int, float)):
            raise TypeError("new_speed must be numeric or None")
        if not math.isfinite(float(new_speed)) or float(new_speed) < 1:
            raise ValueError("new_speed must be finite and >= 1")
    if cumtime_origin not in ("first_stop", "shape_start"):
        raise ValueError(
            f"cumtime_origin must be 'first_stop' or 'shape_start', got {cumtime_origin!r}"
        )
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a bool")
    if isinstance(ncores, bool) or not isinstance(ncores, (int, type(None))):
        raise TypeError("ncores must be an int or None")
    if ncores is not None:
        if ncores < 1:
            raise ValueError("ncores must be >= 1")
        cpu_count = os.cpu_count() or 1
        if parallel and ncores > cpu_count:
            raise ValueError(f"ncores must be <= {cpu_count}")
    resolution = float(spatial_resolution) if isinstance(spatial_resolution, (int, float)) else None
    if resolution is None or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("spatial_resolution must be finite and > 0")
    if isinstance(route_types, str) or not isinstance(route_types, Sequence):
        raise TypeError("route_types must be a sequence of ints")
    if not route_types:
        raise ValueError("route_types must not be empty")
    if service_date is not None and service_ids is not None:
        raise ValueError("service_date and service_ids are mutually exclusive")
    if service_date is not None and not isinstance(service_date, (str, date)):
        raise TypeError("service_date must be a string, date or None")
    if service_ids is not None:
        if isinstance(service_ids, str) or not isinstance(service_ids, Sequence):
            raise TypeError("service_ids must be a sequence of strings")
        if not service_ids:
            raise ValueError("service_ids must not be empty")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if resume and output_path is None:
        raise ValueError("resume=True requires output_path")
    if output_path is not None and not Path(str(output_path)).is_dir():
        raise ValueError(f"output_path {output_path!r} is not an existing directory")


def _build_one_shape(task: ShapeTask) -> gpd.GeoDataFrame:
    """Build one shape from its self-contained mini-feed payload.

    The payload is a :class:`gtfs_kit.Feed` containing only this shape's trips,
    stop times, stops, routes, agency and frequencies; the worker never sees
    the full feed.
    """
    payload = task.payload
    return build_transport_table(payload["feed"], trip_ids=payload["trip_ids"], **payload["kwargs"])


def _parallel_transport(
    feed: GtfsFeed,
    *,
    selected_trips: set[str],
    ncores: int | None,
    build_kwargs: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Build selected shapes through map_shapes, preserving input order.

    Trips that cannot be scheduled are counted in the parent so parallel and
    serial manifests agree. Each scheduled shape receives a mini-feed built
    from its own tables, so the payload size scales with the shape rather than
    the whole feed.
    """
    trips = feed.feed.trips.copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["shape_id"] = trips["shape_id"].astype("string")
    selected_mask = trips["trip_id"].map(lambda value: _base_trip_id(value) in selected_trips)
    shapes = feed.feed.shapes.copy()
    shapes["shape_id"] = shapes["shape_id"].astype(str)
    shape_counts = shapes.groupby("shape_id", sort=False).size()
    shape_ids = set(shape_counts.index)
    valid_shapes = set(shape_counts[shape_counts >= 2].index)
    trip_shape = trips["shape_id"].astype("string")
    missing_shape = selected_mask & (trip_shape.isna() | ~trip_shape.isin(shape_ids))
    short_shape = selected_mask & trip_shape.isin(shape_ids) & ~trip_shape.isin(valid_shapes)
    no_shape = int(missing_shape.sum())
    selected = trips[selected_mask & trip_shape.isin(valid_shapes)]
    stop_times_by_trip = {
        str(trip_id): table
        for trip_id, table in feed.feed.stop_times.groupby("trip_id", sort=False)
    }
    tasks = []
    for shape_id, group in selected.groupby("shape_id", sort=True):
        shape_trip_ids = sorted(group["trip_id"].astype(str))
        mini_trips = trips.loc[trips["trip_id"].isin(shape_trip_ids)].copy()
        mini_stop_times = pd.concat(
            [stop_times_by_trip[trip_id] for trip_id in shape_trip_ids],
            ignore_index=True,
        )
        mini_stops = feed.feed.stops.loc[
            feed.feed.stops["stop_id"].astype(str).isin(set(mini_stop_times["stop_id"].astype(str)))
        ].copy()
        mini_routes = feed.feed.routes.loc[
            feed.feed.routes["route_id"].astype(str).isin(set(mini_trips["route_id"]))
        ].copy()
        mini_shapes = feed.feed.shapes.loc[
            feed.feed.shapes["shape_id"].astype(str) == str(shape_id)
        ].copy()
        mini_frequencies = None
        if feed.feed.frequencies is not None and not feed.feed.frequencies.empty:
            frequencies = feed.feed.frequencies.copy()
            frequencies["trip_id"] = frequencies["trip_id"].astype(str)
            mini_frequencies = frequencies.loc[frequencies["trip_id"].isin(shape_trip_ids)].copy()
        mini_feed = gk.Feed(
            dist_units=feed.feed.dist_units,
            agency=feed.feed.agency.copy() if feed.feed.agency is not None else None,
            stops=mini_stops,
            routes=mini_routes,
            trips=mini_trips,
            stop_times=mini_stop_times,
            shapes=mini_shapes,
            frequencies=mini_frequencies,
        )
        tasks.append(
            ShapeTask(
                shape_id=str(shape_id),
                payload={"feed": mini_feed, "trip_ids": shape_trip_ids, "kwargs": build_kwargs},
            )
        )
    if not tasks:
        return build_transport_table(feed, trip_ids=sorted(selected_trips), **build_kwargs)
    outcome = map_shapes(_build_one_shape, tasks, n_jobs=ncores)
    frames = [result for result in outcome.results if result is not None and not result.empty]
    if not frames:
        if outcome.failures:
            raise ShapeProcessingError(outcome.failures)
        return build_transport_table(feed, trip_ids=sorted(selected_trips), **build_kwargs)
    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs
    )
    stats: dict[str, int] = {
        "skipped_no_shape": no_shape,
        "skipped_short_shape": int(short_shape.sum()),
    }
    warnings: list[str] = []
    for frame in frames:
        for key, value in frame.attrs.get("build_stats", {}).items():
            stats[key] = stats.get(key, 0) + int(value)
        warnings.extend(frame.attrs.get("warnings", ()))
    combined.attrs["build_stats"] = stats
    combined.attrs["warnings"] = tuple(warnings)
    combined.attrs["failures"] = tuple(outcome.failures)
    combined.attrs["shape_start_km"] = {
        key: value
        for frame in frames
        for key, value in frame.attrs.get("shape_start_km", {}).items()
    }
    combined.attrs["shape_replacement_speed_kmh"] = {
        key: value
        for frame in frames
        for key, value in frame.attrs.get("shape_replacement_speed_kmh", {}).items()
    }
    return combined.sort_values(
        ["shape_id", "trip_id", "stop_sequence"], kind="stable"
    ).reset_index(drop=True)


def _select_trips(
    feed: GtfsFeed,
    *,
    route_types: list[int],
    service_date: object,
    service_ids: Sequence[str] | None,
) -> tuple[set[str], dict[str, int]]:
    """Select trips before the build, returning base trip ids and counts."""
    trips = feed.feed.trips.copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    routes = feed.feed.routes.copy()
    routes["route_id"] = routes["route_id"].astype(str)
    route_type = routes.set_index("route_id")["route_type"].to_dict()
    keep = trips["route_id"].map(route_type).isin(route_types)
    dropped_route = int(len(trips) - int(keep.sum()))
    kept_trips = trips.loc[keep].copy()
    dropped_service = 0
    if service_ids is not None or service_date is not None:
        allowed = (
            set(map(str, service_ids))
            if service_ids is not None
            else _active_service_ids(feed, service_date)
        )
        service = kept_trips["service_id"].astype(str).isin(allowed)
        dropped_service = int(len(kept_trips) - int(service.sum()))
        kept_trips = kept_trips.loc[service]
        if kept_trips.empty:
            raise ValueError(
                "no service runs for the requested "
                f"{'service_ids' if service_ids is not None else 'service_date'}"
            )
    selected = {_base_trip_id(value) for value in kept_trips["trip_id"].astype(str)}
    return selected, {
        "dropped_route_type": dropped_route,
        "dropped_service": dropped_service,
    }


def _base_trip_id(trip_id: str) -> str:
    """Strip the frequency-expansion suffix from a trip id."""
    return _FREQ_SUFFIX.sub("", trip_id)


def _active_service_ids(feed: GtfsFeed, service_date: object) -> set[str]:
    """Return service ids active on a date from calendar and exceptions."""
    parsed = parse_service_date(service_date)  # type: ignore[arg-type]
    compact = parsed.strftime("%Y%m%d")
    weekday = parsed.strftime("%A").lower()
    active: set[str] = set()
    calendar = getattr(feed.feed, "calendar", None)
    if calendar is not None and not calendar.empty:
        frame = calendar.copy()
        for column in ("start_date", "end_date"):
            frame[column] = frame[column].astype(str).str.replace("-", "", regex=False)
        mask = (
            (frame[weekday].astype(str) == "1")
            & (frame["start_date"] <= compact)
            & (frame["end_date"] >= compact)
        )
        active |= set(frame.loc[mask, "service_id"].astype(str))
    exceptions = getattr(feed.feed, "calendar_dates", None)
    if exceptions is not None and not exceptions.empty:
        frame = exceptions.copy()
        frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False)
        day = frame[frame["date"] == compact]
        active |= set(day.loc[day["exception_type"] == 1, "service_id"].astype(str))
        active -= set(day.loc[day["exception_type"] == 2, "service_id"].astype(str))
    return active


def _sha256(source: str) -> str | None:
    """Return the feed digest, or None for an in-memory feed.

    A file is hashed byte-for-byte. A directory feed is hashed over its sorted
    relative file names, sizes and per-file SHA-256 values, so a same-named
    changed file invalidates resume while directory-listing order does not.
    """
    path = Path(source)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        file_digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        digest.update(file_digest.encode("ascii"))
    return digest.hexdigest()
