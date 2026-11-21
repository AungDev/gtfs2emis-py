"""Trajectory interpolation for the transport model.

This module turns a GTFS
feed into the per-interval table that contract §4 defines: one row per
consecutive stop pair per trip, with a densified WGS84 polyline, two-layer
distance (R haversine for vertex/speed, S2 for published length), service-day
seconds and speed.

What is reused and what is in-house:

- ``gtfs-kit`` supplies ``Feed.build_geometry_by_shape()`` (the shape vertex
  sequence) and ``Feed.expand_frequencies()``; a one-point shape comes back as a
  ``Point`` and is skipped, which is contract §8's short-shape rule.
- Densification, stop snapping and timestamp interpolation are in-house. The
  distance rules are contract §6: R's haversine for vertex placement/speed and
  S2 `S1ChordAngle` for the published link length, so the module does not use
  gtfs-kit's planar/UTM distance helpers.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
from shapely.geometry import LineString

from gtfs2emis_py.gtfs.reader import GtfsFeed, read_gtfs

__all__ = [
    "BUS_ROUTE_TYPES",
    "SNAP_WARN_M",
    "build_transport_table",
    "parse_service_date",
    "parse_service_day_seconds",
]

logger = logging.getLogger(__name__)

# GTFS route_type 3 (bus) plus the extended 700-799 bus classes.
# `transport_model()` exposes this as the default `route_types` selection; the
# builder processes the trips it is given and leaves route filtering to it.
BUS_ROUTE_TYPES: Final = (3, *range(700, 800))

# A stop farther than this from the densified shape is a warning, not a failure
# (contract §6).
SNAP_WARN_M: Final = 100.0

# sf::st_length() on EPSG:4326 uses the s2 great-circle length, whose mean
# Earth radius is 6,371,010 m. R's internal vertex distance uses a different
# sphere (6,378,137 m); keep both because speed and the published link length
# come from different stages of the reference pipeline.
_S2_RADIUS_M: Final = 6_371_010.0
# gtfs2gps' C++ helper hard-codes this approximation of pi/180 rather than
# calling M_PI/180. For speeds that difference is visible at six decimals.
_R_RADIAN: Final = 0.01745329251

# GTFS times are wall-clock offsets from the service-day start and may exceed
# 24:00:00. They are never wrapped.
_MAX_SERVICE_DAY_SECONDS: Final = 48 * 3600


def parse_service_day_seconds(value: str | None) -> int | None:
    """Parse a GTFS ``HH:MM:SS`` time into seconds since the service-day start.

    Args:
        value: A GTFS time such as ``"08:05:00"`` or ``"25:10:00"``; ``None`` or
            a blank string stays ``None`` (an untimed intermediate stop).

    Returns:
        Whole seconds, never wrapped at 86 400.

    Raises:
        ValueError: If the value is not ``H:MM:SS``/``HH:MM:SS`` or is outside
            the 48-hour ceiling.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if not isinstance(value, str):
        raise ValueError(f"GTFS time must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed GTFS time {value!r}; expected HH:MM:SS")
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        raise ValueError(f"malformed GTFS time {value!r}; expected HH:MM:SS") from None
    if minutes > 59 or seconds > 59:
        raise ValueError(f"malformed GTFS time {value!r}; minutes/seconds must be < 60")
    total = hours * 3600 + minutes * 60 + seconds
    if total < 0 or total >= _MAX_SERVICE_DAY_SECONDS:
        raise ValueError(f"GTFS time {value!r} is outside the 0..48 h service day")
    return total


def _densify(
    coordinates: Sequence[tuple[float, float]], resolution_m: float
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """Insert vertices exactly as lwgeom's spherical segmentizer does.

    R's ``sf::st_segmentize()`` on lon/lat calls
    ``lwgeom::st_geod_segmentize()``, which recursively halves each edge until
    its spherical length is at most the requested maximum. The recursion makes
    the number of subsegments a power of two; it is not ``ceil(distance / max)``
    and the difference changes the vertex count and therefore snapping.

    Returns ``(vertices, cumulative_speed_m, cumulative_angle_rad)``:

    - ``cumulative_speed_m`` sums R's ``rcpp_distance_haversine`` (sphere
      radius 6,378,137 m) between consecutive vertices; it is the quantity
      R's ``update_dt()`` uses to derive link speed.
    - ``cumulative_angle_rad`` sums S2's unit-sphere angles between the same
      vertices. ``sf::st_length()`` multiplies the *summed* angle by the S2
      radius, so the caller must scale after subtraction, not per segment.
    """
    vertices: list[tuple[float, float]] = []
    max_angle = resolution_m / 6_378_137.0
    for (lon1, lat1), (lon2, lat2) in zip(coordinates[:-1], coordinates[1:], strict=False):
        distance = _haversine_m(lon1, lat1, lon2, lat2)
        if not math.isfinite(distance) or distance <= 0:
            vertices.append((lon2, lat2))
            continue
        _segmentize_edge(
            (lon1, lat1),
            (lon2, lat2),
            distance / 6_378_137.0,
            max_angle,
            vertices,
        )
    vertices.append(coordinates[-1])
    # lwgeom::st_geod_segmentize() normalises every returned longitude and
    # latitude with ((value + offset) %% 360) - offset after the C++ kernel.
    # R therefore changes even the untouched original vertices in their last
    # bits; reproducing that arithmetic is what lets the cumulative values
    # land on the same side of a six-decimal boundary.
    vertices = [
        (((lon + 180.0) % 360.0) - 180.0, ((lat + 90.0) % 360.0) - 90.0) for lon, lat in vertices
    ]
    cumulative_speed = _cumulative_distances(vertices, _haversine_m)
    cumulative_angle = _cumulative_distances(vertices, _s2_angle)
    return vertices, cumulative_speed, cumulative_angle


def _segmentize_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    angle: float,
    max_angle: float,
    vertices: list[tuple[float, float]],
) -> None:
    """Append the left endpoint of each recursively halved spherical edge."""
    if angle <= max_angle:
        vertices.append(start)
        return
    mid = _spherical_midpoint(start, end)
    _segmentize_edge(start, mid, angle / 2.0, max_angle, vertices)
    _segmentize_edge(mid, end, angle / 2.0, max_angle, vertices)


def _spherical_midpoint(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float]:
    """Midpoint on the unit sphere by normalising the Cartesian midpoint."""
    lon1, lat1 = math.radians(start[0]), math.radians(start[1])
    lon2, lat2 = math.radians(end[0]), math.radians(end[1])
    x = math.cos(lat1) * math.cos(lon1) + math.cos(lat2) * math.cos(lon2)
    y = math.cos(lat1) * math.sin(lon1) + math.cos(lat2) * math.sin(lon2)
    z = math.sin(lat1) + math.sin(lat2)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        return start
    return (math.degrees(math.atan2(y / norm, x / norm)), math.degrees(math.asin(z / norm)))


def _cumulative_distances(
    vertices: Sequence[tuple[float, float]],
    distance: Callable[[float, float, float, float], float],
) -> list[float]:
    """Cumulative distance at each vertex using ``distance`` for every segment."""
    cumulative = [0.0]
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(vertices[:-1], vertices[1:], strict=False):
        total += distance(lon1, lat1, lon2, lat2)
        cumulative.append(total)
    return cumulative


def _s2_angle(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """S2's ``S1ChordAngle::radians()``, not a direct central-angle formula.

    R's s2 package accumulates ``S1ChordAngle(edge).radians()``; S1ChordAngle
    stores the squared chord length and converts it with
    ``2 * asin(0.5 * sqrt(chord2))``. The distinction is visible in a long
    cumsum at the sixth decimal.
    """
    a = _unit_vector(lon1, lat1)
    b = _unit_vector(lon2, lat2)
    chord2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return 2.0 * math.asin(0.5 * math.sqrt(chord2))


def _s2_haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """S2 distance in metres (radius 6,371,010 m)."""
    return _s2_angle(lon1, lat1, lon2, lat2) * _S2_RADIUS_M


def _unit_vector(lon: float, lat: float) -> tuple[float, float, float]:
    """Return the S2 unit vector for a lon/lat pair."""
    lon_r, lat_r = math.radians(lon), math.radians(lat)
    cos_lat = math.cos(lat_r)
    return (cos_lat * math.cos(lon_r), cos_lat * math.sin(lon_r), math.sin(lat_r))


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """R's ``distanceHaversine``: metres on a sphere of radius 6,378,137 m."""
    return _spherical_haversine_m(
        lon1, lat1, lon2, lat2, 6_378_137.0, to_rad=lambda value: value * _R_RADIAN
    )


def _spherical_haversine_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    radius: float,
    *,
    to_rad: Callable[[float], float] = math.radians,
) -> float:
    """Great-circle distance on a sphere of the given radius."""
    lat1_rad, lat2_rad = to_rad(lat1), to_rad(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = to_rad(lon2 - lon1)
    d = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    if d > 1:
        d = 1.0
    return 2 * math.atan2(math.sqrt(d), math.sqrt(1 - d)) * radius


def _snap_nearest2(
    stop_coordinates: Sequence[tuple[float, float]],
    vertices: Sequence[tuple[float, float]],
    resolution_m: float,
) -> list[int]:
    """Port of gtfs2gps ``cpp_snap_points_nearest2``.

    Scans forward from the previous vertex and accepts the first vertex whose
    distance is below the tolerance and below the next vertex's distance. If
    any stop cannot be placed, the whole shape is retried with a doubled
    tolerance, up to four levels. Returns 0-based vertex indices; an empty list
    means the shape cannot be snapped.
    """
    n_stops = len(stop_coordinates)
    n_ref = len(vertices)
    if n_ref < 2:
        return []
    for level in range(4):
        tolerance = resolution_m * (2**level)
        result: list[int] = []
        ref_index = -1
        stop_index = 0
        while stop_index < n_stops and ref_index + 2 < n_ref:
            stop_lon, stop_lat = stop_coordinates[stop_index]
            found = False
            while True:
                ref_index += 1
                ref_lon, ref_lat = vertices[ref_index]
                next_lon, next_lat = vertices[ref_index + 1]
                distance = _haversine_m(stop_lon, stop_lat, ref_lon, ref_lat)
                distance_next = _haversine_m(stop_lon, stop_lat, next_lon, next_lat)
                if distance < tolerance and distance < distance_next:
                    result.append(ref_index)
                    found = True
                if not (ref_index + 2 < n_ref and not found):
                    break
            stop_index += 1
        if len(result) < n_stops:
            # C++ returns 1-based positions and appends ``ref_nrow - 1``; in
            # this 0-based port that is the penultimate vertex, not the last.
            result.append(n_ref - 2)
        if len(result) >= n_stops:
            return result[:n_stops]
    return []


def _interpolate_times(
    times: list[int | None], cumulative_at_stop: Sequence[float], *, trip_id: str
) -> list[int]:
    """Interpolate untimed stops linearly in cumulative distance (contract §5).

    The reader guarantees the first and last stop are timed, so every missing
    stop has a timed neighbour on each side.
    """
    known = [index for index, value in enumerate(times) if value is not None]
    if not known:
        raise ValueError(f"trip {trip_id} has no timed stops; a trajectory cannot be anchored")
    filled = list(times)
    for index, value in enumerate(times):
        if value is not None:
            continue
        before = max(candidate for candidate in known if candidate < index)
        after = min(candidate for candidate in known if candidate > index)
        start_distance = cumulative_at_stop[before]
        end_distance = cumulative_at_stop[after]
        fraction = (
            0.0
            if end_distance == start_distance
            else (cumulative_at_stop[index] - start_distance) / (end_distance - start_distance)
        )
        start_time = times[before]
        end_time = times[after]
        assert start_time is not None and end_time is not None
        filled[index] = int(round(start_time + fraction * (end_time - start_time)))
    resolved: list[int] = []
    for value in filled:
        assert value is not None  # every position was either timed or filled above
        resolved.append(value)
    return resolved


def _agency_timezone(feed: gk.Feed) -> str:
    """Return the feed's single agency timezone, raising when it is absent or mixed."""
    agency = feed.agency
    if agency is None or agency.empty or "agency_timezone" not in agency.columns:
        raise ValueError("a service_date needs agency_timezone, and the feed has none")
    zones = sorted(set(agency["agency_timezone"].dropna().astype(str)))
    if len(zones) != 1:
        raise ValueError(f"a service_date needs exactly one agency_timezone; the feed has {zones}")
    return zones[0]


def parse_service_date(service_date: str | date) -> date:
    """Parse either ISO service-date form into a :class:`datetime.date`.

    ``date.fromisoformat()`` accepts the basic ``YYYYMMDD`` form only on Python
    3.11+, so this helper adds a ``strptime`` fallback and is the single parser
    used by both the builder and the service filter.

    Args:
        service_date: A ``date`` or a string in ``YYYY-MM-DD`` or ``YYYYMMDD``
            form.

    Returns:
        The parsed date.

    Raises:
        ValueError: If a string is neither accepted form or is not a real date.
        TypeError: If the value is neither a string nor a date.
    """
    if isinstance(service_date, date):
        return service_date
    if isinstance(service_date, str):
        try:
            return date.fromisoformat(service_date)
        except ValueError:
            try:
                return datetime.strptime(service_date, "%Y%m%d").date()
            except ValueError as exc:
                raise ValueError(
                    "service_date must be an ISO date (YYYY-MM-DD or YYYYMMDD), "
                    f"got {service_date!r}"
                ) from exc
    raise TypeError(
        f"service_date must be an ISO date string or date, got {type(service_date).__name__}"
    )


def _service_day_anchor(service_date: str | date, timezone: str) -> datetime:
    """Return local noon minus 12 h for the service date, in the feed's zone.

    GTFS defines the service day as noon minus 12 h, which is what keeps a DST
    transition day correct.
    """
    day = parse_service_date(service_date)
    zone = ZoneInfo(timezone)
    return datetime.combine(day, clock_time(12), tzinfo=zone) - timedelta(hours=12)


def _timestamp(anchor: datetime | None, seconds: int) -> pd.Timestamp | None:
    """Materialise a service-day second as a timezone-aware timestamp."""
    if anchor is None:
        return None
    return pd.Timestamp(anchor + timedelta(seconds=seconds))


def _coerce_feed(gtfs_data: str | Path | GtfsFeed | gk.Feed) -> gk.Feed:
    """Return a ``gtfs_kit.Feed`` from the accepted inputs, copying in-memory feeds."""
    if isinstance(gtfs_data, GtfsFeed):
        return gtfs_data.feed
    if isinstance(gtfs_data, gk.Feed):
        return gtfs_data.copy()
    return read_gtfs(gtfs_data).feed


def _vertex_speed_totals(
    cumulative: list[float],
    cumulative_at_stop: list[float],
    departures: list[int],
    arrivals: list[int],
    min_speed: float,
    max_speed: float,
) -> tuple[float, float]:
    """Return R's distance-weighted sum of valid vertex speeds and weights.

    R's ``adjust_speed()`` weights every densified vertex's speed by the
    distance since the previous vertex. The stop-to-stop table only stores
    interval averages, so reproducing its leading-``cumtime`` replacement speed
    requires the vertex-level weights used here.
    """
    if len(cumulative_at_stop) < 2:
        return 0.0, 0.0
    times: list[float] = []
    for distance in cumulative:
        if distance <= cumulative_at_stop[0]:
            times.append(float(departures[0]))
        elif distance >= cumulative_at_stop[-1]:
            times.append(float(arrivals[-1]))
        else:
            position = bisect_right(cumulative_at_stop, distance) - 1
            start = cumulative_at_stop[position]
            end = cumulative_at_stop[position + 1]
            fraction = 0.0 if end <= start else (distance - start) / (end - start)
            times.append(
                float(departures[position])
                + fraction * (float(arrivals[position + 1]) - float(departures[position]))
            )
    weighted_sum = 0.0
    total_weight = 0.0
    for position in range(1, len(cumulative)):
        duration = times[position] - times[position - 1]
        distance = cumulative[position] - cumulative[position - 1]
        if duration <= 0 or distance <= 0:
            continue
        speed = distance * 3.6 / duration
        if min_speed <= speed <= max_speed:
            weighted_sum += speed * distance
            total_weight += distance
    return weighted_sum, total_weight


def build_transport_table(
    gtfs_data: str | Path | GtfsFeed | gk.Feed,
    *,
    spatial_resolution: float = 100.0,
    service_date: str | date | None = None,
    include_shape_start: bool = False,
    min_speed: float = 2.0,
    max_speed: float = 80.0,
    trip_ids: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Build the per-interval transport table for a feed.

    Args:
        gtfs_data: A path, an already-read :class:`GtfsFeed`, or a
            ``gtfs_kit.Feed``. In-memory feeds are copied.
        spatial_resolution: Maximum spherical edge length between consecutive
            densified vertices, in metres. Must be finite and positive.
        service_date: Optional date used to materialise
            ``from_timestamp``/``to_timestamp``. Without it the columns are
            present and null (contract §5).
        include_shape_start: When true, compute the vertex-weighted replacement
            speed per shape used by ``cumtime_origin="shape_start"``. Off by
            default because it is only needed for R's absolute cumtime.
        min_speed: Lower valid speed bound for that vertex weighting, in km/h.
        max_speed: Upper valid speed bound for that vertex weighting, in km/h.
        trip_ids: Optional selected base trip ids. When supplied, only those
            trips (and their expanded frequency copies) are densified; the
            selection is applied before any stop/snap work.

    Returns:
        A :class:`geopandas.GeoDataFrame` with the 16 columns of contract §4,
        sorted by ``shape_id``, ``trip_id`` and ``stop_sequence``. Trips without
        a shape, shapes with fewer than two points and trips with fewer than two
        usable stops are skipped and logged.

    Raises:
        ValueError: For a non-positive resolution, a missing/mixed
            ``agency_timezone`` when ``service_date`` is given, a malformed
            GTFS time, or a trip with no timed stops.
        TypeError: If ``service_date`` is neither a string nor a date.
    """
    if not math.isfinite(spatial_resolution) or spatial_resolution <= 0:
        raise ValueError(
            "spatial_resolution must be a finite positive number of metres, "
            f"got {spatial_resolution!r}"
        )
    feed = _coerce_feed(gtfs_data)
    if feed.frequencies is not None and not feed.frequencies.empty:
        feed = feed.expand_frequencies()
    selected: set[str] | None = None
    if trip_ids is not None:
        selected = {str(value) for value in trip_ids}

    anchor = _service_day_anchor(service_date, _agency_timezone(feed)) if service_date else None

    stats = {
        "dropped_route_type": 0,
        "dropped_service": 0,
        "skipped_no_shape": 0,
        "skipped_short_shape": 0,
        "skipped_short_trip": 0,
        "skipped_no_timed_stops": 0,
        "zero_duration_intervals": 0,
    }
    snap_warnings: list[str] = []
    stops = feed.stops.set_index("stop_id")
    stop_times_by_trip = {
        str(trip_id): table.sort_values("stop_sequence", kind="stable")
        for trip_id, table in feed.stop_times.groupby("trip_id", sort=False)
    }
    routes = feed.routes.set_index("route_id")
    geometries = feed.build_geometry_by_shape()

    def _is_selected(trip_id: object) -> bool:
        """Return True when the trip matches the service selection."""
        if selected is None:
            return True
        text = str(trip_id)
        return text in selected or any(text.startswith(f"{base}-freq-") for base in selected)

    representatives: dict[str, tuple[tuple[str, ...], list[tuple[float, float]]]] = {}
    for shape_id, group in feed.trips.groupby("shape_id", sort=False):
        group = group[group["trip_id"].map(_is_selected)]
        candidates: list[tuple[int, tuple[str, ...], list[tuple[float, float]]]] = []
        for trip_id in group["trip_id"].astype(str):
            stop_times = stop_times_by_trip.get(trip_id)
            if stop_times is None or len(stop_times) < 2:
                continue
            coordinates_for_trip: list[tuple[float, float]] = []
            for stop_id in stop_times["stop_id"]:
                if stop_id not in stops.index:
                    coordinates_for_trip = []
                    break
                stop = stops.loc[stop_id]
                if pd.isna(stop["stop_lon"]) or pd.isna(stop["stop_lat"]):
                    coordinates_for_trip = []
                    break
                coordinates_for_trip.append((float(stop["stop_lon"]), float(stop["stop_lat"])))
            if coordinates_for_trip:
                candidates.append(
                    (
                        len(coordinates_for_trip),
                        tuple(stop_times["stop_id"].astype(str)),
                        coordinates_for_trip,
                    )
                )
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            representatives[str(shape_id)] = (best[1], best[2])

    rows: list[dict[str, object]] = []
    shape_start_km: dict[str, float] = {}
    shape_speed_sums: dict[str, list[float]] = {}
    snap_cache: dict[tuple[str, tuple[tuple[float, float], ...]], list[int]] = {}
    for _, trip in feed.trips.iterrows():
        if not _is_selected(trip.get("trip_id")):
            continue
        shape_id = trip.get("shape_id")
        if not isinstance(shape_id, str) or shape_id not in geometries:
            logger.warning("trip %s has no usable shape; skipped", trip.get("trip_id"))
            stats["skipped_no_shape"] += 1
            continue
        geometry = geometries[shape_id]
        coordinates = list(getattr(geometry, "coords", []))
        if geometry.geom_type != "LineString" or len(coordinates) < 2:
            logger.warning("shape %s has fewer than two points; its trips are skipped", shape_id)
            stats["skipped_short_shape"] += 1
            continue
        trip_id = str(trip["trip_id"])
        stop_times = stop_times_by_trip.get(trip_id)
        if stop_times is None or len(stop_times) < 2:
            logger.warning("trip %s has fewer than two stops; skipped", trip_id)
            stats["skipped_short_trip"] += 1
            continue

        route_id = trip.get("route_id")
        route_type = routes.loc[route_id, "route_type"] if route_id in routes.index else None
        if route_type is None or pd.isna(route_type):
            logger.warning("trip %s has no route_type; skipped", trip_id)
            continue

        stop_coordinates: list[tuple[float, float]] = []
        usable = True
        for stop_id in stop_times["stop_id"]:
            if stop_id not in stops.index:
                logger.warning("trip %s references unknown stop %s; skipped", trip_id, stop_id)
                stats["skipped_short_trip"] += 1
                usable = False
                break
            stop = stops.loc[stop_id]
            if pd.isna(stop["stop_lon"]) or pd.isna(stop["stop_lat"]):
                logger.warning("trip %s stop %s has no coordinates; skipped", trip_id, stop_id)
                usable = False
                break
            stop_coordinates.append((float(stop["stop_lon"]), float(stop["stop_lat"])))
        if not usable:
            continue

        vertices, cumulative_speed, cumulative_angle = _densify(coordinates, spatial_resolution)
        representative = representatives.get(shape_id)
        trip_stop_ids = tuple(stop_times["stop_id"].astype(str))
        snap_coordinates = (
            representative[1]
            if representative is not None and representative[0] == trip_stop_ids
            else stop_coordinates
        )
        cache_key = (shape_id, tuple(snap_coordinates))
        snapped = snap_cache.get(cache_key)
        if snapped is None:
            snapped = _snap_nearest2(snap_coordinates, vertices, spatial_resolution)
            if snapped:
                snap_cache[cache_key] = snapped
        if not snapped:
            logger.warning("trip %s: could not snap stops to shape %s; skipped", trip_id, shape_id)
            stats["skipped_short_trip"] += 1
            continue
        for stop, vertex_index in zip(stop_coordinates, snapped, strict=False):
            distance = _haversine_m(
                stop[0], stop[1], vertices[vertex_index][0], vertices[vertex_index][1]
            )
            if distance > SNAP_WARN_M:
                snap_warnings.append(
                    f"trip {trip_id}: stop snapped {distance:.1f} m from shape {shape_id}"
                )
        cumulative_at_stop = [cumulative_angle[index] * _S2_RADIUS_M for index in snapped]
        speed_cumulative_at_stop = [cumulative_speed[index] for index in snapped]
        shape_start_km[trip_id] = cumulative_at_stop[0] / 1000.0

        raw_departures = [
            parse_service_day_seconds(value) for value in stop_times["departure_time"]
        ]
        raw_arrivals = [parse_service_day_seconds(value) for value in stop_times["arrival_time"]]
        if not any(value is not None for value in raw_departures) and not any(
            value is not None for value in raw_arrivals
        ):
            stats["skipped_no_timed_stops"] += 1
            logger.warning("trip %s has no timed stops; skipped", trip_id)
            continue
        departures = _interpolate_times(raw_departures, cumulative_at_stop, trip_id=trip_id)
        arrivals = _interpolate_times(raw_arrivals, cumulative_at_stop, trip_id=trip_id)
        if include_shape_start:
            weighted_sum, total_weight = _vertex_speed_totals(
                cumulative_speed,
                speed_cumulative_at_stop,
                departures,
                arrivals,
                min_speed,
                max_speed,
            )
            totals = shape_speed_sums.setdefault(shape_id, [0.0, 0.0])
            totals[0] += weighted_sum
            totals[1] += total_weight

        first_departure = departures[0]
        cumulative_distance = 0.0
        stop_ids = list(stop_times["stop_id"].astype(str))
        sequences = list(stop_times["stop_sequence"])
        for position in range(len(stop_ids) - 1):
            start = snapped[position]
            end = snapped[position + 1]
            interval_vertices = vertices[start : end + 1]
            if len(interval_vertices) < 2:
                interval_vertices = [vertices[start], vertices[end]]
            geometry = LineString(interval_vertices)
            distance_km = (
                max(
                    0.0,
                    (cumulative_angle[snapped[position + 1]] - cumulative_angle[snapped[position]])
                    * _S2_RADIUS_M,
                )
                / 1000.0
            )
            cumulative_distance += distance_km
            duration = arrivals[position + 1] - departures[position]
            if duration <= 0:
                stats["zero_duration_intervals"] += 1
            speed_distance_m = max(
                0.0,
                speed_cumulative_at_stop[position + 1] - speed_cumulative_at_stop[position],
            )
            # R stores speed in m/s and converts with 3.6 at the boundary.
            # Keep that operation order: km / (h) introduces a different
            # sixth-decimal rounding.
            speed_mps = speed_distance_m / duration if duration > 0 else 0.0
            speed = 3.6 * speed_mps
            rows.append(
                {
                    "shape_id": shape_id,
                    "trip_id": trip_id,
                    "route_id": str(route_id),
                    "route_type": int(route_type),
                    "stop_sequence": int(sequences[position]),
                    "from_stop_id": stop_ids[position],
                    "to_stop_id": stop_ids[position + 1],
                    "from_service_day_seconds": departures[position],
                    "to_service_day_seconds": arrivals[position + 1],
                    "from_timestamp": _timestamp(anchor, departures[position]),
                    "to_timestamp": _timestamp(anchor, arrivals[position + 1]),
                    "dist_km": distance_km,
                    "cumdist_km": cumulative_distance,
                    "speed_kmh": speed,
                    "cumtime_s": float(arrivals[position + 1] - first_departure),
                    "geometry": geometry,
                }
            )

    if not rows:
        return _empty_table()
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if frame.empty:
        return _empty_table()
    frame = frame.astype(
        {
            "shape_id": "string",
            "trip_id": "string",
            "route_id": "string",
            "route_type": "Int16",
            "stop_sequence": "Int32",
            "from_stop_id": "string",
            "to_stop_id": "string",
            "from_service_day_seconds": "Int32",
            "to_service_day_seconds": "Int32",
            "dist_km": "float64",
            "cumdist_km": "float64",
            "speed_kmh": "float64",
            "cumtime_s": "float64",
        }
    )
    if anchor is not None:
        frame["from_timestamp"] = pd.to_datetime(frame["from_timestamp"], utc=True).dt.tz_convert(
            anchor.tzinfo
        )
        frame["to_timestamp"] = pd.to_datetime(frame["to_timestamp"], utc=True).dt.tz_convert(
            anchor.tzinfo
        )
    else:
        frame["from_timestamp"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        frame["to_timestamp"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    result = frame.sort_values(["shape_id", "trip_id", "stop_sequence"], kind="stable").reset_index(
        drop=True
    )
    result.attrs["build_stats"] = stats
    result.attrs["warnings"] = tuple(snap_warnings)
    result.attrs["shape_start_km"] = shape_start_km
    result.attrs["shape_replacement_speed_kmh"] = {
        shape: values[0] / values[1] for shape, values in shape_speed_sums.items() if values[1] > 0
    }
    return result


def _empty_table() -> gpd.GeoDataFrame:
    """Return an empty table with the contract's columns and dtypes."""
    frame = gpd.GeoDataFrame(
        {
            "shape_id": pd.Series(dtype="string"),
            "trip_id": pd.Series(dtype="string"),
            "route_id": pd.Series(dtype="string"),
            "route_type": pd.Series(dtype="Int16"),
            "stop_sequence": pd.Series(dtype="Int32"),
            "from_stop_id": pd.Series(dtype="string"),
            "to_stop_id": pd.Series(dtype="string"),
            "from_service_day_seconds": pd.Series(dtype="Int32"),
            "to_service_day_seconds": pd.Series(dtype="Int32"),
            "from_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "to_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "dist_km": pd.Series(dtype="float64"),
            "cumdist_km": pd.Series(dtype="float64"),
            "speed_kmh": pd.Series(dtype="float64"),
            "cumtime_s": pd.Series(dtype="float64"),
        },
        geometry=[],
        crs="EPSG:4326",
    )
    frame.attrs["build_stats"] = {
        "dropped_route_type": 0,
        "dropped_service": 0,
        "skipped_no_shape": 0,
        "skipped_short_shape": 0,
        "skipped_short_trip": 0,
        "skipped_no_timed_stops": 0,
        "zero_duration_intervals": 0,
    }
    frame.attrs["warnings"] = ()
    frame.attrs["shape_start_km"] = {}
    frame.attrs["shape_replacement_speed_kmh"] = {}
    return frame
