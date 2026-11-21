"""GTFS reader wrapper around ``gtfs-kit``.

This module provides :func:`read_gtfs` (and the :class:`GtfsFeed` convenience
wrapper) — the unified read/validate entry point that the rest of the pipeline
(trajectory interpolation, speed calculation, parallel processing) builds on.

The R ``gtfs2emis`` package calls ``gtfstools::read_gtfs()`` directly inside
``transport_model()``, accepting either a path to a GTFS file or an already-read
list of ``data.table``s. ``gtfs-kit`` provides the equivalent I/O via
``gtfs_kit.read_feed``, but it deliberately ships *no* GTFS validator
(``assess_quality`` "is not a GTFS validator"). This wrapper closes that gap by:

1. Reading a feed from a path (zip or directory) or accepting an already-loaded
   ``gtfs_kit.Feed`` (mirroring R's ``is.character(gtfs_data)`` branch).
2. Validating that the feed has the tables and columns the transport model
   actually needs (``shapes``, ``trips``, ``stop_times``, ``stops`` …) and that
   trips reference existing shapes.
3. Optionally cleaning the feed (``Feed.clean``) and normalizing distance units
   to ``km`` — the unit the rest of the pipeline standardizes on (R uses
   ``units::set_units(..., "km")`` throughout).

Downstream phases should call :func:`read_gtfs` once and pass the resulting
:class:`GtfsFeed` forward, rather than re-reading or re-validating.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import gtfs_kit as gk
import pandas as pd

if TYPE_CHECKING:
    from gtfs_kit import Feed

__all__ = [
    "GtfsFeed",
    "GtfsReadError",
    "GtfsValidationError",
    "ValidationResult",
    "normalise_id_columns",
    "read_gtfs",
    "validate_gtfs",
]

# Distance unit the rest of the pipeline standardizes on. R uses
# ``units::set_units(..., "km")`` for every distance/speed, so normalize feeds
# to kilometres at read time rather than scattering conversions downstream.
_DIST_UNITS_KM: Final = "km"

# GTFS tables the transport model cannot function without. ``agency`` is
# technically required by the spec but unused by gtfs2emis, so it is not listed
# here; ``shapes`` is optional in the spec but mandatory for trajectory
# interpolation, hence its inclusion.
_REQUIRED_TABLES: tuple[str, ...] = ("stops", "routes", "trips", "stop_times", "shapes")

# Columns each required table must have. This map is the single source of
# truth: ``_check_tables`` iterates exactly ``_REQUIRED_TABLES`` and looks each
# table up here, so an entry for a table that is not required would never be
# checked. ``agency``, ``calendar`` and ``calendar_dates`` are therefore absent —
# the transport model reads none of them (``calendar`` only drives a warning in
# ``_check_service_calendar``). ``trips.service_id`` and the two ``stop_times``
# time columns are included because transport_model() cannot build trajectories
# without them.
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "stops": ("stop_id", "stop_lat", "stop_lon"),
    "routes": ("route_id", "route_type"),
    "trips": ("trip_id", "route_id", "service_id"),
    "stop_times": ("trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"),
    "shapes": ("shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"),
}

# GTFS ID columns, per table. IDs are opaque text: a feed may write shape_id as
# an integer and another table may write the same value as a string, and pandas
# will then join nothing. Every column here is cast to pandas ``string`` dtype
# before validation and before any downstream join.
_ID_COLUMNS: dict[str, tuple[str, ...]] = {
    "agency": ("agency_id",),
    "stops": ("stop_id", "parent_station", "zone_id"),
    "routes": ("route_id", "agency_id"),
    "trips": ("trip_id", "route_id", "service_id", "shape_id", "block_id"),
    "stop_times": ("trip_id", "stop_id"),
    "shapes": ("shape_id",),
    "calendar": ("service_id",),
    "calendar_dates": ("service_id",),
    "transfers": (
        "from_stop_id",
        "to_stop_id",
        "from_route_id",
        "to_route_id",
        "from_trip_id",
        "to_trip_id",
    ),
}


class GtfsValidationError(ValueError):
    """Raised when a GTFS feed is missing tables/columns the transport model needs.

    Distinct from a generic ``ValueError`` so callers (and tests) can catch
    feed-shape problems specifically, separate from I/O errors raised by
    ``gtfs-kit`` itself (e.g. a missing file or bad zip).
    """


class GtfsReadError(RuntimeError):
    """Raised when a GTFS source cannot be read or decoded.

    Wraps the errors ``gtfs_kit.read_feed`` raises for an unreadable source — a
    missing path or URL, an ``OSError`` from the filesystem, or a
    ``zipfile.BadZipFile`` — so callers catch one documented type instead of
    guessing which library failed. A feed that reads but fails the structural
    checks raises :class:`GtfsValidationError` instead.
    """


@dataclass
class ValidationResult:
    """Outcome of validating a GTFS feed.

    Attributes:
        errors: Hard failures — the feed cannot be used by the transport model.
            Non-empty ``errors`` triggers a :class:`GtfsValidationError`.
        warnings: Soft issues that do not block processing but may degrade
            output quality (e.g. trips missing ``shape_id``, duplicate route
            short names). Surfaced for the caller to log or act on.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no hard errors (warnings are tolerated)."""
        return not self.errors


def _coerce_feed(source: str | Path | Feed) -> Feed:
    """Return a ``gtfs_kit.Feed`` from a path/URL or pass an existing Feed through.

    Mirrors R's ``if (is.character(gtfs_data)) read_gtfs(...) else copy(gtfs_data)``
    branch in ``transport_model()``: callers may hand us a filesystem path/URL
    *or* an already-loaded feed.
    """
    if isinstance(source, gk.Feed):
        # Copy so callers can keep mutating their original reference without
        # surprising the rest of the pipeline. R does ``data.table::copy``.
        return source.copy()
    if isinstance(source, str | Path):
        try:
            return gk.read_feed(source, dist_units=_DIST_UNITS_KM)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            # gtfs-kit raises ValueError for a path/URL it cannot open and lets
            # BadZipFile/OSError through from the archive layer. One documented
            # type is what callers can actually handle.
            raise GtfsReadError(f"Could not read GTFS feed from {source!r}: {exc}") from exc
    raise TypeError(
        "gtfs source must be a path/URL (str | Path) or a gtfs_kit.Feed, "
        f"got {type(source).__name__}"
    )


def normalise_id_columns(feed: Feed) -> None:
    """Cast every GTFS ID column present in ``feed`` to pandas ``string`` dtype.

    GTFS IDs are opaque text. Feeds in the wild are inconsistent about it — one
    feed writes ``shape_id`` as an integer in ``trips`` and as a string in
    ``shapes`` — and pandas then joins nothing, producing spurious "missing
    shape" warnings and empty trajectories downstream.

    The cast is in place on the feed's tables, idempotent, and leaves non-ID
    numeric columns (coordinates, sequences, flags) untouched. Columns absent
    from a table are skipped, so single-agency feeds without ``agency_id`` and
    optional ``parent_station``/``block_id`` fields are fine.
    """
    for table, columns in _ID_COLUMNS.items():
        df = getattr(feed, table, None)
        if df is None or df.empty:
            continue
        for column in columns:
            if column in df.columns and df[column].dtype != "string":
                df[column] = df[column].astype("string")


def _check_id_dtypes(feed: Feed, result: ValidationResult) -> None:
    """Assert that :func:`normalise_id_columns` did its job.

    This is an invariant check, not a feed-quality check: validation calls
    normalisation first, so a failure here means the cast silently did not
    happen and every later join is suspect.
    """
    wrong = [
        f"{table}.{column}"
        for table, columns in _ID_COLUMNS.items()
        for column in columns
        if (df := getattr(feed, table, None)) is not None
        and not df.empty
        and column in df.columns
        and df[column].dtype != "string"
    ]
    if wrong:
        result.errors.append(
            "GTFS ID column(s) were not normalised to pandas string dtype: "
            f"{wrong}. This is a bug in the reader, not in the feed."
        )


def _check_tables(feed: Feed, result: ValidationResult) -> None:
    """Validate presence and required columns of the GTFS tables we depend on."""
    for table in _REQUIRED_TABLES:
        df = getattr(feed, table, None)
        if df is None or df.empty:
            result.errors.append(f"Required GTFS table '{table}' is missing or empty.")
            continue
        missing = [c for c in _REQUIRED_COLUMNS.get(table, ()) if c not in df.columns]
        if missing:
            result.errors.append(f"GTFS table '{table}' is missing required column(s): {missing}.")


def _check_referential_integrity(feed: Feed, result: ValidationResult) -> None:
    """Cross-table checks the transport model relies on.

    These are the failure modes that would silently produce empty or wrong
    trajectories downstream (e.g. trips whose ``shape_id`` does not exist in
    ``shapes``), so they are worth catching at read time.
    """
    trips = feed.trips
    shapes = feed.shapes
    stop_times = feed.stop_times

    # trips.route_id -> routes.route_id
    if "route_id" in trips.columns and feed.routes is not None and not feed.routes.empty:
        known_routes = set(feed.routes["route_id"].dropna())
        orphan = set(trips["route_id"].dropna()) - known_routes
        if orphan:
            result.warnings.append(
                f"{len(orphan)} trip route_id(s) reference missing routes "
                f"(e.g. {sorted(orphan)[:3]})."
            )

    # trips.shape_id -> shapes.shape_id
    # ``_check_tables`` already failed the feed when ``shapes`` is missing or
    # empty, and validation returns before reaching this function, so there is
    # no empty-shapes branch to handle here.
    if "shape_id" in trips.columns:
        known_shapes = set(shapes["shape_id"].dropna())
        missing_shape = trips["shape_id"].notna() & ~trips["shape_id"].isin(known_shapes)
        n_missing = int(missing_shape.sum())
        if n_missing:
            result.warnings.append(
                f"{n_missing} trip(s) reference shape_ids not present in 'shapes'."
            )
        n_no_shape = int(trips["shape_id"].isna().sum())
        if n_no_shape:
            result.warnings.append(
                f"{n_no_shape} trip(s) have no shape_id; "
                "these will be skipped by the transport model."
            )

    # stop_times.trip_id -> trips.trip_id
    if "trip_id" in stop_times.columns and "trip_id" in trips.columns:
        known_trips = set(trips["trip_id"].dropna())
        orphan_st = set(stop_times["trip_id"].dropna()) - known_trips
        if orphan_st:
            result.warnings.append(
                f"{len(orphan_st)} stop_time trip_id(s) reference missing trips."
            )

    # stop_times.stop_id -> stops.stop_id
    if "stop_id" in stop_times.columns and feed.stops is not None and not feed.stops.empty:
        known_stops = set(feed.stops["stop_id"].dropna())
        orphan_stops = set(stop_times["stop_id"].dropna()) - known_stops
        if orphan_stops:
            result.warnings.append(
                f"{len(orphan_stops)} stop_time stop_id(s) reference missing stops."
            )


def _check_stop_time_values(feed: Feed, result: ValidationResult) -> None:
    """Require a time at each trip's first and last stop.

    GTFS marks ``arrival_time``/``departure_time`` conditionally required: both
    are required at the first and last stop of a trip, while intermediate stops
    may be blank and are interpolated. The transport model interpolates between
    timepoints, so a trip with no time at either endpoint has nothing to
    interpolate from and cannot be built — a hard error, not a warning.
    """
    stop_times = feed.stop_times
    time_columns = ("arrival_time", "departure_time")
    if any(column not in stop_times.columns for column in time_columns):
        return  # reported by _check_tables
    ordered = stop_times.sort_values(["trip_id", "stop_sequence"], kind="stable")
    grouped = ordered.groupby("trip_id", sort=False)
    endpoints = pd.concat([grouped.head(1), grouped.tail(1)]).drop_duplicates()
    missing = endpoints[endpoints[list(time_columns)].isna().any(axis=1)]
    if not missing.empty:
        examples = sorted(map(str, missing["trip_id"].unique()))[:3]
        result.errors.append(
            f"{len(missing)} trip endpoint stop_time(s) have a missing arrival_time or "
            f"departure_time (e.g. trips {examples}); a trajectory cannot be anchored "
            "without times at the first and last stop."
        )


def _check_service_calendar(feed: Feed, result: ValidationResult) -> None:
    """Warn if neither ``calendar`` nor ``calendar_dates`` provides any service.

    Without service definitions the transport model cannot pick active trips
    for a date, which is the common case users hit when feeds are clipped.
    """
    cal = feed.calendar
    cal_dates = feed.calendar_dates
    cal_empty = cal is None or cal.empty
    cal_dates_empty = cal_dates is None or cal_dates.empty
    if cal_empty and cal_dates_empty:
        result.warnings.append(
            "Both 'calendar' and 'calendar_dates' are empty; "
            "no service days are defined, so no trips will be active on any date."
        )


def validate_gtfs(feed: Feed) -> ValidationResult:
    """Validate that a ``gtfs_kit.Feed`` is usable by the transport model.

    This is *not* a full GTFS Schedule spec validator (use the Google/ MobilityData
    Canonical GTFS Validator for that). It checks only the subset of structural
    properties that ``transport_model`` and the emission pipeline depend on:

    - required tables (``stops``, ``routes``, ``trips``, ``stop_times``, ``shapes``)
      are present and non-empty with their required columns — including
      ``trips.service_id`` and the two ``stop_times`` time columns;
    - every trip has an arrival and a departure time at its first and last stop;
    - trips reference routes/shapes that actually exist;
    - stop_times reference trips/stops that actually exist;
    - at least one service-definition table is populated.

    ID columns in ``feed`` are normalised to pandas ``string`` dtype **in place**
    before the checks run (see :func:`normalise_id_columns`): GTFS IDs are text,
    and comparing a numeric ``shape_id`` against a string one is what produces
    spurious "missing shape" warnings. The cast is idempotent; :func:`read_gtfs`
    applies it as well, so feeds read through this module are already text.

    Hard failures are collected in ``result.errors``; soft issues go to
    ``result.warnings``. Only a non-empty ``errors`` list blocks processing.
    """
    result = ValidationResult()
    normalise_id_columns(feed)
    _check_tables(feed, result)
    # Skip the cross-table checks below if the tables themselves are bad —
    # they would just produce noisy duplicate messages.
    if result.errors:
        return result
    _check_id_dtypes(feed, result)
    _check_referential_integrity(feed, result)
    _check_stop_time_values(feed, result)
    _check_service_calendar(feed, result)
    return result


@dataclass
class GtfsFeed:
    """Thin, typed wrapper around a ``gtfs_kit.Feed``.

    Exposes the validated underlying feed via :attr:`feed` plus the
    :class:`ValidationResult` captured at read time, so downstream code does
    not need to re-validate. Pass this object forward through the pipeline instead
    of a raw ``gtfs_kit.Feed``.

    Attributes:
        feed: The underlying, cleaned, distance-normalized ``gtfs_kit.Feed``.
        validation: Validation result captured at read time, or ``None`` when the
            caller passed ``validate=False``. ``validation.ok`` is True when the
            feed passed all hard checks; ``None`` means "not checked", so an
            unchecked feed can never be mistaken for a good one.
        source: The original source (path/URL string, or ``"<Feed>"`` for an
            in-memory feed) — useful for error messages downstream.
        clean_report: Positive row-count drops per table caused by
            ``clean=True``. Empty when cleaning changed nothing.
    """

    feed: Feed
    validation: ValidationResult | None
    source: str
    clean_report: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool | None:
        """``True``/``False`` from validation, or ``None`` when it was skipped.

        A tri-state rather than ``bool`` because ``validate=False`` means the
        feed was not checked: returning ``True`` there claimed a broken feed was
        fine.
        """
        if self.validation is None:
            return None
        return self.validation.ok

    @property
    def dist_units(self) -> str:
        """Distance units of the underlying feed.

        ``'km'`` after a default read; the unit ``read_gtfs(dist_units=...)``
        requested otherwise. Kilometres are the pipeline's default, not a
        guarantee of this property.
        """
        return self.feed.dist_units

    def restrict_to_dates(self, dates: str | Sequence[str], *, validate: bool = True) -> GtfsFeed:
        """Return a new :class:`GtfsFeed` restricted to the given date(s).

        Thin pass-through to ``Feed.restrict_to_dates`` that preserves the
        wrapper and re-validates, because restricting can drop every active trip
        and with it a table the transport model requires.

        Args:
            dates: One date or a sequence of dates, in GTFS ``YYYYMMDD`` form.
            validate: If True (default), raise :class:`GtfsValidationError` when
                the restricted feed fails validation — the same policy as
                :func:`read_gtfs`. If False, skip validation and leave
                ``validation``/``ok`` as ``None``, matching ``read_gtfs``'s
                tri-state.

        Raises:
            GtfsValidationError: If ``validate`` is True and the restricted feed
                has hard structural errors.
        """
        if isinstance(dates, str):
            dates = [dates]
        new_feed = self.feed.restrict_to_dates(list(dates))
        return self._restricted(
            new_feed,
            source=f"{self.source} (restricted to {list(dates)})",
            validate=validate,
        )

    def restrict_to_routes(self, route_ids: Iterable[str], *, validate: bool = True) -> GtfsFeed:
        """Return a new :class:`GtfsFeed` restricted to the given route_ids.

        Args and failure policy are the same as :meth:`restrict_to_dates`: a
        restriction that leaves the feed structurally unusable raises
        :class:`GtfsValidationError` unless ``validate=False``.
        """
        new_feed = self.feed.restrict_to_routes(list(route_ids))
        return self._restricted(
            new_feed,
            source=f"{self.source} (restricted to routes)",
            validate=validate,
        )

    def _restricted(self, feed: Feed, *, source: str, validate: bool) -> GtfsFeed:
        """Wrap a restricted feed, applying the read-time failure policy."""
        result = validate_gtfs(feed) if validate else None
        if result is not None and not result.ok:
            raise GtfsValidationError(
                "Restricted GTFS feed failed validation:\n  - "
                + "\n  - ".join(result.errors)
                + f"\nSource: {source}"
            )
        return GtfsFeed(feed=feed, validation=result, source=source)

    def assess_quality(self) -> pd.DataFrame:
        """Return ``gtfs_kit``'s quality-indicator DataFrame for this feed.

        Distinct from :meth:`validation`: this is gtfs-kit's heuristic
        "how broken is this feed" table (duplicate route names, missing
        departure times, etc.), not the structural pass/fail check.
        """
        return gk.assess_quality(self.feed)


_CLEAN_TABLES = (
    "agency",
    "stops",
    "routes",
    "trips",
    "stop_times",
    "shapes",
    "calendar",
    "calendar_dates",
)


def _row_counts(feed: Feed) -> dict[str, int]:
    """Return row counts for the GTFS tables that ``clean`` can change."""
    counts: dict[str, int] = {}
    for name in _CLEAN_TABLES:
        table = getattr(feed, name, None)
        if table is not None:
            counts[name] = int(len(table))
    return counts


def read_gtfs(
    source: str | Path | Feed,
    *,
    dist_units: Literal["km", "m", "mi", "ft"] = _DIST_UNITS_KM,
    clean: bool = True,
    validate: bool = True,
    drop_invalid_columns: bool = False,
) -> GtfsFeed:
    """Read and (optionally) validate a GTFS feed into a :class:`GtfsFeed`.

    This is the unified entry point. It wraps ``gtfs_kit.read_feed``
    and adds the structural validation that ``gtfs-kit`` deliberately omits.

    Parameters:
        source: Path or URL to a GTFS zip/directory, or an already-loaded
            ``gtfs_kit.Feed`` (mirrors R's acceptance of either a path or an
            in-memory feed list).
        dist_units: Distance units to normalize the feed to. Defaults to
            ``'km'`` because the rest of the pipeline (mirroring R's
            ``units::set_units(..., "km")``) assumes kilometres. Must be one
            of ``gtfs_kit.constants.DIST_UNITS`` (``'km'``, ``'m'``, ``'mi'``,
            ``'ft'``). Kilometres are the default, not a guarantee: another
            value keeps the feed in that unit.
        clean: If True (default), apply ``gtfs_kit.Feed.clean`` — strips
            whitespace from IDs, normalizes times, drops zombie trips. This is
            idempotent and matches what ``gtfs2gps`` expects upstream. Positive
            row drops are returned in ``GtfsFeed.clean_report`` and appended to
            ``ValidationResult.warnings``.
        validate: If True (default), run :func:`validate_gtfs` and raise
            :class:`GtfsValidationError` on hard failures. Set to False to skip
            validation entirely: ``gf.validation`` is then ``None`` and
            ``gf.ok`` is ``None`` — not ``True`` — so an unchecked feed cannot
            be mistaken for a good one.
        drop_invalid_columns: If True, drop GTFS columns not in the spec via
            ``Feed.drop_invalid_columns``. Off by default — non-standard
            columns are harmless to the transport model and often carry useful
            metadata.

    Returns:
        A :class:`GtfsFeed` wrapping the loaded, cleaned, distance-normalized,
        ID-normalized ``gtfs_kit.Feed`` plus the captured
        :class:`ValidationResult` (``None`` when ``validate=False``) and the
        per-table ``clean_report``.

    Raises:
        GtfsReadError: If the source cannot be read or decoded (missing path,
            unreachable URL, corrupt zip).
        GtfsValidationError: If ``validate`` is True and the feed has hard
            structural errors (missing tables/columns, unusable stop times).
        TypeError: If ``source`` is neither a path/URL nor a ``gtfs_kit.Feed``.

    Example:
        >>> from gtfs2emis_py.gtfs import read_gtfs
        >>> gf = read_gtfs("tests/data/usa_det_gtfs.zip")  # doctest: +SKIP
        >>> gf.ok
        True
        >>> gf.feed.shapes is not None
        True
    """
    feed = _coerce_feed(source)

    # Normalize distance units first so every downstream distance/speed is in
    # the same units. ``convert_dist`` is a no-op when already on the target.
    if feed.dist_units != dist_units:
        feed = feed.convert_dist(dist_units)

    if drop_invalid_columns:
        feed = feed.drop_invalid_columns()

    clean_report: dict[str, int] = {}
    if clean:
        before = _row_counts(feed)
        feed = feed.clean()
        after = _row_counts(feed)
        clean_report = {
            table: before[table] - after.get(table, 0)
            for table in before
            if before[table] != after.get(table, 0)
        }

    # GTFS IDs are opaque text: normalise before validation and before any
    # downstream join. ``validate_gtfs`` repeats this call
    # idempotently for callers that use it directly.
    normalise_id_columns(feed)

    source_str = str(source) if not isinstance(source, gk.Feed) else "<gtfs_kit.Feed>"

    result = validate_gtfs(feed) if validate else None
    if result is not None and clean_report:
        dropped = ", ".join(f"{table} -{count}" for table, count in sorted(clean_report.items()))
        result.warnings.append(f"clean=True dropped rows: {dropped}")
    if result is not None and not result.ok:
        raise GtfsValidationError(
            "GTFS feed failed validation:\n  - "
            + "\n  - ".join(result.errors)
            + f"\nSource: {source_str}"
        )

    return GtfsFeed(feed=feed, validation=result, source=source_str, clean_report=clean_report)
