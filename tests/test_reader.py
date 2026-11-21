"""Unit tests for the GTFS reader wrapper.

Covers :func:`read_gtfs`, :func:`validate_gtfs`, :class:`GtfsFeed`,
:class:`ValidationResult`, and :class:`GtfsValidationError` — the unified
read/validate entry point over ``gtfs-kit``.

Tests are organized by surface area:

- ``read_gtfs``: source coercion (Feed / path / bad type), dist-unit
  normalization, ``clean`` / ``drop_invalid_columns`` flags, ``validate``
  flag, and error propagation.
- ``validate_gtfs``: required-table checks, required-column checks,
  referential-integrity warnings, and the service-calendar warning.
- ``GtfsFeed`` wrapper: ``ok`` / ``dist_units`` / ``source`` attributes,
  ``restrict_to_dates`` / ``restrict_to_routes`` re-validation, and
  ``assess_quality`` pass-through.
- ``ValidationResult``: ``ok`` semantics and errors/warnings split.
"""

from __future__ import annotations

from pathlib import Path

import gtfs_kit as gk
import pandas as pd
import pytest

from gtfs2emis_py.gtfs.reader import (
    GtfsFeed,
    GtfsReadError,
    GtfsValidationError,
    ValidationResult,
    normalise_id_columns,
    read_gtfs,
    validate_gtfs,
)
from tests._fixtures import build_minimal_feed

# ---------------------------------------------------------------------------
# read_gtfs — source coercion
# ---------------------------------------------------------------------------


class TestReadGtfsSource:
    """``read_gtfs`` accepts a Feed, a path, or raises TypeError."""

    def test_accepts_in_memory_feed(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        assert isinstance(gf, GtfsFeed)
        assert gf.ok
        assert gf.dist_units == "km"
        # In-memory feeds should not mutate the caller's reference (clean copy).
        assert minimal_feed is not gf.feed

    def test_accepts_feed_string_source_label(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        assert gf.source == "<gtfs_kit.Feed>"

    def test_accepts_zip_path(self, minimal_feed_zip: Path) -> None:
        gf = read_gtfs(minimal_feed_zip)
        assert gf.ok
        assert gf.feed.shapes is not None
        assert not gf.feed.shapes.empty
        assert gf.source == str(minimal_feed_zip)

    def test_accepts_str_path(self, minimal_feed_zip: Path) -> None:
        gf = read_gtfs(str(minimal_feed_zip))
        assert gf.ok
        assert gf.source == str(minimal_feed_zip)

    def test_rejects_bad_source_type(self) -> None:
        with pytest.raises(TypeError, match="path/URL"):
            read_gtfs(123)  # type: ignore[arg-type]

    def test_rejects_none_source(self) -> None:
        with pytest.raises(TypeError, match="path/URL"):
            read_gtfs(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# read_gtfs — dist-unit normalization
# ---------------------------------------------------------------------------


class TestReadGtfsDistUnits:
    """Distance units are normalized at read time."""

    def test_default_is_km(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        assert gf.dist_units == "km"

    def test_convert_to_metres(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed, dist_units="m")
        assert gf.dist_units == "m"

    def test_convert_to_miles(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed, dist_units="mi")
        assert gf.dist_units == "mi"

    def test_convert_is_noop_when_already_target(self, minimal_feed: gk.Feed) -> None:
        # Feed is already km; converting to km should be a no-op on units.
        gf = read_gtfs(minimal_feed, dist_units="km")
        assert gf.dist_units == "km"


# ---------------------------------------------------------------------------
# read_gtfs — clean / drop_invalid_columns flags
# ---------------------------------------------------------------------------


class TestReadGtfsCleanFlags:
    """``clean`` and ``drop_invalid_columns`` flags are honored."""

    def test_clean_default_drops_orphan_shape(self) -> None:
        # A shape that no trip references is a zombie; ``clean`` should drop it.
        feed = build_minimal_feed()
        orphan_shape = pd.DataFrame(
            [
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.0,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 0,
                },
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.001,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 1,
                },
            ]
        )
        feed.shapes = pd.concat([feed.shapes, orphan_shape], ignore_index=True)

        gf = read_gtfs(feed, clean=True)
        assert "SH_ORPHAN" not in set(gf.feed.shapes["shape_id"])
        assert "SH1" in set(gf.feed.shapes["shape_id"])
        assert gf.clean_report["shapes"] == 2
        assert any("clean=True dropped rows" in warning for warning in gf.validation.warnings)

    def test_clean_false_preserves_orphan_shape(self) -> None:
        feed = build_minimal_feed()
        orphan_shape = pd.DataFrame(
            [
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.0,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 0,
                },
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.001,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 1,
                },
            ]
        )
        feed.shapes = pd.concat([feed.shapes, orphan_shape], ignore_index=True)

        gf = read_gtfs(feed, clean=False, validate=False)
        assert "SH_ORPHAN" in set(gf.feed.shapes["shape_id"])

    def test_clean_report_is_available_without_validation(self) -> None:
        feed = build_minimal_feed()
        orphan = pd.DataFrame(
            [
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.0,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 0,
                },
                {
                    "shape_id": "SH_ORPHAN",
                    "shape_pt_lat": 0.001,
                    "shape_pt_lon": 0.0,
                    "shape_pt_sequence": 1,
                },
            ]
        )
        feed.shapes = pd.concat([feed.shapes, orphan], ignore_index=True)
        gf = read_gtfs(feed, clean=True, validate=False)
        assert gf.validation is None
        assert gf.clean_report == {"shapes": 2}

    def test_drop_invalid_columns_runs_without_error(self, minimal_feed: gk.Feed) -> None:
        # The minimal feed has only spec-valid columns, so this is a no-op,
        # but it should not raise.
        gf = read_gtfs(minimal_feed, drop_invalid_columns=True)
        assert gf.ok


# ---------------------------------------------------------------------------
# read_gtfs — validate flag and error propagation
# ---------------------------------------------------------------------------


class TestReadGtfsValidate:
    """``validate`` flag controls raise-vs-return behavior on broken feeds."""

    @staticmethod
    def _feed_missing_table(table: str) -> gk.Feed:
        feed = build_minimal_feed()
        setattr(feed, table, feed.__getattribute__(table).iloc[:0])
        return feed

    def test_validate_true_raises_on_missing_shapes(self) -> None:
        feed = self._feed_missing_table("shapes")
        with pytest.raises(GtfsValidationError, match="shapes"):
            read_gtfs(feed)

    def test_validate_false_skips_validation(self) -> None:
        # With validate=False no validation runs: the wrapper records that the
        # feed was not checked (validation is None, ok is None) instead of
        # reporting an empty, passing result.
        feed = self._feed_missing_table("shapes")
        gf = read_gtfs(feed, validate=False, clean=False)
        assert gf.validation is None
        assert gf.ok is None

    def test_validate_false_on_good_feed_is_still_unchecked(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed, validate=False)
        assert gf.validation is None
        assert gf.ok is None

    def test_error_message_lists_failures(self) -> None:
        feed = build_minimal_feed()
        feed.stops = feed.stops.drop(columns=["stop_lat"])
        with pytest.raises(GtfsValidationError) as exc_info:
            read_gtfs(feed, clean=False)
        assert "stop_lat" in str(exc_info.value)
        assert "stops" in str(exc_info.value)


# ---------------------------------------------------------------------------
# read_gtfs — unreadable sources
# ---------------------------------------------------------------------------


class TestReadGtfsErrors:
    """I/O and decoding failures surface as one documented exception type."""

    def test_corrupt_zip_raises_read_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"this is not a zip archive")
        with pytest.raises(GtfsReadError, match="Could not read GTFS feed"):
            read_gtfs(bad)

    def test_missing_path_raises_read_error(self, tmp_path: Path) -> None:
        with pytest.raises(GtfsReadError, match="Could not read GTFS feed"):
            read_gtfs(tmp_path / "does-not-exist.zip")

    def test_read_error_chains_the_original_exception(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"still not a zip")
        with pytest.raises(GtfsReadError) as exc_info:
            read_gtfs(bad)
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# validate_gtfs — required tables
# ---------------------------------------------------------------------------


class TestValidateRequiredTables:
    """Each required table must be present and non-empty."""

    @pytest.mark.parametrize(
        "table",
        ["stops", "routes", "trips", "stop_times", "shapes"],
    )
    def test_empty_required_table_is_error(self, table: str) -> None:
        feed = build_minimal_feed()
        setattr(feed, table, feed.__getattribute__(table).iloc[:0])
        result = validate_gtfs(feed)
        assert not result.ok
        assert any(table in msg for msg in result.errors)

    def test_missing_required_table_short_circuits(self) -> None:
        # When tables are bad, referential checks are skipped — only one
        # error per missing table, no noisy duplicates.
        feed = build_minimal_feed()
        feed.shapes = feed.shapes.iloc[:0]
        result = validate_gtfs(feed)
        # Exactly one error (about shapes), no referential-integrity noise.
        assert len(result.errors) == 1
        assert "shapes" in result.errors[0]


# ---------------------------------------------------------------------------
# validate_gtfs — required columns
# ---------------------------------------------------------------------------


class TestValidateRequiredColumns:
    """Required columns must be present on each required table."""

    @pytest.mark.parametrize(
        ("table", "column"),
        [
            ("stops", "stop_id"),
            ("stops", "stop_lat"),
            ("stops", "stop_lon"),
            ("routes", "route_id"),
            ("routes", "route_type"),
            ("trips", "trip_id"),
            ("trips", "route_id"),
            ("trips", "service_id"),
            ("stop_times", "trip_id"),
            ("stop_times", "stop_id"),
            ("stop_times", "stop_sequence"),
            ("stop_times", "arrival_time"),
            ("stop_times", "departure_time"),
            ("shapes", "shape_id"),
            ("shapes", "shape_pt_lat"),
            ("shapes", "shape_pt_lon"),
            ("shapes", "shape_pt_sequence"),
        ],
    )
    def test_missing_required_column_is_error(self, table: str, column: str) -> None:
        feed = build_minimal_feed()
        setattr(feed, table, feed.__getattribute__(table).drop(columns=[column]))
        result = validate_gtfs(feed)
        assert not result.ok
        assert any(column in msg for msg in result.errors)


# ---------------------------------------------------------------------------
# validate_gtfs — stop-time values
# ---------------------------------------------------------------------------


class TestValidateStopTimeValues:
    """Times are required at trip endpoints, optional at intermediate stops."""

    def test_missing_endpoint_arrival_is_error(self) -> None:
        feed = build_minimal_feed()
        last = feed.stop_times["stop_sequence"].max()
        feed.stop_times.loc[feed.stop_times["stop_sequence"] == last, "arrival_time"] = None
        result = validate_gtfs(feed)
        assert not result.ok
        assert any("arrival_time" in msg and "endpoint" in msg for msg in result.errors)

    def test_blank_intermediate_times_are_allowed(self) -> None:
        # GTFS allows empty times at intermediate stops; only the endpoints
        # need one. Re-sequence the last stop and insert a blank one before it.
        feed = build_minimal_feed()
        stop_times = feed.stop_times.copy()
        stop_times.loc[stop_times["stop_sequence"] == 2, "stop_sequence"] = 3
        intermediate = pd.DataFrame(
            [
                {
                    "trip_id": "T1",
                    "stop_id": "S1",
                    "stop_sequence": 2,
                    "arrival_time": None,
                    "departure_time": None,
                }
            ]
        )
        feed.stop_times = pd.concat([stop_times, intermediate], ignore_index=True)
        result = validate_gtfs(feed)
        assert result.ok, result.errors

    def test_valid_feed_with_endpoint_times_passes(self, minimal_feed_zip: Path) -> None:
        gf = read_gtfs(minimal_feed_zip)
        assert gf.ok

    def test_missing_time_column_is_not_reported_twice(self) -> None:
        # A missing column is a table-shape error (_check_tables); the value
        # check must stay quiet instead of duplicating it.
        from gtfs2emis_py.gtfs.reader import _check_stop_time_values

        feed = build_minimal_feed()
        feed.stop_times = feed.stop_times.drop(columns=["arrival_time"])
        result = ValidationResult()
        _check_stop_time_values(feed, result)
        assert result.ok


# ---------------------------------------------------------------------------
# validate_gtfs — referential integrity (warnings)
# ---------------------------------------------------------------------------


class TestValidateReferentialIntegrity:
    """Cross-table checks produce warnings (not errors) for orphan references."""

    def test_good_feed_has_no_warnings(self, minimal_feed: gk.Feed) -> None:
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert result.warnings == []

    def test_trip_referencing_missing_route_warns(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.trips.loc[0, "route_id"] = "R_NOPE"
        result = validate_gtfs(minimal_feed)
        assert result.ok  # warning, not error
        assert any("route_id" in w for w in result.warnings)

    def test_trip_referencing_missing_shape_warns(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.trips.loc[0, "shape_id"] = "SH_NOPE"
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert any("shape_id" in w for w in result.warnings)

    def test_stop_time_referencing_missing_trip_warns(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.stop_times.loc[0, "trip_id"] = "T_NOPE"
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert any("trip_id" in w for w in result.warnings)

    def test_stop_time_referencing_missing_stop_warns(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.stop_times.loc[0, "stop_id"] = "S_NOPE"
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert any("stop_id" in w for w in result.warnings)

    def test_trips_without_shape_id_warn(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.trips["shape_id"] = None
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert any("no shape_id" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_gtfs — ID normalisation
# ---------------------------------------------------------------------------


class TestIdNormalisation:
    """GTFS IDs are text before any join or check."""

    @staticmethod
    def _numeric_ids_feed() -> gk.Feed:
        feed = build_minimal_feed()
        # Same values in both tables, written as integers — the shape the
        # spurious "missing shape" warnings were reported with.
        feed.shapes["shape_id"] = feed.shapes["shape_id"].str.replace("SH", "", regex=False)
        feed.trips["shape_id"] = feed.trips["shape_id"].str.replace("SH", "", regex=False)
        feed.shapes["shape_id"] = feed.shapes["shape_id"].astype("int64")
        feed.trips["shape_id"] = feed.trips["shape_id"].astype("int64")
        return feed

    def test_direct_validation_accepts_numeric_ids(self) -> None:
        feed = self._numeric_ids_feed()
        result = validate_gtfs(feed)
        assert result.ok
        assert not any("shape_ids not present" in w for w in result.warnings)

    def test_validation_normalises_ids_in_place(self) -> None:
        feed = self._numeric_ids_feed()
        validate_gtfs(feed)
        for table, columns in (
            ("trips", ("trip_id", "route_id", "service_id", "shape_id")),
            ("shapes", ("shape_id",)),
            ("stops", ("stop_id",)),
            ("routes", ("route_id",)),
            ("calendar", ("service_id",)),
        ):
            frame = getattr(feed, table)
            for column in columns:
                assert frame[column].dtype == "string", f"{table}.{column}"

    def test_read_gtfs_normalises_ids(self, minimal_feed_zip: Path) -> None:
        gf = read_gtfs(minimal_feed_zip)
        assert gf.feed.trips["shape_id"].dtype == "string"
        assert gf.feed.shapes["shape_id"].dtype == "string"

    def test_non_id_columns_keep_their_dtype(self) -> None:
        feed = build_minimal_feed()
        sequence_dtype = feed.shapes["shape_pt_sequence"].dtype
        normalise_id_columns(feed)
        assert feed.shapes["shape_pt_sequence"].dtype == sequence_dtype
        assert feed.stops["stop_lat"].dtype == "float64"

    def test_normalisation_is_idempotent(self) -> None:
        feed = build_minimal_feed()
        normalise_id_columns(feed)
        once = feed.trips["trip_id"].dtype
        normalise_id_columns(feed)
        assert feed.trips["trip_id"].dtype == once

    def test_id_dtype_assertion_catches_unnormalised_feed(self) -> None:
        # The reader normalises before asserting, so this invariant only fires
        # when the cast silently failed. Exercise it directly.
        from gtfs2emis_py.gtfs.reader import _check_id_dtypes

        feed = self._numeric_ids_feed()
        result = ValidationResult()
        _check_id_dtypes(feed, result)
        assert not result.ok
        assert any("not normalised" in msg for msg in result.errors)


# ---------------------------------------------------------------------------
# validate_gtfs — service calendar
# ---------------------------------------------------------------------------


class TestValidateServiceCalendar:
    """A feed with no service definitions warns but does not error."""

    def test_empty_calendar_and_calendar_dates_warns(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.calendar = minimal_feed.calendar.iloc[:0]
        # calendar_dates is None on the minimal feed already.
        result = validate_gtfs(minimal_feed)
        assert result.ok
        assert any("calendar" in w for w in result.warnings)

    def test_populated_calendar_dates_satisfies_check(self, minimal_feed: gk.Feed) -> None:
        # Drop calendar, add calendar_dates — should not warn about service.
        minimal_feed.calendar = minimal_feed.calendar.iloc[:0]
        minimal_feed.calendar_dates = pd.DataFrame(
            [{"service_id": "WK", "date": "20220103", "exception_type": 1}]
        )
        result = validate_gtfs(minimal_feed)
        assert not any("calendar" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# GtfsFeed wrapper
# ---------------------------------------------------------------------------


class TestGtfsFeedWrapper:
    """The :class:`GtfsFeed` dataclass exposes the right attributes and methods."""

    def test_attributes(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        assert gf.feed is not None
        assert isinstance(gf.validation, ValidationResult)
        assert isinstance(gf.source, str)
        assert gf.ok is True
        assert gf.dist_units == "km"

    def test_ok_reflects_validation(self, minimal_feed: gk.Feed) -> None:
        # Break the feed post-read and re-validate to confirm ok tracks errors.
        minimal_feed.stops = minimal_feed.stops.iloc[:0]
        result = validate_gtfs(minimal_feed)
        wrapper = GtfsFeed(feed=minimal_feed, validation=result, source="test")
        assert wrapper.ok is False

    def test_restrict_to_dates_returns_new_wrapper(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        restricted = gf.restrict_to_dates("20220103")  # a Monday
        assert restricted is not gf
        assert isinstance(restricted, GtfsFeed)
        assert "restricted to" in restricted.source
        # A Monday within the calendar window should keep the trip.
        assert not restricted.feed.trips.empty

    def test_restrict_to_dates_accepts_list(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        restricted = gf.restrict_to_dates(["20220103", "20220104"])
        assert isinstance(restricted, GtfsFeed)

    def test_restrict_to_dates_empty_when_no_service_raises(self, minimal_feed: gk.Feed) -> None:
        # 20220102 is a Sunday; the minimal calendar has sunday=0, so the
        # restriction empties `trips` and the wrapper applies the read-time policy.
        gf = read_gtfs(minimal_feed)
        with pytest.raises(GtfsValidationError):
            gf.restrict_to_dates("20220102")

    def test_restrict_to_dates_validate_false_returns_unchecked_feed(
        self, minimal_feed: gk.Feed
    ) -> None:
        gf = read_gtfs(minimal_feed)
        restricted = gf.restrict_to_dates("20220102", validate=False)
        assert restricted.feed.trips.empty
        assert restricted.validation is None
        assert restricted.ok is None

    def test_restrict_to_routes_returns_new_wrapper(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        restricted = gf.restrict_to_routes(["R1"])
        assert restricted is not gf
        assert isinstance(restricted, GtfsFeed)
        assert set(restricted.feed.routes["route_id"]) == {"R1"}

    def test_restrict_to_routes_unknown_route_raises(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        with pytest.raises(GtfsValidationError):
            gf.restrict_to_routes(["R_DOES_NOT_EXIST"])

    def test_restrict_to_routes_validate_false_returns_unchecked_feed(
        self, minimal_feed: gk.Feed
    ) -> None:
        gf = read_gtfs(minimal_feed)
        restricted = gf.restrict_to_routes(["R_DOES_NOT_EXIST"], validate=False)
        assert restricted.feed.trips.empty
        assert restricted.validation is None
        assert restricted.ok is None

    def test_assess_quality_returns_dataframe(self, minimal_feed: gk.Feed) -> None:
        gf = read_gtfs(minimal_feed)
        df = gf.assess_quality()
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    """``ValidationResult.ok`` is True iff there are no errors."""

    def test_default_is_ok(self) -> None:
        result = ValidationResult()
        assert result.ok is True
        assert result.errors == []
        assert result.warnings == []

    def test_errors_make_not_ok(self) -> None:
        result = ValidationResult(errors=["bad"])
        assert result.ok is False

    def test_warnings_alone_stay_ok(self) -> None:
        result = ValidationResult(warnings=["hmm"])
        assert result.ok is True


# ---------------------------------------------------------------------------
# GtfsValidationError
# ---------------------------------------------------------------------------


class TestGtfsValidationError:
    """``GtfsValidationError`` is a ``ValueError`` subclass for targeted catching."""

    def test_is_value_error_subclass(self) -> None:
        assert issubclass(GtfsValidationError, ValueError)

    def test_catchable_as_value_error(self, minimal_feed: gk.Feed) -> None:
        minimal_feed.stops = minimal_feed.stops.iloc[:0]
        with pytest.raises(ValueError):
            read_gtfs(minimal_feed, clean=False)


# ---------------------------------------------------------------------------
# Integration: bundled Detroit sample feed (skipped if absent)
# ---------------------------------------------------------------------------


class TestDetroitSampleFeed:
    """End-to-end read of the bundled Detroit sample, when present."""

    def test_read_detroit_feed(self, det_gtfs_path: Path) -> None:
        gf = read_gtfs(det_gtfs_path)
        assert gf.ok
        assert gf.dist_units == "km"
        assert gf.feed.shapes is not None and not gf.feed.shapes.empty
        assert gf.feed.stop_times is not None and not gf.feed.stop_times.empty

    def test_validate_detroit_feed(self, det_gtfs_path: Path) -> None:
        feed = gk.read_feed(det_gtfs_path, dist_units="km")
        result = validate_gtfs(feed)
        assert result.ok, f"unexpected errors: {result.errors}"
