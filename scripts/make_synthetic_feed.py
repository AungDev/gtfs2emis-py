#!/usr/bin/env python3
"""Generate the synthetic GTFS fixture used by the transport-model tests.

The three real sample feeds shipped with the R package cover normal operation,
not the edge cases the contract requires a rule for. This fixture
contains those cases in one small feed:

- a shape with a single point (``SH_ONE``)
- a shape with duplicate consecutive points (``SH_DUP``)
- a loop shape whose first and last point coincide (``SH_LOOP``)
- times after ``24:00:00`` on the service day (``T_LATE``)
- no ``shape_dist_traveled`` column in any table
- a ``frequencies.txt``-driven trip (``T_FREQ``)

Regenerating the file reproduces it byte for byte: rows are written in a fixed
order, with ``\\n`` line endings and a fixed zip timestamp, so
``tests/test_fixtures.py`` can compare hashes instead of content.

Usage:
    python scripts/make_synthetic_feed.py [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "synthetic_edge_cases.zip"

# Fixed zip metadata: a real timestamp changes the archive bytes on every run
# and would turn the determinism test into a clock test.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

# table name -> (header, rows). Every table is written in this order.
TABLES: dict[str, tuple[list[str], list[list[str]]]] = {
    "agency.txt": (
        ["agency_id", "agency_name", "agency_url", "agency_timezone"],
        [["AG", "Synthetic Transit", "https://example.org", "UTC"]],
    ),
    "stops.txt": (
        ["stop_id", "stop_name", "stop_lat", "stop_lon"],
        [
            ["S1", "First", "0.0000", "0.0000"],
            ["S2", "Second", "0.0050", "0.0000"],
            ["S3", "Third", "0.0100", "0.0000"],
            ["S4", "Fourth", "0.0100", "0.0050"],
        ],
    ),
    "routes.txt": (
        ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
        [["R1", "AG", "1", "Synthetic line", "3"]],
    ),
    # No shape_dist_traveled column: the transport model has to derive distance
    # from the geometry.
    "shapes.txt": (
        ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        [
            ["SH_ONE", "0.0000", "0.0000", "1"],
            ["SH_DUP", "0.0000", "0.0000", "1"],
            ["SH_DUP", "0.0050", "0.0000", "2"],
            ["SH_DUP", "0.0050", "0.0000", "3"],
            ["SH_DUP", "0.0100", "0.0000", "4"],
            ["SH_LOOP", "0.0000", "0.0000", "1"],
            ["SH_LOOP", "0.0020", "0.0020", "2"],
            ["SH_LOOP", "0.0000", "0.0000", "3"],
            ["SH_NORMAL", "0.0000", "0.0000", "1"],
            ["SH_NORMAL", "0.0050", "0.0000", "2"],
            ["SH_NORMAL", "0.0100", "0.0000", "3"],
            ["SH_NORMAL", "0.0100", "0.0050", "4"],
        ],
    ),
    "trips.txt": (
        ["route_id", "service_id", "trip_id", "shape_id"],
        [
            ["R1", "WK", "T_ONEPOINT", "SH_ONE"],
            ["R1", "WK", "T_DUP", "SH_DUP"],
            ["R1", "WK", "T_LOOP", "SH_LOOP"],
            ["R1", "WK", "T_LATE", "SH_NORMAL"],
            ["R1", "WK", "T_FREQ", "SH_NORMAL"],
        ],
    ),
    "stop_times.txt": (
        ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
        [
            ["T_ONEPOINT", "08:00:00", "08:00:00", "S1", "1"],
            ["T_ONEPOINT", "08:05:00", "08:05:00", "S2", "2"],
            ["T_DUP", "08:10:00", "08:10:00", "S1", "1"],
            ["T_DUP", "08:15:00", "08:15:00", "S2", "2"],
            ["T_DUP", "08:20:00", "08:20:00", "S3", "3"],
            ["T_LOOP", "08:30:00", "08:30:00", "S1", "1"],
            ["T_LOOP", "08:34:00", "08:34:00", "S2", "2"],
            ["T_LOOP", "08:40:00", "08:40:00", "S1", "3"],
            # > 24:00:00: the after-midnight service day keeps on its
            # own day instead of wrapping to 01:10.
            ["T_LATE", "25:10:00", "25:10:00", "S1", "1"],
            ["T_LATE", "25:20:00", "25:20:00", "S2", "2"],
            ["T_FREQ", "06:00:00", "06:00:00", "S1", "1"],
            ["T_FREQ", "06:10:00", "06:10:00", "S4", "2"],
        ],
    ),
    "calendar.txt": (
        [
            "service_id",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "start_date",
            "end_date",
        ],
        [["WK", "1", "1", "1", "1", "1", "1", "1", "20260101", "20261231"]],
    ),
    "frequencies.txt": (
        ["trip_id", "start_time", "end_time", "headway_secs"],
        [["T_FREQ", "06:00:00", "08:00:00", "600"]],
    ),
}


def build_archive(path: Path) -> None:
    """Write the synthetic feed to ``path`` as a deterministic zip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(TABLES):
            header, rows = TABLES[name]
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the committed synthetic GTFS fixtures."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output zip path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    build_archive(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
