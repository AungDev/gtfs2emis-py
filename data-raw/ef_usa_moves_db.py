#!/usr/bin/env python3
"""Build the MOVES EF table.

Raw input: ``ef_usa_moves_db.rda`` from ``ipeaGIT/gtfs2emis`` v0.1.2
(commit ``0dbca7d``). The table has no nulls.

Usage:
    uv run --extra data python data-raw/ef_usa_moves_db.py --raw <file-or-dir> --write
    uv run python data-raw/ef_usa_moves_db.py --check-packaged
"""

from __future__ import annotations

import pyarrow as pa
from _manifest import TableSpec, main

SPEC = TableSpec(
    name="usa_moves",
    r_object="ef_usa_moves_db",
    source="EPA MOVES",
    edition="MOVES3",
    source_url=(
        "https://www.epa.gov/moves",
        "https://github.com/ipeaGIT/gtfs2emis/tree/master/data",
    ),
    licence="LICENSES/usa_moves.txt",
    raw_filename="ef_usa_moves_db.rda",
    build_script="data-raw/ef_usa_moves_db.py",
    schema=(
        ("fuel_type", pa.string(), False),
        ("reference_year", pa.int32(), False),
        ("model_year", pa.int32(), False),
        ("pollutant", pa.string(), False),
        ("lower_speed_interval", pa.float64(), False),
        ("upper_speed_interval", pa.float64(), False),
        ("ef", pa.float64(), False),
        ("source_type", pa.string(), False),
        ("id_speed", pa.int32(), False),
    ),
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
