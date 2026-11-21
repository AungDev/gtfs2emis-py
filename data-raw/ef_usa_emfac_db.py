#!/usr/bin/env python3
"""Build the EMFAC EF table.

Raw input: ``ef_usa_emfac_db.rda`` from ``ipeaGIT/gtfs2emis`` v0.1.2
(commit ``0dbca7d``). The table has no nulls.

Usage:
    uv run --extra data python data-raw/ef_usa_emfac_db.py --raw <file-or-dir> --write
    uv run python data-raw/ef_usa_emfac_db.py --check-packaged
"""

from __future__ import annotations

import pyarrow as pa
from _manifest import TableSpec, main

SPEC = TableSpec(
    name="usa_emfac",
    r_object="ef_usa_emfac_db",
    source="CARB EMFAC",
    edition="EMFAC2017",
    source_url=(
        "https://ww2.arb.ca.gov/emfac",
        "https://github.com/ipeaGIT/gtfs2emis/tree/master/data",
    ),
    licence="LICENSES/usa_emfac.txt",
    raw_filename="ef_usa_emfac_db.rda",
    build_script="data-raw/ef_usa_emfac_db.py",
    schema=(
        ("reference_year", pa.int32(), False),
        ("model_year", pa.int32(), False),
        ("fuel", pa.string(), False),
        ("vkt", pa.float64(), False),
        ("ef", pa.float64(), False),
        ("pollutant", pa.string(), False),
        ("lower_speed_interval", pa.float64(), False),
        ("upper_speed_interval", pa.float64(), False),
    ),
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
