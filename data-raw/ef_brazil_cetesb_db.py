#!/usr/bin/env python3
"""Build the CETESB EF table.

Raw input: ``ef_brazil_cetesb_db.rda`` from ``ipeaGIT/gtfs2emis`` v0.1.2
(commit ``0dbca7d``). The 183 nulls in each ``BUS_*`` column are the reference
data's own missing values; they are preserved, not cleaned.

Usage:
    uv run --extra data python data-raw/ef_brazil_cetesb_db.py --raw <file-or-dir> --write
    uv run python data-raw/ef_brazil_cetesb_db.py --check-packaged
"""

from __future__ import annotations

import pyarrow as pa
from _manifest import TableSpec, main

SPEC = TableSpec(
    name="brazil_cetesb",
    r_object="ef_brazil_cetesb_db",
    source="CETESB",
    edition="CETESB 2019",
    source_url=(
        "https://cetesb.sp.gov.br/veicular/",
        "https://github.com/ipeaGIT/gtfs2emis/tree/master/data",
    ),
    licence="LICENSES/brazil_cetesb.txt",
    raw_filename="ef_brazil_cetesb_db.rda",
    build_script="data-raw/ef_brazil_cetesb_db.py",
    schema=(
        ("calendar_year", pa.int32(), False),
        ("model_year", pa.int32(), False),
        ("pollutant", pa.string(), False),
        ("Proconve_HDV", pa.string(), False),
        ("Euro_EqHDV", pa.string(), False),
        ("BUS_URBAN_D", pa.float64(), True),
        ("BUS_MICRO_D", pa.float64(), True),
        ("BUS_COACH_D", pa.float64(), True),
        ("BUS_ARTIC_D", pa.float64(), True),
    ),
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
