#!/usr/bin/env python3
"""Build the EMEP/EEA EF table.

Raw input: ``ef_europe_emep_db.rda`` from ``ipeaGIT/gtfs2emis`` v0.1.2
(commit ``0dbca7d``). ``Delta`` (368) and ``Epsilon`` (494) are null where a
``Function_ID`` family does not use that coefficient; they are preserved.

Usage:
    uv run --extra data python data-raw/ef_europe_emep_db.py --raw <file-or-dir> --write
    uv run python data-raw/ef_europe_emep_db.py --check-packaged
"""

from __future__ import annotations

import pyarrow as pa
from _manifest import TableSpec, main

_FLOAT_COLUMNS = (
    "Slope",
    "Load",
    "Vmin",
    "Vmax",
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zita",
    "Hta",
    "RF",
    "k",
    "Thita",
)

SPEC = TableSpec(
    name="europe_emep",
    r_object="ef_europe_emep_db",
    source="EMEP/EEA",
    edition="EMEP/EEA 2019",
    source_url=(
        "https://www.eea.europa.eu/publications/emep-eea-guidebook-2019",
        "https://github.com/ipeaGIT/gtfs2emis/tree/master/data",
    ),
    licence="LICENSES/europe_emep.txt",
    raw_filename="ef_europe_emep_db.rda",
    build_script="data-raw/ef_europe_emep_db.py",
    schema=(
        ("Category", pa.string(), False),
        ("Fuel", pa.string(), False),
        ("Segment", pa.string(), False),
        ("Euro", pa.string(), False),
        ("Technology", pa.string(), False),
        ("Pol", pa.string(), False),
        ("Slope", pa.float64(), False),
        ("Load", pa.float64(), False),
        ("Vmin", pa.float64(), False),
        ("Vmax", pa.float64(), False),
        ("Alpha", pa.float64(), False),
        ("Beta", pa.float64(), False),
        ("Gamma", pa.float64(), False),
        ("Delta", pa.float64(), True),
        ("Epsilon", pa.float64(), True),
        ("Zita", pa.float64(), False),
        ("Hta", pa.float64(), False),
        ("RF", pa.float64(), False),
        ("Function", pa.string(), False),
        ("Function_ID", pa.string(), False),
        ("k", pa.float64(), False),
        ("Thita", pa.float64(), False),
    ),
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
