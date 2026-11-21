"""GTFS reader and transport model.

Step 2.1 — unified read/validate entry point over ``gtfs-kit``; step 2.5 — the
per-shape Parquet/GeoParquet writer and resume manifest.
"""

from gtfs2emis_py.gtfs.reader import (
    GtfsFeed,
    GtfsReadError,
    GtfsValidationError,
    ValidationResult,
    normalise_id_columns,
    read_gtfs,
    validate_gtfs,
)
from gtfs2emis_py.gtfs.slope import (
    SLOPE_CLASSES,
    SlopeWarning,
    classify_slope,
    slope_class_europe_emep,
)
from gtfs2emis_py.gtfs.writer import (
    SCHEMA_VERSION,
    ManifestError,
    TransportWriteError,
    read_transport_manifest,
    write_transport_output,
)

__all__ = [
    "SCHEMA_VERSION",
    "SLOPE_CLASSES",
    "GtfsFeed",
    "GtfsReadError",
    "GtfsValidationError",
    "ManifestError",
    "SlopeWarning",
    "TransportWriteError",
    "ValidationResult",
    "classify_slope",
    "normalise_id_columns",
    "read_gtfs",
    "read_transport_manifest",
    "slope_class_europe_emep",
    "validate_gtfs",
    "write_transport_output",
]
