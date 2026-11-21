# Core dependencies

Core dependencies pinned in `gtfs2emis-py/pyproject.toml`. Versions are locked in
`gtfs2emis-py/uv.lock`; CI installs from the lock and also runs the suite against
the declared minimums.

| Python dependencies | Purpose |
|---|---|
| `gtfs-kit>=13.0.1,<14` | GTFS I/O (interpolation built in-house — no Python `gtfs2gps`). 13.0.1 is the first release of the 13 line; `>=10` spanned four majors. |
| `pandas>=2.2.2,<3` | Tabular data. **The upper bound is a compatibility guard:** gtfs-kit 13.0.1 declares `fare_attributes.transfer_duration` as `Int16` and pandas 3 refuses the cast that feeds need (14400, 86400), so `read_feed()` fails on the Detroit sample. Remove the bound once gtfs-kit widens the dtype. |
| `geopandas>=1.0` + `shapely>=2.0` | LineString geometries, spatial ops |
| `numpy>=1.24` | Numeric backbone (transitive but pinned) |
| `rasterio>=1.3` | DEM raster I/O for `slope_class_europe_emep` |
| `joblib>=1.3` | Per-shape parallelism |
| `pyarrow>=15` | Parquet EF tables and Parquet/GeoParquet output |

Units are carried as `float64` in documented canonical units (km, km/h, g/km, g, m, s) with checks at module boundaries — no unit library. No separate decision record is needed for this choice.
