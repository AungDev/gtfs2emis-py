# Analysis helpers

`emis_summary()` and `emis_grid()` operate on an `EmissionResult`. They never change the result; both return new tables.

## `emis_summary`

```python
from gtfs2emis_py import emis_summary

emis_summary(result)  # pollutant x process
emis_summary(result, by=["pollutant", "time"])
emis_summary(result, by=["pollutant", "vehicle"], segment_vars=["route_id"])
```

- Dimensions always include `pollutant` and `process`.
- `by="vehicle"` adds `vehicle_type`; `by="time"` adds `hour_of_day` and
  `service_hour`.
- `hour_of_day = (from_service_day_seconds // 3600) % 24` matches R's clock
  bucket. `service_hour = from_service_day_seconds // 3600` is **not** wrapped,
  so a trip after midnight keeps counting.
- `veh_vars` accepts `vehicle_type`/`fleet_row` from the emission table;
  `segment_vars` accepts transport columns joined by `segment_id` (for example
  `route_id`, `route_type`, `shape_id`). A name already present in the emission
  table (`shape_id`, `trip_id`, `stop_sequence`) is used directly rather than
  joined twice, so all three are valid `segment_vars`.
- `from_service_day_seconds` is joined only when `by` includes `time`; a custom
  transport table without that column can still be summarized by pollutant or
  vehicle.
- `value` is summed; `unit` is the emitted value unit (`g`, `MJ`, `kJ`, `#`, `L`), not the factor unit (`g/km` etc.).

Time is taken from the segment's `from_service_day_seconds`, so the segment is
attributed to its start time.

## `emis_grid`

```python
from gtfs2emis_py import emis_grid

grid_result = emis_grid(result, grid, time_resolution="hour")
```

Each segment's emissions are split across grid cells in proportion to the
length of the segment that intersects each cell:

```
value(cell) = value(segment) * length(segment ∩ cell) / length(segment)
```

The lengths are planar `EPSG:4326` degree lengths, not geodesic metres. The
same metric is used for the segment and the overlap, so the ratio conserves
mass; project the grid and transport table to a metric CRS first if you need a
geodesic allocation.

Consequences:

- The default `on_uncovered="raise"` refuses a partial grid; `"warn"` reports the uncovered mass ratio, and `"drop"` explicitly accepts the loss. Only a full-cover grid conserves mass (subject to floating point).
- The grid must be a Polygon/MultiPolygon `GeoDataFrame` with a CRS, and its CRS
  must equal the transport table's CRS. A mismatch raises rather than silently
  reprojecting.
- `time_resolution` adds an integer `time_bucket` (`day`, `hour` or `minute`)
  from the same service-day seconds; it is not wrapped at midnight.
- `aggregate=False` returns one row per segment/cell with `segment_id`, without
  geometry.

## Memory

`to_long()` and `from_long()` copy the frame. At São Paulo scale the long table
is segments × vehicles × pollutants × processes, so aggregate with
`emis_summary()`/`emis_grid()` before materialising it rather than concatenating
per-shape tables in memory. `from_long()` rebuilds a result for tests and for
round-tripping a written output; it does not restore the applied `ef` table.
