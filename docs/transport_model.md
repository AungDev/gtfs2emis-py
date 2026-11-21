# transport_model()

`transport_model()` turns a GTFS feed into the stop-to-stop table every emission
calculation consumes. It is the Python counterpart of
`gtfs2emis::transport_model()`. The schema and rules are described here and
in the API reference.

```python
from gtfs2emis_py import transport_model

model = transport_model("gtfs.zip", route_types=(3,), spatial_resolution=100)
model.table  # GeoDataFrame, 16 columns
model.manifest  # provenance, counts, skipped counters
```

## Defaults

| Parameter | Default | Notes |
|---|---|---|
| `min_speed` / `max_speed` | `2` / `80` km/h | R's bounds; invalid speeds are replaced, not clamped, by `adjust_speeds()` |
| `new_speed` | `None` | arithmetic mean of valid speeds weighted by distance; scheduled dwell is not carried into the adjusted timetable (R parity) |
| `cumtime_origin` | `"first_stop"` | `"shape_start"` adds R's leading shape-start interval to `cumtime_s` |
| `spatial_resolution` | `100` m | maximum spherical edge length between densified vertices |
| `route_types` | `(3, 700…799)` | bus-only by default |
| `service_date` / `service_ids` | `None` | mutually exclusive; `service_date` accepts an ISO date or `date` object (`YYYY-MM-DD` and `YYYYMMDD` are both accepted); no date selection means every trip, as in R |
| `parallel` / `ncores` | `True` / `None` | `ncores=None` → `max(1, cpu_count() - 1)`; explicit values accept `1..cpu_count()`; true splits by shape through `map_shapes()`, false or `ncores=1` runs serially |
| `output_path` / `resume` | `None` / `False` | see below |

## `cumtime_s` origin

By default `cumtime_s` starts at the first stop-to-stop interval. That is the
physically meaningful origin: a GTFS shape often begins before the first stop
(depot pull-out, layover, or simply the line extending past the terminal), and
there is no timestamp proving the vehicle covered that distance in service.
R instead keeps a leading `shape start → first stop` row, so its absolute
`cumtime` includes that unobserved travel at the shape's replacement speed.

`cumtime_origin="shape_start"` reproduces R's absolute values. It is opt-in
because it makes the first row's `cumtime_s` include travel *before* its
`from_stop_id`. The default is the correct definition; the option exists for
row-for-row comparisons with R:

```python
model = transport_model("gtfs.zip", cumtime_origin="shape_start")
```

Only `cumtime_s` changes. `from_service_day_seconds`, `to_service_day_seconds`
and the timestamps are identical for both origins: they start at the first stop
and are not shifted by the unobserved leading interval.

## Output modes

- **In memory** (`output_path=None`): returns `TransportModel(table, manifest)`.
  `table` is the 16-column GeoDataFrame from contract §4; `manifest` is a
  `TransportManifest` with the run parameters, counts and skipped counters.
- **On disk** (`output_path=<existing dir>`): writes one GeoParquet file per
  shape under `shapes/` plus `manifest.json` (see [`transport_output.md`](transport_output.md)), then returns a
  `TransportManifest` whose `shapes` entries name the files.

`resume=True` without `output_path` raises `ValueError`. A shape that fails is
recorded in `manifest.errors`; the rest of the run completes.

`manifest.parameters` records the effective run settings: `min_speed`,
`max_speed`, `new_speed`, `cumtime_origin`, `spatial_resolution`,
`route_types`, `service_date`, `service_ids`, `parallel`, `ncores`,
`output_path`, `resume` and the input `gtfs_sha256`. If no shape
produces rows, `TransportModelError` is raised instead of returning an empty
table.

## Divergences from R

| R | Python | Why |
|---|---|---|
| returns `sf` / `NULL` | `TransportModel` / `TransportManifest` | typed result carries provenance |
| `continue` argument | `resume` | `continue` is a Python keyword |
| `parallel` (deprecated) | `parallel` bool + `ncores` | R deprecated its flag |
| timestamps wrap at 24:00 | service-day seconds may exceed 86 400 | service-day seconds are canonical; datetimes only with `service_date` |
| zero-duration speed is `Inf` | `0.0` | an emission calculation must not receive `Inf` |
| no date/route filtering | `route_types`, `service_date`, `service_ids` | bus-only package with a service selection |

Parity against R's values is owned by the transport parity suite,
which runs the committed R fixtures.
