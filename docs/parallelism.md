# Parallelism and performance

`transport_model(parallel=True, ncores=None)` splits work by `shape_id` and maps
it through `gtfs2emis_py.gtfs.parallel.map_shapes()`. Set `parallel=False` or
`ncores=1` for the serial path. `ncores` is validated against
`1..cpu_count()` when parallel; `None` means `max(1, cpu_count() - 1)`.

`map_shapes()` preserves input order, returns one `None` slot for a failed task,
and exposes the failures as `ShapeFailure` entries. A transport run records
failed shapes in `TransportManifest.errors`, keeps the successful shape files
and writes the manifest before raising. A build failure with no usable frames
raises `ShapeProcessingError`; a per-shape write failure after the table exists
raises `TransportWriteError`, whose `.manifest` contains the failed entries.

The emission path consumes the already-built transport table; it does not split
by shape again. The writer emits one Parquet file per shape, so downstream work
can be resumed shape by shape.

Budgets, measured component/pipeline timings and the coverage mode
(`GTFS2EMIS_COVERAGE_SERIAL=1`) are in [Performance](performance.md). The city-scale pipeline benchmark is
`benchmarks/pipeline_benchmark.py`; the smaller component benchmark is
`benchmarks/benchmark.py`.

```python
from gtfs2emis_py import transport_model

model = transport_model("feed.zip", parallel=True, ncores=2)
```
