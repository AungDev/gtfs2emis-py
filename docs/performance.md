# Performance

`benchmarks/benchmark.py` measures the hot paths on a synthetic transport table
(100 stops per shape, one-tenth of a kilometre per segment). Run:

```bash
cd gtfs2emis-py
uv run python benchmarks/benchmark.py --segments 20000
uv run python benchmarks/benchmark.py --segments 5000 --check   # CI budget gate
```

`--check` compares wall time and parent peak RSS with `benchmarks/budgets.json`
and exits non-zero when a budget is exceeded. CI runs the 5,000-segment size.

## Measured (Linux, 8 logical cores, Python 3.12)

| Path | 5,000 segments | 20,000 segments |
|---|---:|---:|
| `adjust_speeds` | 157 ms | 576 ms |
| EMEP EF lookup | 94 ms | 109 ms |
| `multiply_ef` (one pollutant) | 6.1 ms | 11.5 ms |
| `map_shapes` (loky, cpu_count - 1) | 1,595 ms | 1,583 ms |
| `map_shapes` (serial) | 2.1 ms | 6.9 ms |
| parent peak RSS | 218 MB | 238 MB |

The `map_shapes` rows use a pure-Python 200-value payload, so the parallel row is
dominated by loky worker start-up; that microbenchmark is not the transport
path measurement.

## Coverage mode

The CI coverage job sets `GTFS2EMIS_COVERAGE_SERIAL=1`, which forces the
transport build through the serial path. Coverage does not follow loky child
processes, so without this switch `interpolation.py` reports different
percentages on 2-core and 8-core hosts; the serial run measures one stable
program and still covers the numerical core. Production runs are unchanged and
keep `parallel=True`.

## Transport serial vs parallel

After the payload change, each task carries only that shape's trips, stop times,
stops, routes, agency, shape and frequencies — not the whole feed. Measured with
`ncores=2` on Linux, 8 logical cores, Python 3.12:

| Feed | Serial | Parallel | Serial / parallel |
|---|---:|---:|---:|
| Detroit (632 rows) | 0.62 s | 0.97 s | 0.64× |
| Synthetic 5,000 segments | 0.64 s | 1.69 s | 0.38× |
| Synthetic 20,000 segments | 2.44 s | 1.68 s | 1.45× |
| Synthetic 50,000 segments | 6.11 s | 4.52 s | 1.35× |

The crossover is roughly 15,000 segments: below it, worker start-up costs more
than the build. `parallel=True` stays the default for signature parity with R
and because city-scale feeds win; callers building small feeds should pass
`parallel=False`. The old whole-feed payload made 20,000 and 50,000 segments
slower in parallel; those measurements are superseded by this table.

## Budgets

`benchmarks/budgets.json` sets deliberately generous ceilings for CI:
`adjust_speeds` 5 s, EMEP lookup 10 s, `multiply_ef` 2 s, either map shape 30 s,
and 2 GB parent peak RSS at 5,000 segments. A regression beyond these values
fails the CI job; tighten them only with a matching performance change.

## Memory

Peak RSS is measured with `resource.getrusage(RUSAGE_SELF)`, so it covers the
parent process only. Worker memory is additive under loky; the current paths do
not process the full feed in workers, and the emission table is built in the
parent. For a São Paulo-scale run, aggregate with `emis_summary()`/`emis_grid()`
before materialising the long table (see `analysis.md`).

## Reproducing a city-scale run

The São Paulo feed is not bundled. Use the same script with
`--segments` set to the segment count of your transport table, or run
`transport_model()` on the feed and feed its table into the emission model. The
script's synthetic table is a stand-in for timing, not an emissions result.

## Pipeline benchmark

`benchmarks/pipeline_benchmark.py` generates an in-memory city-scale GTFS feed
(100 shapes × 1,000 stops) and runs `transport_model` → `emission_model` →
per-shape writer. Measured on the development host (Python 3.12, 8 logical
cores):

| Stage | Time | Rows / output |
|---|---:|---:|
| `transport_model(parallel=True, ncores=2)` | 29.6 s | 99,900 segments |
| `emission_model` (10-row EMEP fleet, CO + NOx) | 43.8 s | 1,996,000 rows |
| `write_emission_output` | 5.8 s | 100 shapes, 12.6 MB |
| parent peak RSS | — | 974 MB |

`benchmarks/pipeline_budgets.json` stores ceilings of 600 s per stage, 6 GB RSS
and 500 MB output. The measured run is within budget, so no chunking change is
required at this scale. The component table above remains the fast CI gate;
the pipeline run is the scheduled city-scale check.
