# Emission model

`emission_model()` is the second public entry point. It
combines a transport table, a validated fleet and one emission-factor model
into a typed `EmissionResult`.

```python
from gtfs2emis_py import emission_model, load_fleet, transport_model

tp = transport_model("bra_cur_gtfs.zip")
fleet = load_fleet("bra_cur_fleet.txt", ef_model="ef_brazil_cetesb")
result = emission_model(tp, "ef_brazil_cetesb", fleet, ["CO", "NOx"])
result.to_long()  # segment x vehicle x pollutant x process, grams
result.to_frame()  # wide per-segment table
```

## Supported models

| `ef_model` | Fleet keys | Notes |
|---|---|---|
| `ef_brazil_cetesb` | `type_name_br`, `model_year`, `fuel` | constant factor per vehicle |
| `ef_europe_emep` | `veh_type`, `euro`, `fuel`, `tech` | speed-dependent, optional DEM slope |
| `ef_usa_emfac` | `model_year`, `fuel` | `reference_year`, right-closed speed bins |
| `ef_usa_moves` | `model_year`, `fuel` | `reference_year`, right-closed speed bins |
| `ef_brazil_scaled_euro` | CETESB keys plus `type_name_eu`, `euro`, `tech` | CETESB local factor scaled by the EMEP curve |

`model_year` must be an integer year; fractional values are rejected by `load_fleet()` rather than truncated. The per-model valid ranges are applied by the EF port.

A per-vehicle fleet (a `shape_id` or `route_id` column) is joined to the
transport table by that key; an average fleet applies to the whole feed.

## Scope

- `process` accepts `hot_exhaust` and the EMEP wear processes `tyre`, `brake`
  and `road`. Wear is EMEP-only; other models raise a clear error. Wear
  pollutants are `TSP`, `PM10`, `PM2.5`, `PM1.0` and `PM0.1`, except road wear,
  which rejects `PM1.0`/`PM0.1`. Wear values are grams, like the reference.
- `parallel` and `ncores` are validated per ; the emission path consumes
  the already-built transport table, so it does not split by shape. Transport
  parallelisation is in `transport_model()`.
- `output_path` writes one Parquet file per shape plus a manifest;
  `resume=True` verifies the request fingerprint, row counts and file hashes
  before reusing a run.
- `emission_model()` has no `quiet` argument; warnings and validation failures
  are always surfaced.
- Zero-distance segments contribute exactly zero and are omitted from the
  result; this matters for feeds with duplicate stops at the same coordinate.
- `heightfile` is only accepted for EMEP-based models and adds
  `slope_class_europe_emep()`'s `slope` column before the lookup. Raw slopes are
  bucketed by R's `fcase` boundaries; values above `0.07` map to the final
  `-0.06` class (not `+0.06`), and the exact `0.07` boundary remains `+0.06`.

## Result

`EmissionResult` has `emi`, `ef`, `tp_model`, `metadata` and `units`.
`emi` columns: `segment_id`, `shape_id`, `trip_id`, `stop_sequence`,
`vehicle_type`, `fleet_row`, `pollutant`, `process`, `value`, `unit`.
`to_long()` returns a deep copy; `to_frame()` pivots to
`<pollutant>_<process>` columns. `to_dataframe()` and `to_pandas()` alias
`to_frame()`, and `emis_to_dt()` aliases `to_long()`.

## Known data limits

- The EMFAC table has no 2005 model year for every reference year, so a fleet
  containing 2005 raises `MissingEmissionFactorError` instead of inventing a
  factor. Use MOVES for that fleet or a reference year that covers it.
- EMEP speeds are rounded to whole km/h exactly as the R reference does; the
  segment join uses that same rounded key.
