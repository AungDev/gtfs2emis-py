# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning rule

- The version lives in two places: `gtfs2emis-py/pyproject.toml`
  (`[project].version`) and `gtfs2emis_py.__version__`. Both move in the same
  commit; `tests/test_version.py` fails when they disagree.
- A release is the pair of that commit and an annotated `vX.Y.Z` tag. The
  `Release` workflow builds the sdist and wheel from the tag, smoke-tests the
  wheel in a clean environment and uploads the artifacts.
- While the major version is 0, a minor bump (`0.y.z` → `0.(y+1).0`) may carry
  breaking changes; a patch bump is fixes only. The first stable release is
  `1.0.0`.
- Entries are written for users: what changed and why it matters, not which
  internal task closed.

## [Unreleased]

### Added

- GTFS reader wrapper with structural validation, ID normalisation, typed
  errors and a tri-state "not checked" result (`read_gtfs`, `validate_gtfs`,
  `GtfsFeed`).
- Trajectory interpolation (`build_transport_table`): geodesic densification,
  monotonic stop snapping, service-day seconds and optional timezone-aware
  timestamps, emitting the frozen transport-model schema.
- Fleet schema and validation (`load_fleet`) for average and per-vehicle
  tables across the four EF models.
- EF tables: the EMEP/EEA table ships as Parquet with a manifest and CC-BY
  licence; CETESB, EMFAC and MOVES are fetched from their original pinned
  upstream `.rda` URLs, converted with the optional `data` extra, and verified
  by SHA-256 and content hash. The code is MIT; data terms are recorded per
  source in the packaged `data/ef/manifest.json` and `data/ef/LICENSES/`.
- Unit boundary helpers and the per-pollutant unit registry.
- `transport_model()` public entry point with R's speed defaults, route-type and
  service selection, and typed `TransportModel` / `TransportManifest` results;
  on-disk runs write per-shape GeoParquet with a resumable manifest.
- Per-shape Parquet/GeoParquet writer (`write_transport_output`) with atomic
  writes, URL-quoted shape ids, resume validation and a JSON-Schema-validated
  manifest.
- Emission-factor lookups for CETESB (`ef_brazil_cetesb`), EMEP/EEA
  (`ef_europe_emep`, 7 of 20 `Function_ID` families evaluable from the R reference; the rest are upstream-aborted), EMFAC2017
  (`ef_usa_emfac`) and MOVES3 (`ef_usa_moves`).
- `emission_model()` public entry point with `multiply_ef()` arithmetic, hot-exhaust
  dispatch for all five EF models, per-vehicle fleet groups, typed
  `EmissionResult` with `to_long()` / `to_frame()` and aliases.
- Wear integration: `process="tyre"|"brake"|"road"` routes through the EMEP wear
  port, keeps hot exhaust separate, and reports grams per process and species.
  On-disk output: `write_emission_output()` reuses the transport writer's atomic
  shape files, URL-quoted names and resume checks, with a JSON-Schema-validated
  manifest and per-file SHA-256.
- Analysis helpers: `emis_summary()` by pollutant/process/time/vehicle and
  `emis_grid()` length-weighted grid allocation with conservation and CRS
  checks; `EmissionResult.from_long()` for round-trips.
- End-to-end integration tests on the bundled Curitiba, Dublin and Detroit feeds
  with exact totals pinned for regression detection.
- Benchmarks for speed adjustment, EMEP lookup, emission arithmetic and the
  loky map, with a CI budget gate and `docs/performance.md`.
- Sphinx documentation (`docs/conf.py`, MyST guides, autodoc API reference) with
  a warning-free CI build.
- Jupyter tutorials for getting started, fleets, EF lookups and wear, executed
  by `notebooks/execute.py` in CI.

### Notes

- This is a development snapshot; parity fixtures
  are still outstanding, so the package is not yet a complete emissions estimator.
