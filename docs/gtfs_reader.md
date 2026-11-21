# GTFS reader (`gtfs2emis_py.gtfs.reader`)

The unified read/validate entry point that the rest of the pipeline (trajectory
interpolation, speed calculation, parallel processing) builds on. It wraps
[`gtfs-kit`](https://gitlab.com/mrcagney/gtfs_kit) and adds the structural
validation that `gtfs-kit` deliberately omits — `assess_quality` "is not a GTFS
validator".

## Why this wrapper exists

The R `gtfs2emis` package calls `gtfstools::read_gtfs()` directly inside
`transport_model()`, accepting either a path or an already-read list of
`data.table`s. `gtfs-kit` provides the equivalent I/O via `gtfs_kit.read_feed`,
but ships no validator. This module closes that gap by:

1. Reading a feed from a path (zip or directory) **or** accepting an
   already-loaded `gtfs_kit.Feed` (mirroring R's `is.character(gtfs_data)`
   branch).
2. Validating that the feed has the tables and columns the transport model
   needs (`shapes`, `trips`, `stop_times`, `stops`, …) and that trips
   reference existing shapes.
3. Optionally cleaning the feed (`Feed.clean`) and normalizing distance units
   to `km` — the unit the rest of the pipeline standardizes on (R uses
   `units::set_units(..., "km")` throughout).

Downstream phases should call `read_gtfs` once and pass the resulting
`GtfsFeed` forward, rather than re-reading or re-validating.

## Quick start

### Read from a zip path

```python
from gtfs2emis_py.gtfs import read_gtfs

gf = read_gtfs("tests/data/usa_det_gtfs.zip")
assert gf.ok
print(gf.dist_units)  # 'km'
```

### Read from an in-memory `gtfs_kit.Feed`

Useful when the feed is constructed or mutated in code rather than read from
disk — mirrors R's acceptance of either a path or an in-memory feed list:

```python
import gtfs_kit as gk
from gtfs2emis_py.gtfs import read_gtfs

feed = gk.read_feed("tests/data/usa_det_gtfs.zip", dist_units="km")
gf = read_gtfs(feed)
```

The caller's `Feed` is copied, so the original reference is not mutated by the
wrapper's `clean` / `convert_dist` steps.

### Inspect a broken feed without raising

By default `read_gtfs` raises `GtfsValidationError` on hard failures. Pass
`validate=False` to skip validation entirely when you only need the data:

```python
from gtfs2emis_py.gtfs import read_gtfs

gf = read_gtfs("maybe_broken.zip", validate=False)
assert gf.validation is None  # nothing was checked
assert gf.ok is None  # not True: an unchecked feed is not a passing feed
print(len(gf.feed.trips))
```

The wrapper records "not checked" as `None` rather than an empty, passing
`ValidationResult`, so a skipped validation can never be mistaken for a good
feed. Call `validate_gtfs(gf.feed)` yourself when you want the result later.

## Validation model

`validate_gtfs` checks only the subset of structural properties that
`transport_model` and the emission pipeline depend on — it is **not** a full
GTFS Schedule spec validator (use the [Canonical GTFS
Validator](https://gtfs-validator.mobilitydata.org/) for that).

Hard failures (collected in `ValidationResult.errors`) block processing and
trigger `GtfsValidationError`:

| Check | Failure mode |
|---|---|
| Required tables present & non-empty | `stops`, `routes`, `trips`, `stop_times`, `shapes` |
| Required columns present | Spec-required columns on each table above, including `trips.service_id` and `stop_times.arrival_time`/`departure_time` |
| Trip endpoint times | The first or last stop_time of a trip has no `arrival_time`/`departure_time` — a trajectory cannot be anchored |

ID columns (`stop_id`, `route_id`, `trip_id`, `service_id`, `shape_id`, …) are
normalised to pandas `string` dtype before the checks run, in place on the
`Feed`. GTFS IDs are opaque text; a feed that writes `shape_id` as an integer in
`trips` and as a string in `shapes` used to produce spurious "missing shape"
warnings. `validate_gtfs(gf.feed)` applies the same normalisation for feeds read
outside this wrapper.

Soft issues (collected in `ValidationResult.warnings`) do not block processing
but may degrade output quality:

| Check | Warning |
|---|---|
| `trips.route_id` → `routes.route_id` | Trips reference missing routes |
| `trips.shape_id` → `shapes.shape_id` | Trips reference missing shapes, or have no `shape_id` |
| `stop_times.trip_id` → `trips.trip_id` | Stop times reference missing trips |
| `stop_times.stop_id` → `stops.stop_id` | Stop times reference missing stops |
| Service calendar | Both `calendar` and `calendar_dates` are empty |

Only a non-empty `errors` list blocks processing; `warnings` are tolerated.

Blank times at intermediate stops are allowed: GTFS treats those values as
conditionally required, and the transport model interpolates between the
timed endpoints.

## The `GtfsFeed` wrapper

`read_gtfs` returns a `GtfsFeed`, a thin typed wrapper around `gtfs_kit.Feed`.
Pass this object forward through the pipeline instead of a raw `Feed` — it carries
the captured validation result and guarantees normalized distance units.

| Attribute / method | Purpose |
|---|---|
| `gf.feed` | The underlying cleaned, distance-normalized, ID-normalized `gtfs_kit.Feed` |
| `gf.validation` | `ValidationResult` captured at read time, or `None` when `validate=False` |
| `gf.clean_report` | Per-table row-count drops caused by `clean=True`; empty when cleaning changed nothing |
| `gf.source` | Original source string (path/URL, or `"<gtfs_kit.Feed>"`) |
| `gf.ok` | `True`/`False` from validation; `None` when validation was skipped |
| `gf.dist_units` | Distance units: `'km'` after a default read, otherwise the requested unit |
| `gf.restrict_to_dates(dates)` | New `GtfsFeed` restricted to date(s); re-validates, raising on failure unless `validate=False` |
| `gf.restrict_to_routes(route_ids)` | New `GtfsFeed` restricted to routes; same failure policy |
| `gf.assess_quality()` | `gtfs-kit`'s heuristic quality DataFrame (distinct from `validation`) |

`restrict_to_dates` and `restrict_to_routes` re-validate because restricting
can drop tables/rows and surface previously-hidden referential issues. They
apply the same failure policy as `read_gtfs`: a restriction that empties a
required table raises `GtfsValidationError`, and `validate=False` returns the
tri-state `validation=None` / `ok=None` for a caller that wants to inspect the
result instead.

```python
gf = read_gtfs("tests/data/usa_det_gtfs.zip")
tuesday = gf.restrict_to_dates("20220705")  # a Tuesday within the feed's calendar
assert tuesday.ok
```

## Read options

`read_gtfs` accepts these keyword-only flags:

| Option | Default | Effect |
|---|---|---|
| `dist_units` | `"km"` | Normalize distances; one of `'km'`, `'m'`, `'mi'`, `'ft'`. `'km'` is the pipeline default, not a guarantee when another unit is requested |
| `clean` | `True` | Apply `Feed.clean` — strips whitespace, drops zombie trips/shapes; per-table row drops are recorded in `GtfsFeed.clean_report` and `ValidationResult.warnings` |
| `validate` | `True` | Run `validate_gtfs` and raise `GtfsValidationError` on hard failures; `False` skips validation and leaves `validation`/`ok` as `None` |
| `drop_invalid_columns` | `False` | Drop non-spec columns via `Feed.drop_invalid_columns` |

Errors raised at read time:

| Exception | Raised when |
|---|---|
| `GtfsReadError` | The source cannot be read or decoded — missing path/URL, `OSError`, corrupt zip. The original exception is chained as `__cause__` |
| `GtfsValidationError` | The feed reads but fails the hard structural checks above |

> **Caveat — `clean=True` can still change what is validated.** `Feed.clean`
> runs *before* validation and may drop rows (e.g. trips referencing a missing
> shape get dropped as zombies), leaving a "valid" but smaller feed. The reader
> now records each positive row drop in `GtfsFeed.clean_report` and appends a
> `clean=True dropped rows: …` warning to `ValidationResult.warnings`. To
> inspect the raw table counts, pass `clean=False, validate=False`.

## API reference

Full parameter/return/raises documentation is in the docstrings:

- [`read_gtfs`](../src/gtfs2emis_py/gtfs/reader.py) — unified read/validate entry point
- [`validate_gtfs`](../src/gtfs2emis_py/gtfs/reader.py) — structural validation
- [`normalise_id_columns`](../src/gtfs2emis_py/gtfs/reader.py) — ID dtype normalisation
- [`GtfsFeed`](../src/gtfs2emis_py/gtfs/reader.py) — typed wrapper passed forward
- [`ValidationResult`](../src/gtfs2emis_py/gtfs/reader.py) — errors/warnings outcome
- [`GtfsReadError`](../src/gtfs2emis_py/gtfs/reader.py) — unreadable source
- [`GtfsValidationError`](../src/gtfs2emis_py/gtfs/reader.py) — distinct error type
