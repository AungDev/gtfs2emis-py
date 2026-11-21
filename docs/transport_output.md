# Transport output layout

`write_transport_output()` writes one file per `shape_id` plus a manifest. The
layout is deliberately small so a resumed run and a fresh run are
indistinguishable:

```
<output_path>/
  manifest.json
  shapes/
    <url-quoted shape_id>.parquet
```

## Files

- One Parquet file per shape. When the table has a `geometry` column of shapely
  geometries, each value is written as WKB and the schema carries GeoParquet
  metadata (`geo`, version 1.0.0, `primary_column: geometry`, encoding `WKB`,
  geometry types `LineString`, and a PROJJSON CRS; the default is EPSG:4326).
  WKB bytes are passed through unchanged.
- Rows inside a file are sorted by `trip_id`, `trip_number`, `stop_sequence`
  when those columns exist, so two runs produce the same order.
- `shape_id` is URL-quoted for the filename (`a/b` -> `a%2Fb.parquet`), so a
  quote can never create a directory or escape `shapes/`. Ids whose quoted form
  exceeds 120 characters keep a 96-character prefix plus a 16-hex-digit hash.
  Filenames that collide case-insensitively are rejected before anything is
  written.

Every file and the manifest are written to a `.tmp<pid>` sibling and moved into
place with `os.replace`; an interrupted run leaves either the old file or the
complete new one.

## Manifest

`manifest.json` validates against
[`docs/schemas/transport_manifest.schema.json`](schemas/transport_manifest.schema.json):

| field | meaning |
|---|---|
| `schema_version` | Currently `1`; `read_transport_manifest()` accepts only the versions in `SUPPORTED_SCHEMA_VERSIONS` and rejects a future version with the supported range in the error |
| `tool_version` | `gtfs2emis_py.__version__` |
| `created_at` | UTC ISO-8601, seconds precision |
| `input_sha256` | SHA-256 over the column list, the `parameters` JSON, the source-feed hash (when available) and the sorted `shape_id<TAB>row_count` lines |
| `table_sha256` | Canonical value hash over every shape row (typed values, nulls, geometry WKB); a changed value is a different table even when row counts match |
| `columns` | ordered column list of the input table |
| `parameters` | caller-supplied, JSON-serialisable run parameters |
| `shapes[]` | `shape_id`, `file`, `rows`, `status` (`done`/`failed`), `error` |
| `rows_total` | input rows |
| `rows_written` | rows written by this run (0 when everything resumed) |
| `shapes_resumed` | shapes skipped because their file was complete |
| `shapes_total` | number of shapes in the input |

## Resume rules

`resume=True` requires an existing manifest unless the directory is empty. The
run refuses to resume when the manifest's `schema_version`, `columns` or
`input_sha256` differs from the current request (`ManifestError`); rerun with
`resume=False` to overwrite. A changed `table_sha256` (same feed/parameters but
different values) is detected and the output is regenerated. A shape is resumed
only when its manifest entry, row count and per-file `sha256` all match.

A shape is skipped only when all four hold:

1. the manifest entry says `status: done`;
2. the Parquet file opens;
3. its row count equals the manifest's `rows`;
4. its file SHA-256 equals the manifest entry's `sha256`.

Anything else — a truncated file, a missing file, a `failed` entry — is
rewritten. A shape that fails again is recorded as `failed` with the exception
type and message, every other shape is still attempted, and the manifest is
written before `TransportWriteError` is raised. No file is ever deleted. The
successful shape files and the failed-shape entries therefore remain available
to a caller that catches the error and rereads `manifest.json`.

For a directory feed, `gtfs_sha256` is the SHA-256 of the sorted relative file
names, sizes and per-file content hashes. A changed file invalidates resume even
when its name and row count are unchanged; reordering a directory listing does
not change the digest.
