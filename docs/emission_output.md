# Emission output

`write_emission_output()` persists an `EmissionResult` as one Parquet file per
shape plus `manifest.json`. It reuses the transport
writer's filename quoting, atomic `os.replace` writes and resume checks.

```python
from gtfs2emis_py import emission_model, transport_model

result = emission_model(
    transport_model("feed.zip"),
    "ef_brazil_cetesb",
    "fleet.csv",
    ["CO", "NOx"],
    output_path="outputs/run-01",
    resume=True,
)
```

Passing `output_path` to `emission_model()` writes the result and records the
directory and manifest in `result.metadata`. Calling the writer directly works
too:

```python
from gtfs2emis_py import write_emission_output

write_emission_output(result, "outputs/run-01", resume=True)
```

## Layout

```
outputs/run-01/
  manifest.json
  shapes/
    <url-quoted shape_id>.parquet
```

Each shape file is the long table (`segment_id`, `shape_id`, `trip_id`,
`stop_sequence`, `vehicle_type`, `pollutant`, `process`, `value`, `unit`),
sorted by trip and stop sequence. `value` is the emitted quantity; its `unit`
is `g` for mass pollutants, `MJ` for EMEP energy, `kJ` for MOVES energy,
`#` for particle number and `L` for volume. The factor unit remains on
`result.ef["unit"]` (`g/km`, `MJ/km`, `kJ/km`, ...); it is never copied onto
the emission row.

## Manifest

`docs/schemas/emission_manifest.schema.json` is the normative schema:
`schema_version`, `generated_at`, `ef_model`, `pollutant`, `process`,
`request_sha256`, `rows`, `columns`, and one entry per shape with `shape_id`,
`file`, `rows` and the file's `sha256`.

Resume accepts an existing manifest only when the emission model, pollutant
set, process set, row count and the table's value hash match, every listed file
opens and its Parquet row count matches and its recorded SHA-256 matches.
Anything else rewrites the complete output; a partial directory is never
trusted, and changed values with the same row count cannot reuse a stale file.

## Limitations

- The writer does not compress across shapes; each file is zstd-compressed
  Parquet.
- `resume=True` without `output_path` raises `ValueError`.
- The transport writer remains the owner of GeoParquet geometry; emission files
  carry no geometry and no GeoParquet metadata.
