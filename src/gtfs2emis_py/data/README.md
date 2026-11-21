# Package data

Only the EMEP/EEA lookup table is packaged: `europe_emep.parquet`, the
`manifest.json` and `LICENSES/`. CETESB, EMFAC and MOVES are fetch-only from
their pinned original `.rda` URLs; `load_ef_table(..., fetch=True)`
converts and verifies them in the cache.

The tables are built from `data-raw/` scripts and read through
`importlib.resources`, so the packaged EMEP file must stay inside the package
directory.
