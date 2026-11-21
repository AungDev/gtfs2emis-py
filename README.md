# gtfs2emis-py

Python port of [gtfs2emis](https://github.com/ipeaGIT/gtfs2emis) — estimate
emissions of public transport systems from GTFS data.

> **Status:** core implementation, tests, tutorials and documentation are complete.
> Requires Python 3.10+.

**Scope.** Bus-only hot exhaust plus EMEP tyre, brake and road wear, and the analysis helpers. Rail, cold start, evaporative emissions, resuspension, uncertainty and a command-line interface are out of scope.

## Numerical parity

Transport distances, speeds, cumulative distance/time and the EF/emission
outputs target exact numerical parity with `gtfs2emis` 0.1.2 (the `DESCRIPTION`
version at commit `0dbca7d`).

`from_timestamp`/`to_timestamp` are intentionally **not** compared against that
pinned R release. Its `adjust_speed()` timestamp handling contains a bug that was
fixed upstream in `gtfs2gps` commit `dae914f283` ("Fix adjust_speed() timestamps,
gps_as_sflinestring() segments and midnight wrap"). This port follows the
documented real-time construction from `service_date` and `cumtime` rather than
reproducing the old bug; the timestamp columns are therefore a documented
structural divergence, not a numeric tolerance. The other transport columns
remain exact.

## Layout

Paths are relative to the package root — the repository root in the released
tree, and `gtfs2emis-py/` in the development checkout:

```
├── src/gtfs2emis_py/      # library source
│   ├── gtfs/              # GTFS reader + transport model
│   ├── ef/                # emission factor models
│   ├── analysis/          # emis_to_dt / emis_summary / emis_grid
│   └── data/              # bundled EF data (EMEP Parquet + manifest)
├── tests/                 # unit + integration tests
│   └── data/              # committed fixtures: sample feeds, fleets, DEM
├── notebooks/             # executable tutorials
├── benchmarks/            # component and pipeline budgets
├── scripts/               # fixture generators (make_synthetic_feed.py)
└── docs/                  # Sphinx documentation
```

## Tutorials

The `notebooks/` directory is an executable tour of the public surface:

| Notebook | Covers |
|---|---|
| `01_getting_started.ipynb` | transport -> fleet -> emissions -> summary |
| `02_fleet_data.ipynb` | fleet modes, composition and validation |
| `03_emission_factors.ipynb` | EMEP and EMFAC factor lookups |
| `04_wear_emissions.ipynb` | tyre, brake and road wear |
| `05_gtfs_reader.ipynb` | reader validation, cleaning report and unchecked reads |
| `06_outputs_and_resume.ipynb` | transport/emission writers, manifests and resume |
| `07_analysis.ipynb` | `emis_summary()` and `emis_grid()` |
| `08_units_and_pollutants.ipynb` | boundary units and factor/value units |
| `09_parallel_and_slope.ipynb` | `map_shapes()`, parallel transport and DEM slope |
| `10_ef_models_and_data.ipynb` | EF loader, CETESB/MOVES factors and the CETESB-scaled-EMEP model |
| `11_validation_and_errors.ipynb` | service dates, fleet/EF/parameter validation and the public error types |

Execute every notebook in memory with:

```bash
uv run python notebooks/execute.py
```

## Install (dev)

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned by
`uv.lock`; CI installs with `--locked`, so a stale lockfile is a build failure
rather than a local-only difference.

```bash
cd gtfs2emis-py
uv sync --locked --extra dev
```

Run the same gates CI runs:

```bash
uv run pytest --cov            # coverage floor is 90 % ([tool.coverage.report])
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`pandas` is capped below 3 in `pyproject.toml` while `gtfs-kit` 13.0.1 declares
`fare_attributes.transfer_duration` as `Int16`; pandas 3 refuses the cast the
Detroit feed needs. The reason and the removal condition are recorded in
[`docs/core_dependencies.md`](docs/core_dependencies.md).

## Emission-factor data

Only the EMEP/EEA table is packaged. The CETESB, EMFAC and MOVES tables are
fetch-only from their original pinned upstream URLs (the `ipeaGIT/gtfs2emis`
data files at `0dbca7d`, whose `DESCRIPTION` declares package version `0.1.2`);
no separate data repository is used. Install the conversion
extra, then call:

```bash
pip install "gtfs2emis-py[data]"
```

```python
from gtfs2emis_py.ef import load_ef_table

table = load_ef_table("usa_emfac", fetch=True)
```

The loader downloads the original `.rda`, verifies its pinned SHA-256,
converts the R object to the declared Arrow schema, and verifies the
`content_sha256` before caching it under `GTFS2EMIS_CACHE` (default
`~/.cache/gtfs2emis-py/ef`). If the source bytes or values change, it raises
`EFDataError` with the expected and actual hashes and does not update the
manifest; a new data edition requires a deliberate manifest and release change.
`GTFS2EMIS_DATA_URL` remains available only for users mirroring the built
Parquet files.

## License

The Python code is licensed under the MIT License. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the upstream `gtfs2emis` attribution and the per-source
data terms.

Emission-factor data is **not** covered by the code licence. The EMEP/EEA table
is redistributed under CC-BY 4.0 with attribution. CETESB, CARB EMFAC and EPA
MOVES tables are not included in the wheel; `load_ef_table(..., fetch=True)`
downloads the original pinned `.rda` from the manifest URL, verifies its
SHA-256, and verifies the converted values before caching. Each source's terms
are recorded in `src/gtfs2emis_py/data/ef/LICENSES/` and `manifest.json`.
