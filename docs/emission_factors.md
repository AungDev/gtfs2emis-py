# Emission factors

The EF ports return an `EmissionFactors` table for a fixed request. Factor
units are not always `g/km`: EMEP energy is `MJ/km`, MOVES energy is `kJ/km`,
and particle number is `#/kWh`. The pollutant registry owns those units; see
[Units and pollutants](units_and_pollutants.md).

## Loading tables

```python
from gtfs2emis_py.ef import list_ef_tables, load_ef_table

list_ef_tables()
table = load_ef_table("europe_emep")  # packaged
cetesb = load_ef_table("brazil_cetesb", fetch=True)  # pinned upstream .rda
```

`list_ef_tables()` returns the four supported names. EMEP/EEA ships in the wheel;
CETESB, EMFAC and MOVES are fetch-only. The loader verifies the
pinned SHA-256, converts the R object to the declared Arrow schema, verifies the
value-level `content_sha256`, and caches the result under `GTFS2EMIS_CACHE`.

## Supported models

| Function | Source | Key inputs | Notes |
|---|---|---|---|
| `ef_brazil_cetesb` | CETESB | pollutant, bus class, model year | constant factor per vehicle |
| `ef_europe_emep` | EMEP/EEA | speed, segment, Euro, fuel, technology, slope, load | 7 of 20 `Function_ID` families are evaluable; R aborts the rest on duplicate edition keys |
| `ef_usa_emfac` | CARB EMFAC2017 | reference year, model year, fuel, speed | right-closed speed bins |
| `ef_usa_moves` | EPA MOVES3 | reference year, model year, fuel, speed | right-closed speed bins |
| `emi_europe_emep_wear` | EMEP/EEA | speed, load, road/tyre/brake process | EMEP-only wear |

`ef_scaled_euro()` and `ef_brazil_scaled_euro()` scale a local factor by the EMEP
speed curve; `EF(SDC)` is evaluated at the same slope and load as the requested
call. `fuel_correction()` returns the EMEP fuel-correction ratio.

## Deliberate divergences

- CETESB `FC`, `FS` and `KML` are rejected because their units are not stated;
  no conversion is invented.
- EMEP `SPN23 [#/kWh]` is rejected because its count-per-energy unit is not
  mass-per-distance and the reference gives no conversion.
- The 13 non-evaluable EMEP families are an upstream data ambiguity, recorded in
  the packaged manifest and in `docs/parity.md`.
