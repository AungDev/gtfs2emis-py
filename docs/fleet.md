# Fleet tables

`load_fleet()` is the single place that reads, normalises and validates fleet
data before `emission_model()` sees it. The validated result is a
[`FleetTable`](api.md) with a `table`, `mode`, `ef_model`, `group_columns` and
`composition_column`.

## Two modes

- **average** — one row per vehicle-characteristics combination, with a
  `fleet_composition` share. Most agencies publish this shape.
- **per-vehicle** — rows carry `shape_id` or `route_id` and a count; the share
  is derived within each assignment group.

`mode="auto"` infers the mode; `"average"` and `"per_vehicle"` assert it. A
table with both assignment columns is rejected.

## Input normalisation

Column aliases are normalised on read:

| Alias | Canonical |
|---|---|
| `year` | `model_year` |
| `N` | `count` |
| `total` | `count` |

The remaining R column names are kept because the EF ports read the same words
as the reference.

## Model requirements

| `ef_model` | Required columns |
|---|---|
| `ef_brazil_cetesb` | `type_name_br`, `model_year`, `fuel` |
| `ef_europe_emep` | `veh_type`, `euro`, `fuel`, `tech` |
| `ef_usa_emfac` | `model_year`, `fuel` |
| `ef_usa_moves` | `model_year`, `fuel` |

The allowed category vocabularies are defined in `gtfs2emis_py.fleet.FLEET_SPECS`
and include both EMEP coach segments; US fuel aliases (`Diesel`, `Gasoline`) are
normalised to `D` and `G` at this boundary.

## Validation rules

- `model_year` must be an integer; fractional years are rejected, not truncated.
- `fleet_composition` must be in `[0, 1]` and sum to 1 per group within
  `tolerance` (default `1e-6`).
- Counts must be positive when composition is derived.
- Unknown categories, missing required columns, duplicate keys and contradictory
  modes raise `FleetError` with the offending values.

```python
from gtfs2emis_py import load_fleet

fleet = load_fleet("fleet.csv", ef_model="ef_europe_emep")
fleet.table.head()
```
