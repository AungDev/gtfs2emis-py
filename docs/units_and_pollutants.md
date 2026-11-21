# Units and pollutants

The package computes on plain `float64` values in documented canonical units:

| Quantity | Canonical unit |
|---|---|
| distance | `km` |
| speed | `km/h` |
| emission factor | `g/km` |
| mass | `g` |
| spatial resolution | `m` |
| time | `s` |

No unit library is used; `pint` is deliberately not a dependency.
Public boundary functions accept a number and a unit name, convert with
`as_km()`, `as_kmh()`, `as_g_per_km()` or `as_g()`, and reject non-finite,
boolean, negative and unknown-unit inputs with `UnitError`.

## Emission-factor units

`gtfs2emis_py.pollutants` is the registry for the units a factor is actually
stored in:

- mass pollutants — `g/km`
- `EC` — `MJ/km` for EMEP and `kJ/km` for MOVES
- `SPN25` — `#/kWh`
- CETESB `gCO2/KWH` and `gD/KWH` — `g/kWh`

`pollutant_unit(name, model=...)` returns the factor unit;
`emission_value_unit(name, model=...)` returns the unit of the *emitted value*
(grams, megajoules, kilojoules or count). The emission table uses the value
unit, while `EmissionResult.ef["unit"]` keeps the factor unit.

The registry is case-insensitive and accepts common aliases (`nox`, `pm25`,
`nmvoc`). Unknown names raise `UnknownPollutantError` listing the registry.

```python
from gtfs2emis_py.pollutants import emission_value_unit, pollutant_unit

pollutant_unit("EC", model="ef_usa_moves")  # "kJ/km"
emission_value_unit("EC", model="ef_usa_moves")  # "kJ"
```
