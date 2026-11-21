# gtfs2emis-py

Python port of the R package `gtfs2emis` (v0.1.2). It reads a GTFS feed, builds
a transport table, and estimates hot-exhaust and non-exhaust emissions from a
fleet table.

```{toctree}
:maxdepth: 2
:caption: Guides

transport_model
fleet
emission_model
emission_factors
emission_output
analysis
units_and_pollutants
slope
parallelism
performance
gtfs_reader
transport_output
core_dependencies
api
```

## Quick start

```python
from gtfs2emis_py import emission_model, load_fleet, transport_model

tp = transport_model("feed.zip")
fleet = load_fleet("fleet.csv", ef_model="ef_brazil_cetesb")
result = emission_model(tp, "ef_brazil_cetesb", fleet, ["CO", "NOx"])
print(result.to_long().head())
```

The transport column schema and the rules behind units, the result type, time
and parallelism are documented in the guide pages and API reference.

## Indices

- {ref}`genindex`
- {ref}`search`
