# API reference

Public entry points live in `gtfs2emis_py`; the modules below are the supported
surface. Everything else is internal. The `gtfs2emis_py`, `gtfs2emis_py.ef` and
`gtfs2emis_py.gtfs` packages are re-export namespaces; their public symbols are
documented in the concrete submodules below.

## Top-level

```{eval-rst}
.. automodule:: gtfs2emis_py.transport_model
   :members:
   :show-inheritance:

.. automodule:: gtfs2emis_py.emission_model
   :members:
   :show-inheritance:

.. automodule:: gtfs2emis_py.multiply
   :members:

.. automodule:: gtfs2emis_py.analysis
   :members:

.. automodule:: gtfs2emis_py.analysis.core
   :members:

.. automodule:: gtfs2emis_py.emission_writer
   :members:
```

## Fleet and factors

```{eval-rst}
.. automodule:: gtfs2emis_py.fleet
   :members:

.. automodule:: gtfs2emis_py.ef.base
   :members:

.. automodule:: gtfs2emis_py.ef.data
   :members:

.. automodule:: gtfs2emis_py.ef.brazil_cetesb
   :members:

.. automodule:: gtfs2emis_py.ef.europe_emep
   :members:

.. automodule:: gtfs2emis_py.ef.usa_emfac
   :members:

.. automodule:: gtfs2emis_py.ef.usa_moves
   :members:

.. automodule:: gtfs2emis_py.ef.transforms
   :members:

.. automodule:: gtfs2emis_py.ef.wear
   :members:
```

## GTFS internals

```{eval-rst}
.. automodule:: gtfs2emis_py.gtfs.reader
   :members:

.. automodule:: gtfs2emis_py.gtfs.interpolation
   :members:

.. automodule:: gtfs2emis_py.gtfs.speed
   :members:

.. automodule:: gtfs2emis_py.gtfs.parallel
   :members:

.. automodule:: gtfs2emis_py.gtfs.slope
   :members:

.. automodule:: gtfs2emis_py.gtfs.writer
   :members:
```

## Units

```{eval-rst}
.. automodule:: gtfs2emis_py.units
   :members:

.. automodule:: gtfs2emis_py.pollutants
   :members:
```

## Public constants

```{eval-rst}
.. autodata:: gtfs2emis_py.__version__

.. autodata:: gtfs2emis_py.BUS_ROUTE_TYPES
.. autodata:: gtfs2emis_py.EF_MODELS

.. autodata:: gtfs2emis_py.ef.brazil_cetesb.CETESB_VEHICLES
.. autodata:: gtfs2emis_py.ef.wear.PROCESSES
.. autodata:: gtfs2emis_py.emission_model.EF_MODELS
.. py:data:: gtfs2emis_py.fleet.FLEET_SPECS
   :type: dict[str, FleetSpec]

   Mapping of supported EF model names to the fleet columns and category
   vocabularies each model requires.
.. autodata:: gtfs2emis_py.gtfs.interpolation.BUS_ROUTE_TYPES
.. autodata:: gtfs2emis_py.gtfs.interpolation.SNAP_WARN_M
.. autodata:: gtfs2emis_py.gtfs.slope.SLOPE_CLASSES
.. autodata:: gtfs2emis_py.gtfs.writer.SCHEMA_VERSION
.. autodata:: gtfs2emis_py.gtfs.writer.SUPPORTED_SCHEMA_VERSIONS
.. autodata:: gtfs2emis_py.pollutants.PollutantKind
.. autodata:: gtfs2emis_py.transport_model.BUS_ROUTE_TYPES
.. autodata:: gtfs2emis_py.units.CANONICAL_DISTANCE
.. autodata:: gtfs2emis_py.units.CANONICAL_EMISSION_FACTOR
.. autodata:: gtfs2emis_py.units.CANONICAL_LENGTH
.. autodata:: gtfs2emis_py.units.CANONICAL_MASS
.. autodata:: gtfs2emis_py.units.CANONICAL_SPEED
.. autodata:: gtfs2emis_py.units.CANONICAL_TIME
```

## Type aliases

```{eval-rst}
.. py:data:: gtfs2emis_py.ef.base.Process
   :type: Literal["hot_exhaust", "tyre", "brake", "road"]

.. py:data:: gtfs2emis_py.fleet.FleetMode
   :type: Literal["average", "per_vehicle"]

.. py:data:: gtfs2emis_py.units.DistanceUnit
   :type: Literal["km", "m", "mi", "ft"]

.. py:data:: gtfs2emis_py.units.SpeedUnit
   :type: Literal["km/h", "m/s", "mph"]

.. py:data:: gtfs2emis_py.units.EmissionFactorUnit
   :type: Literal["g/km", "g/mi", "g/m", "kg/km"]

.. py:data:: gtfs2emis_py.units.MassUnit
   :type: Literal["g", "kg", "mg", "t"]
```

