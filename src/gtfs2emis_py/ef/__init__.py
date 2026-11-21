"""Emission factor models.

The packaged EF tables and their loader, plus the CETESB port.
"""

from gtfs2emis_py.ef.base import (
    EmissionFactors,
    MissingEmissionFactorError,
    UnsupportedPollutantError,
)
from gtfs2emis_py.ef.brazil_cetesb import CETESB_VEHICLES, ef_brazil_cetesb
from gtfs2emis_py.ef.data import EFDataError, list_ef_tables, load_ef_table
from gtfs2emis_py.ef.europe_emep import EmepFallbackWarning, ef_europe_emep
from gtfs2emis_py.ef.transforms import (
    ef_brazil_scaled_euro,
    ef_scaled_euro,
    fuel_correction,
)
from gtfs2emis_py.ef.usa_emfac import ef_usa_emfac
from gtfs2emis_py.ef.usa_moves import ef_usa_moves
from gtfs2emis_py.ef.wear import PROCESSES, WearError, emi_europe_emep_wear

__all__ = [
    "CETESB_VEHICLES",
    "EFDataError",
    "EmepFallbackWarning",
    "EmissionFactors",
    "MissingEmissionFactorError",
    "PROCESSES",
    "UnsupportedPollutantError",
    "WearError",
    "ef_brazil_cetesb",
    "ef_brazil_scaled_euro",
    "ef_europe_emep",
    "ef_scaled_euro",
    "ef_usa_emfac",
    "ef_usa_moves",
    "emi_europe_emep_wear",
    "fuel_correction",
    "list_ef_tables",
    "load_ef_table",
]
