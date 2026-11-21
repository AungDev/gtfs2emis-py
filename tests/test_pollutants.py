"""Pollutant registry tests.

The registry exists because R tags every emission factor ``g/km``; these tests
pin the exceptions that make the port's output correct — energy in MJ/km or
kJ/km, particle number in #/kWh — and the lookup behaviour the EF ports rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

try:  # tomllib is stdlib from Python 3.11; the dev extra pulls tomli on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the 3.10 CI job
    import tomli as tomllib

from gtfs2emis_py.pollutants import (
    Pollutant,
    PollutantKind,
    UnknownPollutantError,
    emission_value_unit,
    get_pollutant,
    list_pollutants,
    pollutant_unit,
)

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_mass_pollutants_are_g_per_km() -> None:
    for name in ("CO", "CO2", "NOx", "PM10", "PM2.5", "CH4", "NH3", "SO2"):
        pollutant = get_pollutant(name)
        assert pollutant.unit == "g/km"
        assert pollutant.kind == "mass"


def test_ec_defaults_to_mj_per_km() -> None:
    pollutant = get_pollutant("EC")
    assert pollutant.unit == "MJ/km"
    assert pollutant.kind == "energy"


def test_ec_unit_depends_on_the_model() -> None:
    assert get_pollutant("EC", model="ef_europe_emep").unit == "MJ/km"
    assert get_pollutant("EC", model="ef_usa_moves").unit == "kJ/km"
    assert pollutant_unit("EC", model="ef_usa_moves") == "kJ/km"


def test_ec_falls_back_to_canonical_unit_for_other_models() -> None:
    # Only the models that disagree are listed; anything else uses MJ/km.
    assert pollutant_unit("EC", model="ef_brazil_cetesb") == "MJ/km"
    assert pollutant_unit("EC", model="not_a_model") == "MJ/km"


def test_unit_for_without_a_model_uses_the_canonical_unit() -> None:
    assert get_pollutant("EC").unit_for() == "MJ/km"
    assert get_pollutant("CO").unit_for(None) == "g/km"


def test_energy_and_mass_units_are_not_interchangeable() -> None:
    assert get_pollutant("EC").unit != get_pollutant("CO").unit
    assert get_pollutant("EC").kind != get_pollutant("CO").kind


def test_emission_value_unit_is_not_the_factor_unit() -> None:
    assert emission_value_unit("CO") == "g"
    assert emission_value_unit("EC") == "MJ"
    assert emission_value_unit("EC", model="ef_usa_moves") == "kJ"
    assert emission_value_unit("SPN25") == "#"


def test_spn25_is_particle_number() -> None:
    pollutant = get_pollutant("SPN25")
    assert pollutant.unit == "#/kWh"
    assert pollutant.kind == "count"


def test_lookup_is_case_insensitive() -> None:
    assert get_pollutant("nox").name == "NOx"
    assert get_pollutant("NOX").name == "NOx"


@pytest.mark.parametrize(("alias", "canonical"), [("nmvoc", "VOC"), ("pm25", "PM2.5")])
def test_aliases_resolve(alias: str, canonical: str) -> None:
    assert get_pollutant(alias).name == canonical


def test_unknown_pollutant_lists_the_registry() -> None:
    with pytest.raises(UnknownPollutantError) as exc_info:
        get_pollutant("CO3")
    message = str(exc_info.value)
    assert "CO3" in message
    assert "PM10" in message and "SPN25" in message


def test_registry_names_are_unique_and_sorted() -> None:
    names = list_pollutants()
    assert len(names) == len(set(names))
    assert list(names) == sorted(names)


def test_every_entry_has_a_unit_and_a_known_kind() -> None:
    for name in list_pollutants():
        pollutant = get_pollutant(name)
        assert isinstance(pollutant, Pollutant)
        assert pollutant.unit
        assert pollutant.kind in get_args(PollutantKind)


def test_kind_vocabulary_is_the_adr_set() -> None:
    """declares four kinds; ``volume`` is currently unused.

    Enumerating the set means a fifth kind, or a silently dropped one, is a
    visible test change rather than a widened ``Literal`` nobody noticed.
    """
    assert set(get_args(PollutantKind)) == {"mass", "energy", "count", "volume"}
    assert {get_pollutant(name).kind for name in list_pollutants()} <= set(get_args(PollutantKind))


def test_model_resolution_keeps_the_kind() -> None:
    resolved = get_pollutant("EC", model="ef_usa_moves")
    assert resolved.kind == "energy"
    assert resolved.name == "EC"


def test_returned_entry_is_a_copy() -> None:
    """Mutating a lookup result must not change the registry."""
    first = get_pollutant("EC")
    first.units_by_model["EC"] = "g/km"
    first.units_by_model["CO"] = "kg/km"
    assert get_pollutant("EC").unit_for("ef_europe_emep") == "MJ/km"
    assert get_pollutant("CO").unit == "g/km"


def test_model_resolved_entry_is_a_copy() -> None:
    resolved = get_pollutant("EC", model="ef_usa_moves")
    resolved.units_by_model["EC"] = "g/km"
    assert get_pollutant("EC", model="ef_usa_moves").unit == "kJ/km"


def test_returned_entry_is_picklable() -> None:
    """sends results across loky workers by pickle."""
    import pickle

    pollutant = get_pollutant("EC", model="ef_usa_moves")
    assert pickle.loads(pickle.dumps(pollutant)) == pollutant


def test_pint_is_not_a_dependency() -> None:
    """option (b) only: no unit library, even at the boundary."""
    project = tomllib.loads(PYPROJECT.read_text())
    declared = " ".join(project["project"]["dependencies"]).casefold()
    assert "pint" not in declared
