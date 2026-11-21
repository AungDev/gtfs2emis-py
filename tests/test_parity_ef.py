"""EF and wear parity against the committed R fixtures.

The grid fixtures are the reference's own inputs: one representative row per
EMEP family, every EMFAC/MOVES bin boundary, all CETESB combinations, the
scaled composite and the three wear processes. Values are compared at the
fixture's six-decimal precision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gtfs2emis_py.ef import (
    ef_brazil_cetesb,
    ef_brazil_scaled_euro,
    ef_europe_emep,
    ef_scaled_euro,
    ef_usa_emfac,
    ef_usa_moves,
    emi_europe_emep_wear,
)

FIXTURES = Path(__file__).parents[1] / "../parity/fixtures/ef"
FIXTURE_DECIMALS = 6  # R fixture values are serialized with six decimals
pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(
        not FIXTURES.exists(), reason="development-only parity fixtures are not published"
    ),
]


def _grid() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "ef_grid.csv", dtype=str)
    for column in ("model_year", "slope", "load", "speed_kmh", "ef"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _close(actual: Any, expected: Any) -> None:
    """Require exact equality at the R fixture's serialization precision."""
    np.testing.assert_array_equal(
        np.round(np.asarray(actual, dtype=float), FIXTURE_DECIMALS),
        np.round(np.asarray(expected, dtype=float), FIXTURE_DECIMALS),
    )


def test_cetesb_grid_matches_r() -> None:
    frame = _grid()
    expected = frame[frame["model"] == "ef_brazil_cetesb"]
    result = ef_brazil_cetesb(
        sorted(expected["pollutant"].unique()),
        sorted(expected["veh_type"].unique()),
        [2010, 2015, 2020],
    )
    for (pollutant, vehicle), group in expected.groupby(["pollutant", "veh_type"]):
        actual = result.table[
            (result.table["pollutant"] == pollutant) & (result.table["veh_type"] == vehicle)
        ].sort_values("model_year")
        expected_years = group.sort_values("model_year")
        assert actual["model_year"].tolist() == expected_years["model_year"].astype(int).tolist()
        _close(actual["ef"].to_numpy(), expected_years["ef"].to_numpy())


def test_emep_grid_matches_r_and_records_the_ambiguous_families() -> None:
    frame = _grid()
    expected = frame[frame["model"] == "ef_europe_emep"]
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    failures = set(manifest["emep_failures"])
    covered = set(expected["family"].unique())

    assert not failures & covered
    assert len(failures) + len(covered) == 20  # 20 Function_ID families
    assert len(covered) == 7

    for _family, group in expected.groupby("family"):
        first = group.iloc[0]
        result = ef_europe_emep(
            group["speed_kmh"].tolist(),
            str(first["veh_type"]),
            str(first["euro"]),
            str(first["pollutant"]),
            fuel=str(first["fuel"] or "D"),
            tech=str(first["tech"]) if pd.notna(first["tech"]) else "-",
            slope=float(first["slope"]) if pd.notna(first["slope"]) else 0.0,
            load=float(first["load"]) if pd.notna(first["load"]) else 0.5,
        )
        _close(result.table["ef"].to_numpy(), group["ef"].to_numpy())


@pytest.mark.parametrize("model", ["ef_usa_emfac", "ef_usa_moves"])
def test_us_speed_bins_match_r(model: str) -> None:
    frame = _grid()
    expected = frame[frame["model"] == model]
    for pollutant, group in expected.groupby("pollutant"):
        speeds = group["speed_kmh"].tolist()
        if model == "ef_usa_emfac":
            result = ef_usa_emfac(
                pollutant,
                reference_year=2020,
                fuel="D",
                model_year=2015,
                speed=speeds,
            )
        else:
            result = ef_usa_moves(
                pollutant,
                2015,
                reference_year=2020,
                speed=speeds,
                fuel="D",
            )
        _close(result.table["ef"].to_numpy(), group["ef"].to_numpy())


def test_scaled_composite_matches_r() -> None:
    frame = _grid()
    expected = frame[frame["model"] == "ef_brazil_scaled_euro"]
    first = expected.iloc[0]
    result = ef_brazil_scaled_euro(
        str(first["pollutant"]),
        expected["speed_kmh"].tolist(),
        str(first["veh_type"]),
        2015,
        "Ubus Std 15 - 18 t",
        str(first["euro"]),
        fuel=str(first["fuel"]),
        tech=str(first["tech"]),
    )
    _close(result.table["ef"].to_numpy(), expected["ef"].to_numpy())

    local = ef_brazil_cetesb("CO", "BUS_URBAN_D", 2015)
    scaled = ef_scaled_euro(
        local,
        expected["speed_kmh"].tolist(),
        "Ubus Std 15 - 18 t",
        "V",
        "CO",
        fuel="D",
        tech="SCR",
    )
    _close(scaled.table["ef"].to_numpy(), expected["ef"].to_numpy())


def test_wear_grid_matches_r() -> None:
    frame = _grid()
    expected = frame[frame["model"] == "emi_europe_emep_wear"]
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    failures = {tuple(key.split("/", 1)) for key in manifest["wear_failures"]}
    covered = set(zip(expected["family"], expected["pollutant"], strict=True))
    assert failures == {("tyre", "PM1.0"), ("brake", "PM1.0")}
    assert not failures & covered

    for (process, pollutant, vehicle), group in expected.groupby(
        ["family", "pollutant", "veh_type"]
    ):
        speeds = sorted(group["speed_kmh"].unique().tolist())
        result = emi_europe_emep_wear(
            [1.0] * len(speeds),
            speeds,
            pollutant,
            ["Ubus Std 15 - 18 t", "Ubus Artic >18 t"],
            [0.5, 0.5],
            load=0.5,
            process=process,
        )
        actual = result[result["veh_type"] == vehicle].sort_values("segment")["emi_g"].to_numpy()
        _close(actual, group.sort_values("speed_kmh")["ef"].to_numpy())


def test_a_perturbed_factor_fails_exact_parity() -> None:
    frame = _grid()
    expected = frame[frame["model"] == "ef_brazil_cetesb"]
    group = expected.groupby(["pollutant", "veh_type"]).get_group(("CO", "BUS_URBAN_D"))
    perturbed = group.copy()
    perturbed.loc[perturbed.index[0], "ef"] = float(perturbed.iloc[0]["ef"]) + 0.001
    with pytest.raises(AssertionError):
        _close(group["ef"].to_numpy(), perturbed["ef"].to_numpy())
