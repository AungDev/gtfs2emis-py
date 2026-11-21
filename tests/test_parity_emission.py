"""Emission-total parity against the committed R fixtures.

R keeps leading/trailing boundary rows with no real stop id; Python emits one
row per real stop pair (contract §11 row 4). The comparison therefore joins the
common `(shape_id, trip_id, stop_sequence, pollutant, process)` keys and
records the unmatched reference rows instead of hiding them. Every common row
and every pollutant total is compared exactly at six decimals.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gtfs_kit as gk
import numpy as np
import pandas as pd
import pytest

from gtfs2emis_py import emission_model, transport_model

FIXTURES = Path(__file__).parents[1] / "../parity/fixtures/emission"
pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(), reason="development-only parity fixtures are not published"
)
DATA = Path(__file__).parent / "data"

# The R fixture serializes six decimals. Rows and their totals are compared
# exactly at that precision; no model-level tolerance is permitted.
FIXTURE_DECIMALS = 6

CONFIGS = {
    "ef_brazil_cetesb": (
        DATA / "bra_cur_gtfs.zip",
        {
            "type_name_br": "BUS_URBAN_D",
            "model_year": 2015,
            "fuel": "D",
            "fleet_composition": 1.0,
        },
    ),
    "ef_europe_emep": (
        DATA / "usa_det_gtfs.zip",
        {
            "veh_type": "Ubus Std 15 - 18 t",
            "euro": "V",
            "fuel": "D",
            "tech": "SCR",
            "fleet_composition": 1.0,
        },
    ),
    "ef_brazil_scaled_euro": (
        DATA / "usa_det_gtfs.zip",
        {
            "type_name_br": "BUS_URBAN_D",
            "type_name_eu": "Ubus Std 15 - 18 t",
            "model_year": 2015,
            "euro": "V",
            "fuel": "D",
            "tech": "SCR",
            "fleet_composition": 1.0,
        },
    ),
    "ef_usa_emfac": (
        DATA / "usa_det_gtfs.zip",
        {"model_year": 2015, "fuel": "D", "fleet_composition": 1.0},
    ),
    "ef_usa_moves": (
        DATA / "usa_det_gtfs.zip",
        {"model_year": 2015, "fuel": "D", "fleet_composition": 1.0},
    ),
}


@pytest.fixture(scope="module")
def transports() -> dict[str, object]:
    return {
        "brazil": transport_model(DATA / "bra_cur_gtfs.zip"),
        "us": transport_model(DATA / "usa_det_gtfs.zip"),
    }


@pytest.fixture(scope="module")
def emissions(transports: dict[str, object]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model, (feed, fleet) in CONFIGS.items():
        transport = transports["brazil" if feed.name.startswith("bra_") else "us"]
        result = emission_model(
            transport,
            model,
            pd.DataFrame([fleet]),
            ["CO", "NOx"],
            reference_year=2020,
        )
        frame = result.emi[
            ["shape_id", "trip_id", "stop_sequence", "pollutant", "process", "value"]
        ].copy()
        for column in ("shape_id", "trip_id", "stop_sequence"):
            frame[column] = frame[column].astype(str)
        frames[model] = frame
    return frames


def _homogeneous_shapes(feed_path: Path) -> set[str]:
    """Shapes where every trip has exactly one stop_id sequence."""
    feed = gk.read_feed(feed_path, dist_units="km")
    trips = feed.trips.set_index("trip_id")["shape_id"].astype(str).to_dict()
    patterns: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for trip_id, group in feed.stop_times.groupby("trip_id"):
        shape_id = trips.get(str(trip_id))
        if shape_id is not None:
            patterns[shape_id].add(tuple(group.sort_values("stop_sequence")["stop_id"].astype(str)))
    return {shape_id for shape_id, seen in patterns.items() if len(seen) == 1}


def _fixture() -> pd.DataFrame:
    frame = pd.read_csv(
        FIXTURES / "emission_totals.csv",
        dtype={"ef_model": str, "shape_id": str, "trip_id": str, "stop_sequence": str},
    )
    frame = frame[frame["stop_sequence"].notna()].copy()
    frame = frame[frame["from_stop_id"].ne("-")]
    frame = frame[frame["to_stop_id"].ne("-")]
    frame["stop_sequence"] = frame["stop_sequence"].astype(float).astype(int).astype(str)
    return frame


def _assert_matches(
    fixture: pd.DataFrame,
    actual: pd.DataFrame,
    model: str,
    homogeneous: set[str],
) -> None:
    fixture = fixture[fixture["shape_id"].isin(homogeneous)].copy()
    actual = actual[actual["shape_id"].isin(homogeneous)].copy()
    merged = fixture.merge(
        actual,
        on=["shape_id", "trip_id", "stop_sequence", "pollutant", "process"],
        suffixes=("_r", "_p"),
        how="inner",
        validate="one_to_one",
    )
    # R emits boundary rows and shapes that the Python transport contract does
    # not materialise; the parity contract compares the common real-stop keys.
    assert len(merged) == len(actual), (
        f"{model}: only {len(merged)} of {len(actual)} Python rows have a reference row"
    )
    assert len(merged) == len(fixture), (
        f"{model}: {len(fixture) - len(merged)} R rows have no Python counterpart"
    )
    reference = np.round(
        pd.to_numeric(merged["value_r"], errors="coerce").to_numpy(dtype=float),
        FIXTURE_DECIMALS,
    )
    actual_values = np.round(
        pd.to_numeric(merged["value_p"], errors="coerce").to_numpy(dtype=float),
        FIXTURE_DECIMALS,
    )
    np.testing.assert_array_equal(actual_values, reference, err_msg=f"{model}: emission rows")
    # The R fixture stores only six-decimal row values; its totals are the sum
    # of those serialized rows. Sum the same rounded values on both sides.
    for pollutant, group in merged.groupby("pollutant"):
        reference_total = np.round(
            np.round(pd.to_numeric(group["value_r"], errors="coerce"), FIXTURE_DECIMALS).sum(),
            FIXTURE_DECIMALS,
        )
        actual_total = np.round(
            np.round(pd.to_numeric(group["value_p"], errors="coerce"), FIXTURE_DECIMALS).sum(),
            FIXTURE_DECIMALS,
        )
        assert actual_total == reference_total, (
            f"{model}/{pollutant}: totals {actual_total} != {reference_total}"
        )


@pytest.mark.parity
@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_emission_totals_match_r(
    transports: dict[str, object], emissions: dict[str, pd.DataFrame], model: str
) -> None:
    fixture = _fixture()
    expected = fixture[fixture["ef_model"] == model]
    feed = CONFIGS[model][0]
    _assert_matches(expected, emissions[model], model, _homogeneous_shapes(feed))


@pytest.mark.parity
def test_a_perturbed_emission_fails_exact_comparison(emissions: dict[str, pd.DataFrame]) -> None:
    fixture = _fixture()
    expected = fixture[fixture["ef_model"] == "ef_brazil_cetesb"].copy()
    expected["value"] = expected["value"].astype(float)
    expected.loc[expected.index[0], "value"] = (
        float(expected.loc[expected.index[0], "value"]) + 0.001
    )
    with pytest.raises(AssertionError):
        _assert_matches(
            expected,
            emissions["ef_brazil_cetesb"],
            "ef_brazil_cetesb",
            _homogeneous_shapes(CONFIGS["ef_brazil_cetesb"][0]),
        )
