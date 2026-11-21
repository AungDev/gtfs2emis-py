"""Unit boundary helpers.

The helpers exist so a wrong or impossible value is rejected where it enters the
API, not three functions later. These tests pin the conversions, the rejection
rules and the error messages.
"""

from __future__ import annotations

import math

import pytest

from gtfs2emis_py.units import UnitError, as_g, as_g_per_km, as_km, as_kmh


class TestAsKm:
    def test_km_is_a_passthrough(self) -> None:
        assert as_km(2.5) == 2.5
        assert as_km(2.5, "km") == 2.5

    def test_metres_to_km(self) -> None:
        assert as_km(1000, "m") == 1.0
        assert as_km(250, "m") == 0.25

    def test_miles_to_km(self) -> None:
        assert as_km(1, "mi") == pytest.approx(1.609344)

    def test_feet_to_km(self) -> None:
        assert as_km(1000, "ft") == pytest.approx(0.3048)

    def test_zero_is_allowed(self) -> None:
        assert as_km(0, "m") == 0.0

    def test_returns_float(self) -> None:
        assert isinstance(as_km(1, "m"), float)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
    def test_rejects_non_finite_or_negative(self, value: float) -> None:
        with pytest.raises(UnitError, match="distance"):
            as_km(value)

    def test_rejects_bool(self) -> None:
        with pytest.raises(UnitError, match="real number"):
            as_km(True)  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        with pytest.raises(UnitError, match="real number"):
            as_km("1.0")  # type: ignore[arg-type]

    def test_unknown_unit_lists_supported_units(self) -> None:
        with pytest.raises(UnitError, match="km"):
            as_km(1, "yard")  # type: ignore[arg-type]


class TestAsKmh:
    def test_kmh_is_a_passthrough(self) -> None:
        assert as_kmh(50) == 50.0

    def test_metres_per_second_to_kmh(self) -> None:
        assert as_kmh(1, "m/s") == pytest.approx(3.6)

    def test_mph_to_kmh(self) -> None:
        assert as_kmh(60, "mph") == pytest.approx(96.56064)

    def test_rejects_negative_speed(self) -> None:
        with pytest.raises(UnitError, match="speed"):
            as_kmh(-5)


class TestAsGPerKm:
    def test_g_per_km_is_a_passthrough(self) -> None:
        assert as_g_per_km(12.5) == 12.5

    def test_g_per_mile_to_g_per_km(self) -> None:
        assert as_g_per_km(1, "g/mi") == pytest.approx(0.621371192)

    def test_g_per_metre_to_g_per_km(self) -> None:
        assert as_g_per_km(1, "g/m") == pytest.approx(1000.0)

    def test_kg_per_km_to_g_per_km(self) -> None:
        assert as_g_per_km(1, "kg/km") == pytest.approx(1000.0)

    def test_unknown_unit_names_the_quantity(self) -> None:
        with pytest.raises(UnitError, match="emission factor"):
            as_g_per_km(1, "g/kwh")  # type: ignore[arg-type]


class TestAsG:
    def test_grams_is_a_passthrough(self) -> None:
        assert as_g(5) == 5.0

    def test_kilograms_to_grams(self) -> None:
        assert as_g(1, "kg") == 1000.0

    def test_milligrams_to_grams(self) -> None:
        assert as_g(1, "mg") == pytest.approx(0.001)

    def test_tonnes_to_grams(self) -> None:
        assert as_g(1, "t") == 1_000_000.0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001])
    def test_rejects_non_finite_or_negative(self, value: float) -> None:
        with pytest.raises(UnitError, match="mass"):
            as_g(value)


def test_helpers_never_return_objects() -> None:
    """option (b): values crossing the boundary are plain floats."""
    results = [as_km(1, "m"), as_kmh(1, "m/s"), as_g_per_km(1, "kg/km"), as_g(1, "kg")]
    assert all(isinstance(result, float) and math.isfinite(result) for result in results)
