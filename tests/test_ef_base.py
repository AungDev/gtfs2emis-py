"""Tests for the shared EF tuple coercer."""

from __future__ import annotations

import pytest

from gtfs2emis_py.ef.base import as_tuple


def test_as_tuple_accepts_scalars_and_sequences() -> None:
    assert as_tuple(1.0, "value") == (1.0,)
    assert as_tuple([1, 2], "value") == (1, 2)


@pytest.mark.parametrize("value", [object(), None, True])
def test_as_tuple_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(TypeError, match="scalar or a sequence"):
        as_tuple(value, "value")
