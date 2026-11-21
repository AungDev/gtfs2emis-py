"""Parallel-map tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gtfs2emis_py.gtfs import parallel
from gtfs2emis_py.gtfs.parallel import (
    ShapeProcessingError,
    ShapeTask,
    map_shapes,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _double(task: ShapeTask) -> float:
    return float(task.payload) * 2


def _slow(task: ShapeTask) -> float:
    time.sleep(float(task.payload))
    return float(task.payload)


def _fail_on_bad(task: ShapeTask) -> str:
    if task.payload == "bad":
        raise ValueError(f"bad payload for {task.shape_id}")
    return str(task.payload)


def _nested(task: ShapeTask) -> object:
    return map_shapes(_double, [ShapeTask(task.shape_id, 1)], n_jobs=1)


def test_serial_preserves_order() -> None:
    tasks = [ShapeTask(str(index), index) for index in range(3)]
    outcome = map_shapes(_double, tasks, n_jobs=1)
    assert outcome.results == (0.0, 2.0, 4.0)
    assert outcome.failures == ()


def test_parallel_preserves_input_order() -> None:
    tasks = [ShapeTask("a", 0.2), ShapeTask("b", 0.01), ShapeTask("c", 0.1)]
    outcome = map_shapes(_slow, tasks, n_jobs=2)
    assert outcome.results == (0.2, 0.01, 0.1)
    assert outcome.failures == ()


def test_serial_and_parallel_agree() -> None:
    tasks = [ShapeTask(str(index), index) for index in range(4)]
    assert (
        map_shapes(_double, tasks, n_jobs=1).results == map_shapes(_double, tasks, n_jobs=2).results
    )


def test_a_failing_shape_does_not_lose_the_others() -> None:
    tasks = [ShapeTask("good-1", "x"), ShapeTask("bad", "bad"), ShapeTask("good-2", "y")]
    outcome = map_shapes(_fail_on_bad, tasks, n_jobs=2)
    assert outcome.results == ("x", None, "y")
    assert len(outcome.failures) == 1
    failure = outcome.failures[0]
    assert failure.shape_id == "bad"
    assert failure.exception_type == "ValueError"
    assert "bad payload for bad" in failure.message
    assert "Traceback" in failure.traceback


def test_raise_if_failed_names_the_shapes() -> None:
    tasks = [ShapeTask("bad", "bad")]
    outcome = map_shapes(_fail_on_bad, tasks, n_jobs=1)
    with pytest.raises(ShapeProcessingError) as exc_info:
        outcome.raise_if_failed()
    assert "bad" in str(exc_info.value)
    assert exc_info.value.shape_ids == ("bad",)


def test_serial_accepts_a_local_closure() -> None:
    # n_jobs=1 runs in-process, so the callable need not be picklable.
    outcome = map_shapes(lambda task: task.payload + 1, [ShapeTask("a", 1)], n_jobs=1)
    assert outcome.results == (2,)


@pytest.mark.parametrize("value", [0, -1, 10_000, True, "2"])
def test_invalid_n_jobs(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        map_shapes(_double, [ShapeTask("a", 1)], n_jobs=value)  # type: ignore[arg-type]


def test_none_uses_cpu_count_minus_one() -> None:
    assert parallel._resolve_n_jobs(None) == max(1, (parallel.os.cpu_count() or 1) - 1)


def test_nested_map_is_a_failure_in_the_worker() -> None:
    outcome = map_shapes(_nested, [ShapeTask("a", 1)], n_jobs=1)
    assert outcome.failures
    assert outcome.failures[0].exception_type == "RuntimeError"
    assert "nested parallelism" in outcome.failures[0].message


def test_direct_nested_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "_IN_WORKER", True)
    with pytest.raises(RuntimeError, match="nested parallelism"):
        map_shapes(_double, [ShapeTask("a", 1)], n_jobs=1)


def test_dask_is_not_a_dependency() -> None:
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text().casefold()
    lockfile = (PACKAGE_ROOT / "uv.lock").read_text().casefold()
    assert "dask" not in pyproject
    assert "dask" not in lockfile
