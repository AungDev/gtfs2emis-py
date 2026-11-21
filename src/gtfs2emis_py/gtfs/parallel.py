"""Per-shape parallel map.

Every parallel call site in the transport path goes through :func:`map_shapes`.
It is module-private on purpose: the public API exposes ``parallel`` and
``ncores``, not a backend. ``n_jobs=1`` runs serially in-process, which is what
tests and the benchmark use.

Guarantees:

- results keep the input order, not completion order;
- one failing shape does not lose the others — it becomes a
  :class:`ShapeFailure` and the rest of the map completes;
- worker BLAS/OpenMP pools are pinned to one thread;
- no nested parallelism: calling :func:`map_shapes` inside a shape task raises.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from joblib import Parallel, delayed, parallel_config

__all__ = [
    "MapOutcome",
    "ShapeFailure",
    "ShapeProcessingError",
    "ShapeTask",
    "map_shapes",
]

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Set inside a worker while the task runs, so a nested map_shapes call is caught
# even in the serial path.
_IN_WORKER = False


@dataclass(frozen=True)
class ShapeTask:
    """One unit of per-shape work.

    Attributes:
        shape_id: GTFS shape id; the stable ordering and error key.
        payload: A picklable per-shape handle; the transport call site sends a
            mini-:class:`gtfs_kit.Feed`, not the whole feed.
    """

    shape_id: str
    payload: Any


@dataclass(frozen=True)
class ShapeFailure:
    """A shape whose task raised."""

    shape_id: str
    exception_type: str
    message: str
    traceback: str


class ShapeProcessingError(RuntimeError):
    """Aggregate failure raised after every other shape has been computed."""

    def __init__(self, failures: Sequence[ShapeFailure]) -> None:
        """Store the shape failures for a batch error."""
        self.failures = tuple(failures)
        self.shape_ids = tuple(failure.shape_id for failure in self.failures)
        super().__init__(f"{len(self.shape_ids)} shape(s) failed: {', '.join(self.shape_ids)}")


@dataclass(frozen=True)
class MapOutcome(Generic[T]):
    """Results and per-shape failures, both in input order.

    ``results`` has one entry per task; a failed task leaves ``None`` in its
    slot, and the matching :class:`ShapeFailure` is in ``failures``.
    """

    results: tuple[T | None, ...]
    failures: tuple[ShapeFailure, ...]

    def raise_if_failed(self) -> None:
        """Raise :class:`ShapeProcessingError` when any shape failed."""
        if self.failures:
            raise ShapeProcessingError(self.failures)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Return the worker count, validating ``n_jobs``."""
    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        return max(1, cpu_count - 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an int or None, got {type(n_jobs).__name__}")
    if not 1 <= n_jobs <= cpu_count:
        raise ValueError(f"n_jobs must be between 1 and {cpu_count}, got {n_jobs}")
    return n_jobs


def _run_one(
    fn: Callable[[ShapeTask], T], index: int, task: ShapeTask
) -> tuple[int, T | None, ShapeFailure | None]:
    """Run one task, converting an exception into a :class:`ShapeFailure`."""
    global _IN_WORKER
    _IN_WORKER = True
    try:
        return index, fn(task), None
    except Exception as exc:  # noqa: BLE001 - per-shape isolation is the point
        logger.warning("shape %s failed: %s", task.shape_id, exc)
        return (
            index,
            None,
            ShapeFailure(
                shape_id=task.shape_id,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
    finally:
        _IN_WORKER = False


def map_shapes(
    fn: Callable[[ShapeTask], T],
    tasks: Sequence[ShapeTask],
    n_jobs: int | None = None,
) -> MapOutcome[T]:
    """Map ``fn`` over ``tasks``, with per-shape error isolation and stable order.

    Args:
        fn: A module-level (picklable) callable taking one :class:`ShapeTask`.
        tasks: The work items; each carries its shape id and payload.
        n_jobs: Workers. ``None`` means ``max(1, cpu_count() - 1)``; ``1`` runs
            serially in-process. Values above ``cpu_count()`` are rejected.

    Returns:
        A :class:`MapOutcome` with results and failures in input order.

    Raises:
        TypeError: If ``n_jobs`` is not an int or ``None``.
        ValueError: If ``n_jobs`` is outside ``1..cpu_count()``.
        RuntimeError: If called from inside a running shape task (nested
            parallelism is not supported).
    """
    if _IN_WORKER:
        raise RuntimeError("nested parallelism: map_shapes() cannot be called inside a shape task")
    items = tuple(tasks)
    workers = _resolve_n_jobs(n_jobs)
    if workers == 1:
        collected = [_run_one(fn, index, item) for index, item in enumerate(items)]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            collected = Parallel(
                n_jobs=workers,
                prefer="processes",
                return_as="list",
            )(delayed(_run_one)(fn, index, item) for index, item in enumerate(items))
    ordered = sorted(collected, key=lambda item: item[0])
    return MapOutcome(
        results=tuple(result for _index, result, _failure in ordered),
        failures=tuple(failure for _index, _result, failure in ordered if failure),
    )
