"""Micro-benchmarks for the hot paths.

Run:
    uv run python benchmarks/benchmark.py --segments 20000
    uv run python benchmarks/benchmark.py --segments 5000 --check

`--check` compares wall time and parent peak RSS with `benchmarks/budgets.json`
and exits non-zero on a regression. CI runs the smaller size so the step stays
under a minute.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from gtfs2emis_py.ef import ef_europe_emep
from gtfs2emis_py.gtfs.parallel import ShapeTask, map_shapes
from gtfs2emis_py.gtfs.speed import adjust_speeds
from gtfs2emis_py.multiply import multiply_ef

HERE = Path(__file__).resolve().parent
BUDGETS = HERE / "budgets.json"
POLLUTANT = "CO"
VEHICLE = "Ubus Std 15 - 18 t"


def _transport(segments: int) -> gpd.GeoDataFrame:
    """Build a synthetic transport table for the component benchmark."""
    shapes = max(1, segments // 100)
    shape_index = np.repeat(np.arange(shapes), 100)[:segments]
    stop_sequence = np.tile(np.arange(100), shapes)[:segments]
    speed = 10.0 + (np.arange(segments) % 50)
    return gpd.GeoDataFrame(
        {
            "shape_id": [f"S{index:05d}" for index in shape_index],
            "trip_id": [f"T{index:05d}" for index in shape_index],
            "route_id": [f"R{index:03d}" for index in shape_index],
            "stop_sequence": stop_sequence,
            "from_service_day_seconds": (stop_sequence * 60) % 86400,
            "to_service_day_seconds": ((stop_sequence + 1) * 60) % 86400,
            "cumtime_s": (stop_sequence * 60).astype(float),
            "speed_kmh": speed.astype(float),
            "dist_km": np.full(segments, 0.1),
        },
        geometry=[
            LineString([(index * 1e-5, 0.0), ((index + 1) * 1e-5, 0.0)])
            for index in range(segments)
        ],
        crs="EPSG:4326",
    )


def _work(task: ShapeTask) -> float:
    """Run the benchmark worker for one payload."""
    total = 0.0
    for value in task.payload:
        total += float(value) ** 2
    return total


def _timed(fn, *args, **kwargs) -> tuple[float, object]:
    """Run a callable and return its elapsed seconds and result."""
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    return (time.perf_counter() - started) * 1000.0, value


def main() -> int:
    """Run the component benchmark and optionally enforce budgets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=20_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.segments < 100:
        parser.error("--segments must be at least 100")

    table = _transport(args.segments)
    speeds = table["speed_kmh"].to_numpy(dtype=float)
    dist = table["dist_km"].to_numpy(dtype=float)
    constant_ef = pd.DataFrame(
        {"pollutant": [POLLUTANT], "vehicle_index": [0], "ef": [1.0], "unit": ["g/km"]}
    )
    tasks = [
        ShapeTask(shape_id=f"S{index:05d}", payload=tuple(range(200)))
        for index in range(max(1, args.segments // 100))
    ]

    results: dict[str, float] = {}
    results["adjust_speeds_ms"], _ = _timed(adjust_speeds, table)
    results["ef_emep_ms"], _ = _timed(ef_europe_emep, speeds, VEHICLE, "V", [POLLUTANT])
    results["multiply_ef_ms"], _ = _timed(multiply_ef, [1.0], dist, constant_ef)
    workers = max(1, (os.cpu_count() or 1) - 1)
    results["parallel_map_ms"], _ = _timed(map_shapes, _work, tasks, workers)
    results["serial_map_ms"], _ = _timed(map_shapes, _work, tasks, 1)
    results["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    results["segments"] = float(args.segments)

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    if not args.check:
        return 0
    budgets = json.loads(BUDGETS.read_text())
    failures = []
    for key, limit in budgets.items():
        if key == "segments":
            continue
        value = results.get(key)
        if value is None:
            continue
        if value > float(limit):
            failures.append(f"{key}: {value:.2f} > {limit}")
    if failures:
        print("BUDGET FAILURE", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
