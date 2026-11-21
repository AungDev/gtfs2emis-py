"""Synthetic transport-to-emission pipeline benchmark.

Run:
    uv run python benchmarks/pipeline_benchmark.py --segments 100000
    uv run python benchmarks/pipeline_benchmark.py --segments 2000 --check

The feed is generated in memory, so the measurement covers the real
``transport_model`` path, the emission path and the per-shape writer rather than
only component microbenchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import gtfs_kit as gk
import pandas as pd

from gtfs2emis_py import emission_model, transport_model
from gtfs2emis_py.emission_writer import write_emission_output

HERE = Path(__file__).resolve().parent
BUDGETS = HERE / "pipeline_budgets.json"


def _synthetic_feed(segments: int) -> gk.Feed:
    """Build an in-memory synthetic city-scale GTFS feed."""
    shapes = max(1, segments // 1000)
    stops_per_shape = 1000
    stop_rows: list[dict[str, object]] = []
    shape_rows: list[dict[str, object]] = []
    stop_time_rows: list[dict[str, object]] = []
    trip_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    for shape_index in range(shapes):
        shape_id = f"S{shape_index:04d}"
        trip_id = f"T{shape_index:04d}"
        route_id = f"R{shape_index:04d}"
        route_rows.append(
            {"route_id": route_id, "agency_id": "A1", "route_short_name": route_id, "route_type": 3}
        )
        trip_rows.append(
            {"route_id": route_id, "service_id": "WEEK", "trip_id": trip_id, "shape_id": shape_id}
        )
        for position in range(stops_per_shape):
            stop_id = f"{shape_id}-P{position:04d}"
            lon = -49.0 + shape_index * 0.02 + position * 0.00001
            lat = -25.0 + shape_index * 0.001
            stop_rows.append(
                {"stop_id": stop_id, "stop_name": stop_id, "stop_lat": lat, "stop_lon": lon}
            )
            shape_rows.append(
                {
                    "shape_id": shape_id,
                    "shape_pt_lat": lat,
                    "shape_pt_lon": lon,
                    "shape_pt_sequence": position,
                }
            )
            seconds = position
            clock = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
            stop_time_rows.append(
                {
                    "trip_id": trip_id,
                    "arrival_time": clock,
                    "departure_time": clock,
                    "stop_id": stop_id,
                    "stop_sequence": position,
                }
            )
    return gk.Feed(
        dist_units="km",
        agency=pd.DataFrame(
            {
                "agency_id": ["A1"],
                "agency_name": ["Synthetic"],
                "agency_url": ["https://example.invalid"],
                "agency_timezone": ["Africa/Johannesburg"],
            }
        ),
        stops=pd.DataFrame(stop_rows),
        routes=pd.DataFrame(route_rows),
        trips=pd.DataFrame(trip_rows),
        stop_times=pd.DataFrame(stop_time_rows),
        shapes=pd.DataFrame(shape_rows),
    )


def _fleet() -> pd.DataFrame:
    """Build the EMEP fleet table used by the pipeline benchmark."""
    vehicles = [
        "Ubus Midi <=15 t",
        "Ubus Std 15 - 18 t",
        "Ubus Artic >18 t",
        "Coaches Std <=18 t",
        "Coaches Artic >18 t",
    ]
    return pd.DataFrame(
        {
            "veh_type": [vehicle for vehicle in vehicles for _euro in ("V", "IV")],
            "euro": ["V", "IV"] * 5,
            "fuel": ["D"] * 10,
            "tech": ["SCR"] * 10,
            "fleet_composition": [0.1] * 10,
        }
    )


def _timed(fn, *args, **kwargs):
    """Run a callable and return its elapsed seconds and result."""
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    return (time.perf_counter() - started) * 1000.0, value


def main() -> int:
    """Run the pipeline benchmark and optionally enforce budgets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=100_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.segments < 100:
        parser.error("--segments must be at least 100")

    feed = _synthetic_feed(args.segments)
    results: dict[str, float] = {"segments": float(args.segments)}
    results["transport_ms"], transport = _timed(transport_model, feed, parallel=True, ncores=2)
    results["transport_rows"] = float(len(transport.table))
    results["emission_ms"], emission = _timed(
        emission_model,
        transport,
        "ef_europe_emep",
        _fleet(),
        ["CO", "NOx"],
        reference_year=2020,
        parallel=True,
        ncores=2,
    )
    results["emission_rows"] = float(len(emission.emi))
    with tempfile.TemporaryDirectory(prefix="gtfs2emis-bench-") as directory:
        started = time.perf_counter()
        write_emission_output(emission, directory)
        results["writer_ms"] = (time.perf_counter() - started) * 1000.0
        results["output_bytes"] = float(
            sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())
        )
        results["output_shapes"] = float(len(list(Path(directory).rglob("*.parquet"))))
    results["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if not args.check:
        return 0
    budgets = json.loads(BUDGETS.read_text())
    failures = [
        f"{key}: {results[key]:.2f} > {limit}"
        for key, limit in budgets.items()
        if key in results and results[key] > float(limit)
    ]
    if failures:
        print("BUDGET FAILURE", failures, file=os.sys.stderr)
        return 1
    print("within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
