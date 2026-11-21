"""Smoke test for the component benchmark budget gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def test_component_benchmark_runs_and_stays_within_budget() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark.py",
            "--segments",
            "100",
            "--check",
        ],
        cwd=PACKAGE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "within budget" in completed.stdout
