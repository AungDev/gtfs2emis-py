"""Documentation-completeness checks for the production tree."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ("src", "data-raw", "benchmarks", "scripts")


def _production_files() -> list[Path]:
    return sorted(
        path for directory in PRODUCTION_DIRS for path in (PACKAGE_ROOT / directory).rglob("*.py")
    )


def test_every_production_function_and_class_has_a_docstring() -> None:
    missing: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}:{node.name}")
    assert not missing, "undocumented production symbols:\n" + "\n".join(missing)


def test_source_comments_do_not_carry_stale_claims() -> None:
    stale = ("CRS null", "crs: null", "all 20 Function_ID", "Reserved for task_")
    offenders: list[str] = []
    for path in _production_files():
        if not path.is_relative_to(PACKAGE_ROOT / "src"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            for phrase in stale:
                if phrase in line:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}:{phrase}")
    assert not offenders, offenders


def test_spn23_is_not_described_as_unavailable_in_r() -> None:
    source = (PACKAGE_ROOT / "src/gtfs2emis_py/ef/europe_emep.py").read_text()
    assert "excluded by the reference" not in source
    assert "SPN23" in source
