"""Sphinx API coverage checks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
API = PACKAGE_ROOT / "docs" / "api.md"


def _public_modules() -> set[str]:
    modules: set[str] = set()
    for path in (PACKAGE_ROOT / "src" / "gtfs2emis_py").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        module = ".".join(path.relative_to(PACKAGE_ROOT / "src").with_suffix("").parts)
        modules.add(module)
    return modules


def test_every_public_module_has_an_autodoc_entry() -> None:
    documented = set(re.findall(r"\.\. automodule:: ([\w.]+)", API.read_text()))
    missing = sorted(_public_modules() - documented)
    assert not missing, f"modules missing from docs/api.md: {missing}"


def test_public_constants_are_in_the_api_reference() -> None:
    api = API.read_text()
    missing: list[str] = []
    for path in (PACKAGE_ROOT / "src" / "gtfs2emis_py").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        exported: set[str] = set()
        assigned: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                assigned.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
                if any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                ):
                    try:
                        exported.update(ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        pass
        for name in sorted(exported & assigned):
            if name not in api:
                missing.append(f"{path.name}:{name}")
    assert not missing, f"public constants missing from docs/api.md: {missing}"


def test_public_type_aliases_are_in_the_api_reference() -> None:
    api = API.read_text()
    for name in (
        "Process",
        "FleetMode",
        "DistanceUnit",
        "SpeedUnit",
        "EmissionFactorUnit",
        "MassUnit",
    ):
        assert name in api
