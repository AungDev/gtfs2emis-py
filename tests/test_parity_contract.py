"""Guard the exact-parity contract.

The parity suites are a merge gate. No model-output assertion may use a
tolerance helper; cumulative/structural exclusions must be keyed explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
PARITY_FIXTURES = Path(__file__).parents[2] / "parity/fixtures"
pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(
        not PARITY_FIXTURES.exists(),
        reason="development-only parity fixtures are not published",
    ),
]
PARITY_FILES = sorted(TESTS.glob("test_parity_*.py"))
TOLERANCE = re.compile(r"\b(?:abs|rel|atol|rtol)\s*=\s*(?!0(?:\.0*)?\b)")


def test_parity_suites_have_no_model_tolerances() -> None:
    assert PARITY_FILES, "no parity suites found"
    violations: list[str] = []
    for path in PARITY_FILES:
        if path.name == "test_parity_contract.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "pytest.approx" in line or "assert_allclose" in line:
                violations.append(f"{path.name}:{number}: {line.strip()}")
            elif TOLERANCE.search(line) and "np.isclose" not in line:
                violations.append(f"{path.name}:{number}: {line.strip()}")
    assert not violations, "parity tolerance helpers remain:\n" + "\n".join(violations)


def test_parity_files_are_marked() -> None:
    for path in PARITY_FILES:
        text = path.read_text()
        assert "pytest.mark.parity" in text, f"{path.name} is not collected by the parity job"


def test_parity_manifests_use_exact_comparison() -> None:
    manifests = [
        Path(__file__).parents[1] / "../parity/fixtures/transport/manifest.json",
        Path(__file__).parents[1] / "../parity/fixtures/ef/manifest.json",
        Path(__file__).parents[1] / "../parity/fixtures/emission/manifest.json",
    ]
    for path in manifests:
        data = json.loads(path.read_text())
        assert "tolerances" not in data, f"{path} still has a tolerance block"
        comparison = data["comparison"]
        assert comparison["precision"] == 6, f"{path} precision"
        assert "exact equality" in comparison["comparison"], f"{path} comparison"
        assert comparison["exclusions"], f"{path} exclusions"


def test_documented_fixture_hashes_match_files() -> None:
    parity_doc = Path(__file__).parents[2] / "docs/parity.md"
    fixtures = Path(__file__).parents[2] / "parity/fixtures"
    rows = re.findall(r"\| `([^`]+\.csv)` \| `([0-9a-f]{64})` \|", parity_doc.read_text())
    assert len(rows) == 5, f"expected five fixture hashes, found {len(rows)}"
    for relative, expected in rows:
        path = fixtures / relative
        assert path.is_file(), f"missing fixture {path}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"{relative}: documented {expected}, actual {actual}"
