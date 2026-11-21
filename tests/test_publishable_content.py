"""Guard publishable files against development-only references."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SKIP_PARTS = {"_build", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".ipynb", ".json", ".yml", ".yaml"}
SELF = Path(__file__).resolve()
# Workflow files ship verbatim with the release tree. In the development tree
# they sit at the repository root; in a flattened public tree they sit under the
# package root, so both candidates are scanned.
WORKFLOW_DIRS = (
    ROOT / ".github" / "workflows",
    ROOT.parent / ".github" / "workflows",
)
# Release-facing root files. The development repository keeps CHANGELOG.md and
# .gitignore at its root while the package README stays inside the package
# directory, so only the package README is read from ROOT.
ROOT_FILES = (ROOT / "README.md", ROOT / "pyproject.toml")
EXTRA_ROOT_FILES = tuple(
    base / name for base in (ROOT, ROOT.parent) for name in ("CHANGELOG.md", ".gitignore")
)
FORBIDDEN = (
    r"task_\d+",
    r"review_\d+",
    r"ADR-\d{4}",
    r"README \d",
    r"Phase \d",
    r"gitea",
    r"AungDev/gtfs2emis-py-dev",
    r"/home/aung",
)


def _text_files(directory: Path) -> list[Path]:
    """Return the publishable text files under ``directory``."""
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.resolve() == SELF:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        if SKIP_PARTS.intersection(path.parts):
            continue
        found.append(path)
    return found


def _publishable_files() -> list[Path]:
    candidates = [*ROOT_FILES, *EXTRA_ROOT_FILES]
    for directory in ("src", "docs", "tests", "notebooks", "benchmarks", "scripts", "data-raw"):
        candidates.extend(_text_files(ROOT / directory))
    for workflows in WORKFLOW_DIRS:
        candidates.extend(_text_files(workflows))
    files: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if not resolved.is_file() or resolved in seen:
            continue
        seen.add(resolved)
        files.append(path)
    return files


def _display(path: Path) -> str:
    """Path relative to the package root, else to the repository root."""
    for base in (ROOT, ROOT.parent):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def test_publishable_files_contain_no_development_markers() -> None:
    forbidden = re.compile("|".join(FORBIDDEN))
    offenders: list[str] = []
    for path in _publishable_files():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if forbidden.search(text):
            offenders.append(_display(path))
    assert not offenders, f"development-only references in publishable files: {offenders}"
