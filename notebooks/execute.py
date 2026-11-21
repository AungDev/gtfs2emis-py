"""Execute every tutorial notebook.

Run from `gtfs2emis-py/`:
    uv run python notebooks/execute.py

CI uses this script; it executes in memory and does not rewrite the committed
notebooks, so a failure is a real execution failure rather than a git diff.

Notebooks that read an EF table which is not redistributable (CETESB, EMFAC,
MOVES) are skipped when that table is absent. Those tutorials therefore run in
a checkout that carries the development tables and are skipped in a release
tree, which publishes only the EMEP table.
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
TABLES = ("brazil_cetesb", "usa_emfac", "usa_moves", "europe_emep")


def packaged(table: str) -> bool:
    """Return True when ``data/ef/<table>.parquet`` ships with this tree."""
    data = resources.files("gtfs2emis_py").joinpath("data", "ef")
    return data.joinpath(f"{table}.parquet").is_file()


def required_tables(notebook: nbformat.NotebookNode) -> list[str]:
    """EF tables named by the notebook's code cells."""
    code = "\n".join(
        "".join(cell.get("source", "")) for cell in notebook.cells if cell.cell_type == "code"
    )
    return [table for table in TABLES if table in code]


def main() -> int:
    notebooks = sorted(HERE.glob("*.ipynb"))
    if not notebooks:
        print("no notebooks found", file=sys.stderr)
        return 1
    executed = skipped = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        missing = [table for table in required_tables(notebook) if not packaged(table)]
        if missing:
            print(f"skipped {path.name}: requires {', '.join(missing)}")
            skipped += 1
            continue
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(BASE)}},
        )
        client.execute()
        print(f"executed {path.name}")
        executed += 1
    print(f"{executed} executed, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
