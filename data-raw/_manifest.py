"""Shared helpers for the EF Parquet build scripts.

Each ``data-raw/ef_<source>_db.py`` declares a :class:`TableSpec` and calls
:func:`run`. The helpers read the R ``.rda`` binary, cast it to a declared Arrow
schema, enforce the nullability rule, write the Parquet file and update
``src/gtfs2emis_py/data/ef/manifest.json``.

The manifest is the data's bill of materials: source, edition, retrieval date,
licence, file hash, value-level ``content_sha256``, the raw input's hash and the
ordered Arrow schema. The loader verifies the file hash, row count and schema at
load time; ``--check-packaged`` re-verifies the packaged file without the raw
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_VERSION: Final = 1

PACKAGE_ROOT: Final = Path(__file__).resolve().parents[1]
EF_DATA_DIR: Final = PACKAGE_ROOT / "src" / "gtfs2emis_py" / "data" / "ef"
MANIFEST_PATH: Final = EF_DATA_DIR / "manifest.json"
RAW_DIR: Final = Path(__file__).resolve().parent / "raw"


@dataclass(frozen=True)
class TableSpec:
    """One EF table's identity, provenance and schema."""

    name: str
    r_object: str
    source: str
    edition: str
    source_url: tuple[str, ...]
    licence: str
    raw_filename: str
    build_script: str
    schema: tuple[tuple[str, pa.DataType, bool], ...]

    def arrow_schema(self) -> pa.Schema:
        """Return the declared Arrow schema."""
        return pa.schema(
            [pa.field(column, dtype, nullable=nullable) for column, dtype, nullable in self.schema]
        )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(table: pa.Table) -> str:
    """Hash values, nulls and column order, independent of Arrow chunk layout.

    Typed value hashing avoids null-bitmap differences between a table built
    from pandas and the same values read back from Parquet.
    """
    digest = hashlib.sha256()
    for column in table.columns:
        name = column._name
        digest.update(name.encode("utf-8"))
        digest.update(str(column.type).encode("utf-8"))
        for value in column.to_pylist():
            if value is None:
                digest.update(b"\x00")
            else:
                digest.update(b"\x01")
                digest.update(repr(value).encode("utf-8"))
            digest.update(b"\x1f")
    return digest.hexdigest()


def read_rda(path: Path, spec: TableSpec) -> pd.DataFrame:
    """Read the R object ``spec.r_object`` out of an ``.rda`` file.

    ``pyreadr`` is a build-time dependency (the ``data`` extra), not a runtime
    one; the import is local so ``--check-packaged`` works without it.
    """
    try:
        import pyreadr
    except ModuleNotFoundError as exc:  # pragma: no cover - build-time only
        raise SystemExit(
            "pyreadr is required to rebuild EF tables; run `uv sync --extra data` first"
        ) from exc
    result = pyreadr.read_r(str(path))
    if spec.r_object not in result:
        raise SystemExit(f"{path} does not contain {spec.r_object!r}; it has {sorted(result)}")
    return result[spec.r_object]


def build_table(spec: TableSpec, raw_path: Path) -> pa.Table:
    """Read the raw ``.rda`` and return the declared Arrow table."""
    frame = read_rda(raw_path, spec)
    expected = [column for column, _, _ in spec.schema]
    if list(frame.columns) != expected:
        raise SystemExit(
            f"{spec.name}: columns differ from the declared schema\n"
            f"  expected: {expected}\n  found:    {list(frame.columns)}"
        )
    for column, dtype, _ in spec.schema:
        if pa.types.is_integer(dtype):
            frame[column] = frame[column].astype("int32")
        elif pa.types.is_floating(dtype):
            frame[column] = frame[column].astype("float64")
        elif pa.types.is_string(dtype):
            frame[column] = frame[column].astype("string")
    table = pa.Table.from_pandas(frame, schema=spec.arrow_schema(), preserve_index=False)
    for column, _dtype, nullable in spec.schema:
        if not nullable:
            nulls = table.column(column).null_count
            if nulls:
                raise SystemExit(
                    f"{spec.name}: column {column!r} has {nulls} null(s) but the "
                    "schema declares it non-nullable"
                )
    return table


def _manifest_entry(spec: TableSpec, table: pa.Table, parquet_path: Path, raw_path: Path) -> dict:
    """Build one manifest entry from a built EF table."""
    return {
        "file": parquet_path.name,
        "r_object": spec.r_object,
        "rows": table.num_rows,
        "columns": table.num_columns,
        "sha256": file_sha256(parquet_path),
        "content_sha256": content_sha256(table),
        "source": spec.source,
        "edition": spec.edition,
        "source_url": list(spec.source_url),
        "retrieved": date.today().isoformat(),
        "licence": {
            "europe_emep": "CC-BY-4.0 (see LICENSES/europe_emep.txt)",
            "brazil_cetesb": "NOASSERTION (see LICENSES/brazil_cetesb.txt)",
            "usa_emfac": "NOASSERTION (see LICENSES/usa_emfac.txt)",
            "usa_moves": "NOASSERTION (GPL-derived; see LICENSES/usa_moves.txt)",
        }[spec.name],
        "build_script": spec.build_script,
        "raw_inputs": [
            {
                "name": raw_path.name,
                "sha256": file_sha256(raw_path),
                "origin": (
                    "https://raw.githubusercontent.com/ipeaGIT/gtfs2emis/"
                    "0dbca7d779857145888ab695f36dc9e71ff57d98/data/"
                    f"{raw_path.name}"
                ),
            }
        ],
        "pyarrow_version": pa.__version__,
        "schema": [
            {"name": column, "type": str(dtype), "nullable": nullable}
            for column, dtype, nullable in spec.schema
        ],
    }


def _add_remote(entry: dict) -> dict:
    """Mark non-EMEP tables as fetch-only in the manifest."""
    if entry["file"] != "europe_emep.parquet":
        raw = entry["raw_inputs"][0]
        entry["remote"] = {
            "file": entry["file"],
            "sha256": entry["sha256"],
            "url": raw["origin"],
            "format": "rda",
            "raw_file": raw["name"],
            "raw_sha256": raw["sha256"],
            "base_url_env": "GTFS2EMIS_DATA_URL",
        }
    return entry


def read_manifest() -> dict:
    """Return the current manifest, or an empty one."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"schema_version": MANIFEST_VERSION, "tables": {}}


def write_manifest(manifest: dict) -> None:
    """Write the manifest deterministically."""
    manifest["tables"] = dict(sorted(manifest["tables"].items()))
    EF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def write(spec: TableSpec, raw_path: Path) -> None:
    """Build the table, write the Parquet file and update the manifest."""
    table = build_table(spec, raw_path)
    EF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = EF_DATA_DIR / f"{spec.name}.parquet"
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        use_dictionary=True,
        version="2.6",
    )
    manifest = read_manifest()
    manifest["schema_version"] = MANIFEST_VERSION
    manifest["tables"][spec.name] = _add_remote(
        _manifest_entry(spec, table, parquet_path, raw_path)
    )
    write_manifest(manifest)
    print(f"wrote {parquet_path} ({table.num_rows} rows, {table.num_columns} columns)")


def _verify_parquet(spec: TableSpec, entry: dict) -> None:
    """Verify a packaged Parquet file against its manifest entry."""
    parquet_path = EF_DATA_DIR / entry["file"]
    if not parquet_path.exists():
        raise SystemExit(f"{parquet_path} is missing; run the build script with --write")
    actual_sha = file_sha256(parquet_path)
    if actual_sha != entry["sha256"]:
        raise SystemExit(
            f"{parquet_path.name}: sha256 mismatch (manifest {entry['sha256']}, file {actual_sha})"
        )
    metadata = pq.read_metadata(parquet_path)
    if metadata.num_rows != entry["rows"]:
        raise SystemExit(
            f"{parquet_path.name}: row count {metadata.num_rows} != manifest {entry['rows']}"
        )
    schema = pq.read_schema(parquet_path)
    expected = spec.arrow_schema()
    if schema.names != expected.names:
        raise SystemExit(
            f"{parquet_path.name}: columns {schema.names} != declared {expected.names}"
        )
    for field, expected_field in zip(schema, expected, strict=False):
        if field.type != expected_field.type:
            raise SystemExit(
                f"{parquet_path.name}: column {field.name!r} has type {field.type}, "
                f"expected {expected_field.type}"
            )
    table = pq.read_table(parquet_path)
    actual_content = content_sha256(table)
    if actual_content != entry["content_sha256"]:
        raise SystemExit(
            f"{parquet_path.name}: content_sha256 mismatch "
            f"(manifest {entry['content_sha256']}, file {actual_content})"
        )


def check_packaged(spec: TableSpec) -> None:
    """Verify the packaged Parquet against the manifest (no raw input needed)."""
    manifest = read_manifest()
    entry = manifest["tables"].get(spec.name)
    if entry is None:
        raise SystemExit(f"{spec.name} is absent from {MANIFEST_PATH}")
    _verify_parquet(spec, entry)
    print(f"{spec.name}: {entry['rows']} rows, sha256 and schema match the manifest")


def check(spec: TableSpec, raw_path: Path) -> None:
    """Rebuild from the raw input and compare the value-level hash."""
    manifest = read_manifest()
    entry = manifest["tables"].get(spec.name)
    if entry is None:
        raise SystemExit(f"{spec.name} is absent from {MANIFEST_PATH}")
    table = build_table(spec, raw_path)
    actual = content_sha256(table)
    if actual != entry["content_sha256"]:
        raise SystemExit(
            f"{spec.name}: content_sha256 mismatch (manifest {entry['content_sha256']}, "
            f"rebuilt {actual})"
        )
    print(f"{spec.name}: content_sha256 matches the manifest ({actual[:12]}…)")


def run(spec: TableSpec, argv: list[str] | None = None) -> int:
    """Command-line entry point shared by the four build scripts."""
    parser = argparse.ArgumentParser(description=f"Build the {spec.name} EF table.")
    parser.add_argument("--raw", type=Path, help="path to the .rda file or its directory")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="build and write Parquet + manifest")
    action.add_argument("--check", action="store_true", help="rebuild and compare content_sha256")
    action.add_argument(
        "--check-packaged", action="store_true", help="verify the packaged Parquet and manifest"
    )
    args = parser.parse_args(argv)

    if args.check_packaged:
        check_packaged(spec)
        return 0
    raw = args.raw
    if raw is None:
        env_raw = os.environ.get("EF_RAW_DIR")
        if env_raw:
            raw = Path(env_raw)
    if raw is None:
        parser.error("--raw is required with --write/--check (or set EF_RAW_DIR)")
    raw_path = raw / spec.raw_filename if raw.is_dir() else raw
    if not raw_path.exists():
        parser.error(f"raw input not found: {raw_path}")
    if args.write:
        write(spec, raw_path)
    else:
        check(spec, raw_path)
    return 0


def main(spec: TableSpec, argv: list[str] | None = None) -> int:
    """Alias kept for the four scripts' ``raise SystemExit(main(SPEC))``."""
    return run(spec, argv)


if __name__ == "__main__":  # pragma: no cover - helper, not an entry point
    sys.exit("run one of the ef_<source>_db.py scripts instead")
