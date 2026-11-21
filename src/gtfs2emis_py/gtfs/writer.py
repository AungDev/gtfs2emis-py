"""Per-shape transport output, manifest and resume.

The writer is deliberately boring: one Parquet/GeoParquet file per ``shape_id``,
an atomic rename for every file, and a manifest that records what was written.
``resume=True`` trusts the manifest only for shapes whose file still opens and
still has the recorded row count; anything else is rewritten. A file is never
deleted — a bad shape is recorded as ``failed`` and the rest of the run
continues, mirroring :func:`gtfs2emis_py.gtfs.parallel.map_shapes`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from gtfs2emis_py import __version__

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ManifestError",
    "TransportWriteError",
    "read_transport_manifest",
    "write_transport_output",
]

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (SCHEMA_VERSION,)
_SHAPES_DIR = "shapes"
_MANIFEST = "manifest.json"
_MAX_NAME = 120


class ManifestError(ValueError):
    """An existing manifest cannot be resumed against the current request."""


class TransportWriteError(RuntimeError):
    """At least one shape failed; the manifest is already on disk."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        """Build the error message and retain the manifest with failed entries."""
        self.manifest: dict[str, Any] = dict(manifest)
        failed = [
            entry["shape_id"] for entry in self.manifest["shapes"] if entry["status"] == "failed"
        ]
        super().__init__(f"{len(failed)} shape(s) failed to write: {', '.join(failed)}")


def write_transport_output(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    resume: bool = False,
    parameters: Mapping[str, object] | None = None,
    source_sha256: str | None = None,
    skipped: Mapping[str, int] | None = None,
    warnings: Sequence[str] | None = None,
    crs: str = "EPSG:4326",
) -> dict:
    """Write one Parquet file per shape plus a manifest.

    Args:
        table: The transport table; ``shape_id`` is the split key. A
            ``geometry`` column of shapely geometries is written as WKB with
            GeoParquet metadata; WKB bytes are kept as they are.
        output_path: Directory for ``shapes/`` and ``manifest.json``.
        resume: Skip a shape only when the manifest marks it ``done``, its file
            opens and its row count matches. Unreadable or truncated files are
            rewritten.
        parameters: JSON-serialisable run parameters recorded in the manifest.
        source_sha256: Optional digest of the input feed. When supplied it is
            part of the resume fingerprint, so changed feed bytes are detected
            even when per-shape row counts are unchanged.
        skipped: Optional per-reason counters from the build.
        warnings: Optional build warnings (snap distances, skips).
        crs: CRS of the geometry column, recorded in the GeoParquet metadata.
            Defaults to EPSG:4326, the transport contract's output CRS.

    Returns:
        The manifest as a dict, also written to ``output_path/manifest.json``.

    Raises:
        ValueError: For a table without a usable ``shape_id``, empty ids, or
            filenames that collide case-insensitively.
        ManifestError: When ``resume=True`` and the existing manifest has a
            different schema version, columns or input hash.
        TransportWriteError: After every shape has been attempted, if any
            shape failed. The manifest is already written.
    """
    output = Path(output_path)
    shapes_dir = output / _SHAPES_DIR
    shapes_dir.mkdir(parents=True, exist_ok=True)
    _validate_table(table)
    columns = list(table.columns)
    groups = sorted(
        ((str(shape_id), group) for shape_id, group in table.groupby("shape_id", sort=False)),
        key=lambda item: item[0],
    )
    filenames = _filenames([shape_id for shape_id, _group in groups])
    input_sha256 = _input_sha256(columns, groups, parameters or {}, source_sha256)
    table_sha256 = _table_sha256(groups)
    previous = _load_manifest(output) if resume else None
    if previous is not None:
        if previous["schema_version"] != SCHEMA_VERSION:
            raise ManifestError(
                f"manifest schema_version {previous['schema_version']} != {SCHEMA_VERSION}"
            )
        if previous["columns"] != columns:
            raise ManifestError("existing manifest records a different column set")
        if previous["input_sha256"] != input_sha256:
            raise ManifestError(
                "existing manifest was written from a different input; "
                "rerun with resume=False to overwrite"
            )
        if previous.get("table_sha256") != table_sha256:
            # The feed and parameters are the same, but the values changed
            # (for example a reprocessed speed). Rebuild rather than serving
            # stale rows that happen to have the same shape and row count.
            previous = None
    previous_by_id = (
        {entry["shape_id"]: entry for entry in previous["shapes"]} if previous is not None else {}
    )

    entries: list[dict] = []
    rows_written = 0
    shapes_resumed = 0
    for (shape_id, group), filename in zip(groups, filenames, strict=True):
        path = shapes_dir / filename
        known = previous_by_id.get(shape_id)
        if (
            known is not None
            and known["status"] == "done"
            and _file_is_complete(path, known["rows"], known.get("sha256"))
        ):
            entries.append(known)
            shapes_resumed += 1
            continue
        try:
            rows = _write_shape(group, path, crs=crs)
        except Exception as exc:  # noqa: BLE001 - per-shape isolation is the point
            entries.append(
                {
                    "shape_id": shape_id,
                    "file": filename,
                    "rows": 0,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        entries.append(
            {
                "shape_id": shape_id,
                "file": filename,
                "rows": rows,
                "sha256": _file_sha256(path),
                "status": "done",
                "error": None,
            }
        )
        rows_written += rows

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_sha256": input_sha256,
        "table_sha256": table_sha256,
        "columns": columns,
        "parameters": dict(parameters or {}),
        "skipped": dict(skipped or {}),
        "warnings": list(warnings or ()),
        "shapes": entries,
        "rows_total": int(len(table)),
        "rows_written": rows_written,
        "shapes_resumed": shapes_resumed,
        "shapes_total": len(entries),
    }
    _write_json(output / _MANIFEST, manifest)
    if any(entry["status"] == "failed" for entry in entries):
        raise TransportWriteError(manifest)
    return manifest


def read_transport_manifest(output_path: str | Path) -> dict:
    """Read and validate ``output_path/manifest.json``.

    Only ``SUPPORTED_SCHEMA_VERSIONS`` are accepted; a future or malformed
    version raises :class:`ManifestError` naming the supported range.
    """
    manifest = _load_manifest(Path(output_path))
    if manifest is None:
        raise ManifestError(f"no manifest under {output_path}")
    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in SUPPORTED_SCHEMA_VERSIONS)
        raise ManifestError(
            f"manifest schema_version {version!r} is not supported "
            f"(supported schema versions: {supported})"
        )
    return manifest


def _validate_table(table: pd.DataFrame) -> None:
    """Validate shape_id and table shape before writing."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if "shape_id" not in table.columns:
        raise ValueError("table has no 'shape_id' column")
    if table.empty:
        raise ValueError("table is empty")
    ids = table["shape_id"]
    if ids.isna().any():
        raise ValueError("shape_id must not contain nulls")
    if (ids.astype(str).str.len() == 0).any():
        raise ValueError("shape_id must not contain empty strings")


def _filenames(shape_ids: Iterable[str]) -> list[str]:
    """Map shape ids to safe, unique Parquet filenames."""
    names: list[str] = []
    for shape_id in shape_ids:
        if "\x00" in shape_id:
            raise ValueError("shape_id must not contain NUL")
        encoded = _quote(shape_id)
        if len(encoded) > _MAX_NAME:
            encoded = (
                encoded[: _MAX_NAME - 17]
                + "-"
                + hashlib.sha256(shape_id.encode("utf-8")).hexdigest()[:16]
            )
        names.append(encoded + ".parquet")
    lowered = [name.casefold() for name in names]
    if len(set(lowered)) != len(lowered):
        raise ValueError("shape_id values produce filenames that collide case-insensitively")
    return names


def _quote(value: str) -> str:
    """URL-quote a shape id for a filename."""
    return quote(value, safe="")


def _write_shape(group: pd.DataFrame, path: Path, *, crs: str = "EPSG:4326") -> int:
    """Write one shape atomically and return its row count."""
    frame = group.copy(deep=True)
    sort_columns = [
        column for column in ("trip_id", "trip_number", "stop_sequence") if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable")
    geometry = None
    if "geometry" in frame.columns and not frame.empty:
        first = frame["geometry"].iloc[0]
        if isinstance(first, BaseGeometry):
            geometry = [
                item.wkb if isinstance(item, BaseGeometry) else item for item in frame["geometry"]
            ]
            # Plain DataFrame assignment avoids geopandas' warning about
            # replacing the active geometry column with bytes.
            frame = pd.DataFrame(frame)
            frame["geometry"] = geometry
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if geometry is not None:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), b"geo": _geo_metadata(crs).encode("utf-8")}
        )
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(frame)


def _geo_metadata(crs: str = "EPSG:4326") -> str:
    """Return GeoParquet metadata with a real PROJJSON CRS.

    A null CRS makes a "GeoParquet" file unusable without an out-of-band
    convention, so the transport writer always records EPSG:4326 unless the
    caller explicitly supplied another CRS.
    """
    from pyproj import CRS

    return json.dumps(
        {
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["LineString"],
                    "crs": CRS.from_user_input(crs).to_json_dict(),
                }
            },
        },
        sort_keys=True,
    )


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_is_complete(path: Path, expected_rows: int, expected_sha256: str | None = None) -> bool:
    """Return True when a Parquet file has the expected rows and digest."""
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:  # noqa: BLE001 - any unreadable file is repairable
        return False
    if metadata is None or metadata.num_rows != expected_rows:
        return False
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        return False
    return True


def _table_sha256(groups: Iterable[tuple[str, pd.DataFrame]]) -> str:
    """Hash the table's values, independent of row/source order in the manifest.

    The writer already receives the table in the caller's deterministic order;
    sort each shape by the same keys it writes by, stringify once and hash the
    pandas row hashes. The digest is intentionally over values, not Parquet
    bytes, so a codec change does not invalidate a resume, while a speed or
    geometry change does.
    """
    digest = hashlib.sha256()
    for shape_id, group in groups:
        frame = group.copy(deep=True)
        sort_columns = [
            column
            for column in ("trip_id", "trip_number", "stop_sequence")
            if column in frame.columns
        ]
        if sort_columns:
            frame = frame.sort_values(sort_columns, kind="stable")
        frame = frame.reset_index(drop=True)
        if "geometry" in frame.columns:
            # Convert to a plain DataFrame before replacing the active geometry
            # column; assigning strings into a GeoDataFrame geometry column
            # emits ``UserWarning: Geometry column does not contain geometry``
            # once per shape. The values and digest stay identical.
            frame = pd.DataFrame(frame)
            frame["geometry"] = [
                item.wkb.hex() if isinstance(item, BaseGeometry) else str(item)
                for item in frame["geometry"]
            ]
        digest.update(str(shape_id).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(frame.astype("string"), index=False)
            .to_numpy(dtype="uint64")
            .tobytes()
        )
    return digest.hexdigest()


def _fingerprint_parameters(value: object) -> object:
    """Return parameters without run-control flags that do not affect output.

    ``resume`` changes how a run starts, not what a completed shape contains;
    including it would make the second ``transport_model(..., resume=True)``
    call fail its own resume fingerprint.
    """
    if isinstance(value, Mapping):
        return {
            key: _fingerprint_parameters(item) for key, item in value.items() if key != "resume"
        }
    if isinstance(value, list | tuple):
        return [_fingerprint_parameters(item) for item in value]
    return value


def _input_sha256(
    columns: Iterable[str],
    groups: Iterable[tuple[str, pd.DataFrame]],
    parameters: Mapping[str, object],
    source_sha256: str | None = None,
) -> str:
    """Hash the request inputs used for resume compatibility."""
    digest = hashlib.sha256()
    digest.update(",".join(columns).encode("utf-8"))
    digest.update(
        json.dumps(_fingerprint_parameters(parameters), sort_keys=True, default=str).encode()
    )
    if source_sha256 is not None:
        digest.update(str(source_sha256).encode("ascii"))
    for shape_id, group in groups:
        digest.update(f"\n{shape_id}\t{len(group)}".encode())
    return digest.hexdigest()


def _load_manifest(output: Path) -> dict | None:
    """Load and minimally validate an existing manifest, if present."""
    path = output / _MANIFEST
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict) or "schema_version" not in manifest:
        raise ManifestError(f"{path} is not a transport manifest")
    return manifest


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON atomically via a temporary sibling."""
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
