"""On-disk emission output.

Reuses the transport writer's atomic shape-file mechanism, URL-quoted filenames
and manifest conventions instead of duplicating them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gtfs2emis_py.gtfs.writer import (
    ManifestError,
    _file_is_complete,
    _filenames,
    _load_manifest,
    _validate_table,
    _write_json,
    _write_shape,
)

__all__ = [
    "EmissionManifest",
    "EmissionWriteError",
    "read_emission_manifest",
    "write_emission_output",
]

SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"
_SHAPES = "shapes"
_COLUMNS = (
    "segment_id",
    "shape_id",
    "trip_id",
    "stop_sequence",
    "vehicle_type",
    "pollutant",
    "process",
    "value",
    "unit",
)


class EmissionWriteError(RuntimeError):
    """The emission table cannot be written or resumed."""


@dataclass(frozen=True)
class EmissionManifest:
    """A written emission manifest and its directory."""

    manifest: dict
    path: Path


def write_emission_output(
    result: object,
    output_path: str | Path,
    *,
    resume: bool = False,
) -> EmissionManifest:
    """Write one Parquet file per shape plus a manifest.

    Args:
        result: An :class:`EmissionResult` (duck-typed: needs ``emi`` and
            ``metadata``).
        output_path: Destination directory.
        resume: Reuse an existing manifest only when every file is readable and
            its row count matches; otherwise rewrite the complete output.

    Returns:
        The manifest and its directory.

    Raises:
        EmissionWriteError: For a malformed result or an unwritable directory.
        ManifestError: For a manifest that cannot be parsed.
    """
    table = getattr(result, "emi", None)
    metadata = getattr(result, "metadata", None)
    if not isinstance(table, pd.DataFrame):
        raise TypeError("result must expose a pandas DataFrame in .emi")
    if not isinstance(metadata, dict):
        raise TypeError("result must expose a metadata mapping")
    _validate_table(table)
    missing = set(_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"emission table is missing column(s) {sorted(missing)}")
    if (table["value"] < 0).any():
        raise ValueError("emission values must be non-negative")

    output = Path(output_path)
    if output.exists() and not output.is_dir():
        raise EmissionWriteError(f"{output} is not a directory")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EmissionWriteError(f"cannot create {output}: {exc}") from exc
    if not output.is_dir():
        raise EmissionWriteError(f"{output} is not a directory")
    shapes_dir = output / _SHAPES
    shapes_dir.mkdir(exist_ok=True)

    groups = list(table.groupby("shape_id", sort=True))
    ids = [str(shape_id) for shape_id, _ in groups]
    filenames = _filenames(ids)

    request = _request_sha256(metadata, table)
    existing = _load_manifest(output) if resume else None
    if existing is not None and _resume_complete(
        existing, output / _SHAPES, filenames, request, int(len(table))
    ):
        return EmissionManifest(manifest=existing, path=output)

    files = []
    for (shape_id, group), filename in zip(groups, filenames, strict=True):
        path = shapes_dir / filename
        rows = _write_shape(group, path)
        payload = path.read_bytes()
        files.append(
            {
                "shape_id": str(shape_id),
                "file": f"{_SHAPES}/{filename}",
                "rows": rows,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ef_model": metadata.get("ef_model"),
        "pollutant": _as_sorted_list(metadata.get("pollutant")),
        "process": _as_sorted_list(metadata.get("process")),
        "request_sha256": request,
        "rows": int(len(table)),
        "shapes": files,
        "columns": list(_COLUMNS),
    }
    _write_json(output / _MANIFEST, manifest)
    return EmissionManifest(manifest=manifest, path=output)


def read_emission_manifest(output_path: str | Path) -> dict:
    """Read and minimally validate an emission manifest."""
    path = Path(output_path) / _MANIFEST
    if not path.exists():
        raise ManifestError(f"no emission manifest at {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{path} is not a schema-version {SCHEMA_VERSION} emission manifest")
    return manifest


def _as_sorted_list(value: object) -> list[str]:
    """Normalize an optional value to a sorted string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return []
    return sorted(str(item) for item in value)


def _request_sha256(metadata: dict, table: pd.DataFrame) -> str:
    """Hash the emission request identity used for resume."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "ef_model": metadata.get("ef_model"),
                "pollutant": _as_sorted_list(metadata.get("pollutant")),
                "process": _as_sorted_list(metadata.get("process")),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(str(len(table)).encode("ascii"))
    # Row identity alone is not enough: a changed emission value must invalidate
    # the resume rather than reuse a file with the same shape and row count.
    digest.update(
        pd.util.hash_pandas_object(table.astype("string"), index=False)
        .to_numpy(dtype="uint64")
        .tobytes()
    )
    return digest.hexdigest()


def _resume_complete(
    manifest: dict,
    shapes_dir: Path,
    filenames: list[str],
    request_sha256: str,
    rows: int,
) -> bool:
    """Return True when a manifest and all files match the request."""
    if manifest.get("ef_model") is None:
        return False
    if manifest.get("request_sha256") != request_sha256 or manifest.get("rows") != rows:
        return False
    files = manifest.get("shapes")
    if not isinstance(files, list) or len(files) != len(filenames):
        return False
    for entry, filename in zip(files, filenames, strict=True):
        if not isinstance(entry, dict) or entry.get("file") != f"{_SHAPES}/{filename}":
            return False
        if not _file_is_complete(
            shapes_dir / filename,
            int(entry.get("rows", -1)),
            entry.get("sha256"),
        ):
            return False
    return True
