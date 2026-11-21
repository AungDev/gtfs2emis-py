"""EF table loader.

Only the EMEP/EEA table ships inside the wheel. CETESB, EMFAC and MOVES are
fetch-on-request tables: their manifest entries point at the pinned upstream
``.rda`` files, and the loader downloads the original bytes, verifies the raw
SHA-256, converts the R object to the declared Arrow schema, and verifies the
value-level ``content_sha256``. A changed source is rejected with an actionable
hash error; it is never silently accepted.

Fetching a converted table needs the optional ``data`` extra (``pyreadr``).
There is no network access at import time. ``load_ef_table(..., fetch=False)``
(the default) never downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["EFDataError", "list_ef_tables", "load_ef_table"]

_MANIFEST_NAME = "manifest.json"
_DATA_URL_ENV = "GTFS2EMIS_DATA_URL"
_CACHE_ENV = "GTFS2EMIS_CACHE"

# Process-level cache: the EF tables are read once per emission-model call at
# most, and callers receive a copy so the cache cannot be mutated.
_CACHE: dict[str, pd.DataFrame] = {}


class EFDataError(RuntimeError):
    """A packaged EF table is missing or does not match its manifest."""


def _data_dir() -> Any:
    """Return the package-data directory through ``importlib.resources``."""
    return resources.files("gtfs2emis_py.data.ef")


def _read_manifest() -> dict[str, Any]:
    """Read the packaged EF manifest and validate its basic shape."""
    manifest_path = _data_dir().joinpath(_MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise EFDataError(f"{_MANIFEST_NAME} is missing from the package data") from exc
    except json.JSONDecodeError as exc:
        raise EFDataError(f"{_MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise EFDataError(f"unsupported manifest schema_version {manifest.get('schema_version')!r}")
    return manifest


def _resolve(manifest: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a table name or alias to its manifest key and entry."""
    tables = manifest.get("tables", {})
    for key, entry in tables.items():
        aliases = {
            key,
            entry["file"].removesuffix(".parquet"),
            entry["r_object"],
        }
        if name in aliases:
            return key, entry
    raise EFDataError(f"unknown EF table {name!r}; available tables are {sorted(tables)}")


def _cache_dir() -> Path:
    """Return the verified-download cache directory (created on demand)."""
    configured = os.environ.get(_CACHE_ENV)
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "gtfs2emis-py"
    return root / "ef"


def _cache_path(entry: dict[str, Any]) -> Path:
    """Return the cache path for a table's Parquet file."""
    return _cache_dir() / entry["file"]


def _hash_error(key: str, expected: str, actual: str, origin: str) -> EFDataError:
    """Build an EFDataError that reports expected and actual hashes."""
    return EFDataError(
        f"{key}: expected {expected}, got {actual} from {origin}; "
        "update the manifest deliberately after reviewing the new data edition"
    )


def _remote_url(entry: dict[str, Any]) -> str | None:
    """Return the pinned remote URL, if the table is fetchable."""
    remote = entry.get("remote")
    if not isinstance(remote, dict):
        return None
    configured = remote.get("url")
    if isinstance(configured, str) and configured:
        return configured
    # Compatibility path for mirrors that publish the built Parquet files.
    base = os.environ.get(_DATA_URL_ENV, "").rstrip("/")
    if not base:
        return None
    filename = str(remote.get("file") or entry["file"])
    return f"{base}/{filename}"


def _raw_file(entry: dict[str, Any]) -> str:
    """Return the original ``.rda`` filename for a converted remote table."""
    remote = entry.get("remote")
    if isinstance(remote, dict) and remote.get("raw_file"):
        return str(remote["raw_file"])
    return str(entry["file"]).removesuffix(".parquet") + ".rda"


def _raw_cache_path(entry: dict[str, Any]) -> Path:
    """Return the cache path for the table's raw .rda input."""
    return _cache_dir() / _raw_file(entry)


def _raw_sha256(entry: dict[str, Any]) -> str:
    """Return the pinned SHA-256 of the raw .rda input."""
    remote = entry.get("remote")
    if not isinstance(remote, dict) or not remote.get("raw_sha256"):
        raise EFDataError(f"{entry['file']}: remote raw_sha256 is missing from the manifest")
    return str(remote["raw_sha256"])


def _content_sha256(table: pa.Table) -> str:
    """Hash values, nulls and column order, independent of Arrow chunk layout.

    Typed value hashing avoids null-bitmap differences between a table built
    from pandas and the same values read back from Parquet, which the old
    buffer-level hash treated as different data.
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


def _download_and_cache(
    entry: dict[str, Any], key: str, url: str, expected_sha: str, destination: Path
) -> bytes:
    """Download a pinned file, verify it, and atomically place it in the cache."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - explicit data URL
            payload = response.read()
    except OSError as exc:
        raise EFDataError(f"{key}: could not fetch {url}: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha:
        raise _hash_error(key, expected_sha, actual, url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{key}-", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def _raw_payload(entry: dict[str, Any], key: str, *, fetch: bool) -> bytes:
    """Read the original R data file from cache, or fetch it on request."""
    cached = _raw_cache_path(entry)
    expected = _raw_sha256(entry)
    if cached.is_file():
        payload = cached.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise _hash_error(key, expected, actual, str(cached))
        return payload
    if not fetch:
        raise EFDataError(
            f"{key}: table is not packaged and has no verified cache entry; "
            f"call load_ef_table({key!r}, fetch=True) to fetch it from the original "
            f"source URL ({entry.get('licence')})"
        )
    url = _remote_url(entry)
    if url is None:
        raise EFDataError(
            f"{key}: no remote URL is configured for {_raw_file(entry)}; "
            "the manifest must contain remote.url"
        )
    return _download_and_cache(entry, key, url, expected, cached)


def _rda_to_table(payload: bytes, entry: dict[str, Any], key: str) -> pa.Table:
    """Convert the upstream R ``.rda`` object to the manifest's Arrow schema."""
    try:
        import pyreadr
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise EFDataError(
            f"{key}: fetching the original .rda requires pyreadr; install "
            "`gtfs2emis-py[data]` (or `pip install pyreadr`) and retry"
        ) from exc

    handle, temporary = tempfile.mkstemp(prefix=f".{key}-", suffix=".rda")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        try:
            result = pyreadr.read_r(temporary)
        except Exception as exc:
            raise EFDataError(f"{key}: could not read the upstream .rda file: {exc}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    r_object = entry["r_object"]
    if r_object not in result:
        raise EFDataError(
            f"{key}: upstream .rda does not contain {r_object!r}; it has {sorted(result)}"
        )
    frame = result[r_object]
    expected_names = [column["name"] for column in entry["schema"]]
    if list(frame.columns) != expected_names:
        raise EFDataError(
            f"{key}: upstream columns {list(frame.columns)} do not match the manifest "
            f"{expected_names}"
        )
    for column in entry["schema"]:
        name = column["name"]
        dtype = column["type"]
        if dtype == "int32":
            frame[name] = frame[name].astype("int32")
        elif dtype == "int64":
            frame[name] = frame[name].astype("int64")
        elif dtype == "double":
            frame[name] = frame[name].astype("float64")
        elif dtype == "string":
            frame[name] = frame[name].astype("string")
        else:
            raise EFDataError(f"{key}: unsupported manifest type {dtype!r} for {name!r}")

    arrow_schema = pa.schema(
        [
            pa.field(column["name"], pa.type_for_alias(column["type"]), nullable=column["nullable"])
            for column in entry["schema"]
        ]
    )
    table = pa.Table.from_pandas(frame, schema=arrow_schema, preserve_index=False)
    if table.num_rows != entry["rows"]:
        raise EFDataError(
            f"{key}: converted row count {table.num_rows} does not match the manifest "
            f"{entry['rows']}"
        )
    actual = _content_sha256(table)
    expected = entry.get("content_sha256")
    if expected and actual != expected:
        raise _hash_error(key, str(expected), actual, "the converted .rda values")
    return table


def _read_rda(entry: dict[str, Any], key: str, *, fetch: bool) -> pd.DataFrame:
    # Development trees keep the built Parquet files even though the wheel
    # excludes them. Prefer that verified local copy when it is present.
    """Read the R object from an .rda payload and convert it to a DataFrame."""
    if _data_dir().joinpath(entry["file"]).is_file():
        return _verify_and_read(entry, key, fetch=fetch)
    payload = _raw_payload(entry, key, fetch=fetch)
    return _rda_to_table(payload, entry, key).to_pandas()


def _read_payload(entry: dict[str, Any], key: str, *, fetch: bool) -> bytes:
    """Read a packaged table, a verified cache entry, or a pinned remote file."""
    packaged = _data_dir().joinpath(entry["file"])
    if packaged.is_file():
        return packaged.read_bytes()

    cached = _cache_path(entry)
    if cached.is_file():
        payload = cached.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != entry["sha256"]:
            raise _hash_error(key, str(entry["sha256"]), actual, str(cached))
        return payload

    if not fetch:
        raise EFDataError(
            f"{key}: table is not packaged and has no verified cache entry; "
            f"call load_ef_table({key!r}, fetch=True) after setting "
            f"{_DATA_URL_ENV}, or fetch it for licence review ({entry.get('licence')})"
        )
    url = _remote_url(entry)
    if url is None:
        raise EFDataError(
            f"{key}: no remote URL is configured; set {_DATA_URL_ENV} to the "
            f"directory containing {entry['file']} and retry with fetch=True"
        )
    return _download_and_cache(entry, key, url, str(entry["sha256"]), _cache_path(entry))


def _verify_and_read(entry: dict[str, Any], key: str, *, fetch: bool) -> pd.DataFrame:
    """Read a cached Parquet table after verifying its hashes."""
    payload = _read_payload(entry, key, fetch=fetch)

    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != entry["sha256"]:
        raise _hash_error(key, str(entry["sha256"]), actual_sha, "the table payload")

    table = pq.read_table(pa.BufferReader(payload))
    if table.num_rows != entry["rows"]:
        raise EFDataError(
            f"{key}: row count {table.num_rows} does not match the manifest {entry['rows']}"
        )
    expected_names = [column["name"] for column in entry["schema"]]
    if table.column_names != expected_names:
        raise EFDataError(
            f"{key}: columns {table.column_names} do not match the manifest {expected_names}"
        )
    for field, expected in zip(table.schema, entry["schema"], strict=True):
        if str(field.type) != expected["type"]:
            raise EFDataError(
                f"{key}: column {field.name!r} has type {field.type}, "
                f"manifest says {expected['type']}"
            )
    return table.to_pandas()


def list_ef_tables() -> tuple[str, ...]:
    """Return the canonical names of the packaged EF tables, sorted."""
    return tuple(sorted(_read_manifest().get("tables", {})))


def load_ef_table(name: str, *, fetch: bool = False) -> pd.DataFrame:
    """Load one EF table, optionally fetching a verified remote copy.

    Args:
        name: Canonical name (``brazil_cetesb``, ``europe_emep``, ``usa_emfac``,
            ``usa_moves``), the file stem, or the R object name
            (``ef_brazil_cetesb_db``).
        fetch: Download a non-packaged table when it is not in the cache. The
            EMEP table is packaged; CETESB, EMFAC and MOVES are fetch-only and
            are read from the original pinned ``.rda`` URL in the manifest.
            Install the ``data`` extra for conversion. ``False`` never makes a
            network request.

    Returns:
        A deep copy of the validated table. The cached frame is never handed
        out, so mutating the result does not affect later loads.

    Raises:
        EFDataError: If the table is unknown, missing from the package and cache,
            or its hash, row count, columns or column types disagree with the
            manifest. A changed remote payload is rejected with the expected
            and actual hashes; it is never accepted silently.
    """
    if not isinstance(fetch, bool):
        raise TypeError("fetch must be a bool")
    manifest = _read_manifest()
    key, entry = _resolve(manifest, name)
    if key not in _CACHE:
        remote = entry.get("remote")
        if isinstance(remote, dict) and remote.get("format") == "rda":
            _CACHE[key] = _read_rda(entry, key, fetch=fetch)
        else:
            _CACHE[key] = _verify_and_read(entry, key, fetch=fetch)
    return _CACHE[key].copy(deep=True)
