"""Slope classification from a DEM.

Port of `slope_class_europe_emep()`: sample the DEM at each segment's first and
last vertex, divide the height difference by the segment length in metres, and
bucket the result into the seven EEA classes. The sign is from-stop to to-stop,
so uphill is positive.

Sampling is endpoint-only, matching the reference. Heights are read in the
DEM's CRS: endpoints are projected from the table's CRS when the two differ. A
segment outside the raster (or an empty geometry) gets slope 0.0 with a warning
rather than propagating NaN into an emission factor.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString

__all__ = [
    "SLOPE_CLASSES",
    "SlopeWarning",
    "classify_slope",
    "slope_class_europe_emep",
]

SLOPE_CLASSES: tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)

# R's fcase boundaries. The final interval is open above 0.07 and maps to
# -0.06; values exactly at 0.07 stay in the preceding 0.06 class.
_EDGES: tuple[float, ...] = (-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07)
_LABELS: tuple[float, ...] = (-0.06, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, -0.06)


class SlopeWarning(UserWarning):
    """A segment had no height in the DEM and was treated as flat."""


def classify_slope(values: object) -> np.ndarray:
    """Bucket raw slopes into the seven EEA classes (R's `fcase`).

    Values above ``0.07`` use the final ``-0.06`` class; values exactly at
    ``0.07`` use the preceding ``0.06`` class, matching R's boundary order.
    """
    array = np.asarray(values, dtype=float)
    indices = np.digitize(array, _EDGES, right=True)
    labels = np.asarray(_LABELS, dtype=float)
    return labels[indices]


def slope_class_europe_emep(
    table: gpd.GeoDataFrame,
    heightfile: str | Path,
    *,
    keep: bool = False,
) -> gpd.GeoDataFrame:
    """Add ``slope_class`` (optionally the raw heights/slope) to a transport table.

    Args:
        table: A GeoDataFrame with ``dist_km``, a CRS and LineString geometries.
        heightfile: Path to a raster readable by ``rasterio``. Its CRS may differ
            from the table's; endpoints are reprojected to the raster's CRS.
        keep: When True, also keep ``from_height``, ``to_height``, ``dh`` and
            ``slope``; otherwise only ``slope_class`` is added.

    Returns:
        A copy of ``table`` with the added columns, in the input row order.

    Raises:
        TypeError: If ``table`` is not a GeoDataFrame.
        ValueError: For an empty table, a missing CRS, non-LineString
            geometries, a missing ``dist_km`` column, or a raster with no CRS
            while the table has one.
    """
    if not isinstance(table, gpd.GeoDataFrame):
        raise TypeError("table must be a geopandas GeoDataFrame")
    if table.empty:
        raise ValueError("table has no rows")
    if table.crs is None:
        raise ValueError("table must have a coordinate reference system")
    if "dist_km" not in table.columns:
        raise ValueError("table has no 'dist_km' column")
    geometry_types = set(table.geometry.geom_type.dropna())
    if geometry_types != {"LineString"}:
        raise ValueError(
            f"table must contain LineString geometries only, found {sorted(geometry_types)}"
        )

    from_xy: list[tuple[float, float] | None] = []
    to_xy: list[tuple[float, float] | None] = []
    for geometry in table.geometry:
        if not isinstance(geometry, LineString) or geometry.is_empty:
            from_xy.append(None)
            to_xy.append(None)
            continue
        coordinates = list(geometry.coords)
        from_xy.append((float(coordinates[0][0]), float(coordinates[0][1])))
        to_xy.append((float(coordinates[-1][0]), float(coordinates[-1][1])))

    with rasterio.open(heightfile) as dem:
        if dem.crs is None:
            if table.crs is not None:
                raise ValueError("heightfile has no CRS; cannot sample the table")
            transformer = None
        else:
            transformer = (
                Transformer.from_crs(table.crs, dem.crs, always_xy=True)
                if table.crs != dem.crs
                else None
            )

        def sample(point: tuple[float, float] | None) -> float:
            """Sample the DEM at one endpoint, returning NaN outside the raster."""
            if point is None:
                return math.nan
            x, y = transformer.transform(*point) if transformer is not None else point
            value = next(dem.sample([(x, y)]))[0]
            if dem.nodata is not None and (
                value == dem.nodata or (isinstance(value, float) and math.isnan(value))
            ):
                return math.nan
            return float(value)

        heights = [sample(point) for point in from_xy] + [sample(point) for point in to_xy]

    n = len(table)
    from_height = np.asarray(heights[:n], dtype=float)
    to_height = np.asarray(heights[n:], dtype=float)
    dh = to_height - from_height
    no_height = int(np.sum(~np.isfinite(dh)))
    if no_height:
        warnings.warn(
            f"{no_height} segment(s) have no height in {heightfile} "
            "(outside the raster or empty geometry); their slope is set to 0 (flat)",
            SlopeWarning,
            stacklevel=2,
        )
    distance_m = table["dist_km"].to_numpy(dtype=float) * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dh / distance_m
    slope[~np.isfinite(slope)] = 0.0
    slope_class = classify_slope(slope)

    result = table.copy(deep=True)
    if keep:
        result["from_height"] = from_height
        result["to_height"] = to_height
        result["dh"] = dh
        result["slope"] = slope
    result["slope_class"] = slope_class
    return result
