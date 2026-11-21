"""Slope classification tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from gtfs2emis_py.gtfs.slope import (
    SLOPE_CLASSES,
    SlopeWarning,
    classify_slope,
    slope_class_europe_emep,
)


def _ramp_dem(path: Path, *, crs: str = "EPSG:4326") -> Path:
    """11x11 ramp: height 0 at the north edge, 100 m at the south edge."""
    data = np.arange(11, dtype="float32")[:, None] * 10.0
    data = np.repeat(data, 11, axis=1)
    transform = from_origin(0.0, 0.01, 0.001, 0.001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=11,
        width=11,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def _table(geometries: list[LineString], distances: list[float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"dist_km": distances},
        geometry=geometries,
        crs="EPSG:4326",
    )


def test_exact_classes_from_a_known_ramp(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    segment = LineString([(0.0005, 0.0095), (0.0005, 0.0005)])
    table = _table([segment] * 4, [1.0, 4.5, 9.0, 9.0])
    result = slope_class_europe_emep(table, dem)
    # dh = 90 m; 90/1000 = 0.09 (>0.07 maps to R's -0.06 bucket),
    # 90/4500 = 0.02, 90/9000 = 0.01
    assert result["slope_class"].tolist() == [-0.06, 0.02, 0.0, 0.0]
    assert result["slope_class"].isin(SLOPE_CLASSES).all()


def test_reverse_direction_is_negative(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    segment = LineString([(0.0005, 0.0005), (0.0005, 0.0095)])
    table = _table([segment], [4.5])
    result = slope_class_europe_emep(table, dem)
    assert result["slope_class"].tolist() == [-0.02]


def test_bucket_boundaries() -> None:
    values = [
        -0.08,
        -0.07,
        -0.0501,
        -0.05,
        -0.0301,
        -0.03,
        -0.0101,
        -0.01,
        0.0,
        0.01,
        0.0101,
        0.03,
        0.0301,
        0.05,
        0.0501,
        0.07,
        0.0701,
        0.09,
    ]
    expected = [
        -0.06,
        -0.06,
        -0.06,
        -0.06,
        -0.04,
        -0.04,
        -0.02,
        -0.02,
        0.0,
        0.0,
        0.02,
        0.02,
        0.04,
        0.04,
        0.06,
        0.06,
        -0.06,
        -0.06,
    ]
    assert classify_slope(values).tolist() == expected


def test_nodata_is_flat_with_a_warning(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    segment = LineString([(10.0, 10.0), (10.001, 10.001)])
    table = _table([segment], [0.1])
    with pytest.warns(SlopeWarning, match="no height"):
        result = slope_class_europe_emep(table, dem, keep=True)
    assert result["slope"].tolist() == [0.0]
    assert result["slope_class"].tolist() == [0.0]
    assert np.isnan(result["from_height"].iloc[0])


def test_keep_false_drops_raw_columns(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    segment = LineString([(0.0005, 0.0095), (0.0005, 0.0005)])
    table = _table([segment], [4.5])
    result = slope_class_europe_emep(table, dem)
    assert "slope_class" in result.columns
    for column in ("from_height", "to_height", "dh", "slope"):
        assert column not in result.columns


def test_crs_mismatch_is_reprojected(tmp_path: Path) -> None:
    # DEM in Web Mercator covering the transformed segment endpoints.
    path = tmp_path / "mercator.tif"
    data = np.arange(11, dtype="float32")[:, None] * 10.0
    data = np.repeat(data, 11, axis=1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=11,
        width=11,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0.0, 250.0, 22.2, 22.2),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    segment = LineString([(0.0005, 0.0011), (0.0005, 0.0001)])
    table = _table([segment], [1.0])
    result = slope_class_europe_emep(table, path)
    assert result["slope_class"].iloc[0] in SLOPE_CLASSES


def test_invalid_geometry_and_empty_table(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    points = gpd.GeoDataFrame({"dist_km": [1.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="LineString"):
        slope_class_europe_emep(points, dem)
    empty = gpd.GeoDataFrame({"dist_km": []}, geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="no rows"):
        slope_class_europe_emep(empty, dem)


def test_missing_crs_raises(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    table = gpd.GeoDataFrame(
        {"dist_km": [1.0]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )
    with pytest.raises(ValueError, match="coordinate reference system"):
        slope_class_europe_emep(table, dem)


def test_invalid_table_inputs_are_named(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        slope_class_europe_emep(pd.DataFrame({"dist_km": [1.0]}), dem)
    no_distance = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="dist_km"):
        slope_class_europe_emep(no_distance, dem)


def test_empty_geometry_row_is_flat(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif")
    table = gpd.GeoDataFrame({"dist_km": [1.0]}, geometry=[LineString()], crs="EPSG:4326")
    with pytest.warns(SlopeWarning, match="no height"):
        result = slope_class_europe_emep(table, dem)
    assert result["slope_class"].tolist() == [0.0]


def test_dem_without_crs_is_rejected(tmp_path: Path) -> None:
    dem = _ramp_dem(tmp_path / "ramp.tif", crs=None)
    table = _table([LineString([(0, 0), (1, 1)])], [1.0])
    with pytest.raises(ValueError, match="heightfile has no CRS"):
        slope_class_europe_emep(table, dem)


def test_warning_type_is_exported() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert issubclass(SlopeWarning, UserWarning)
