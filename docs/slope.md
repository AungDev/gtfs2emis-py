# Slope classification

`slope_class_europe_emep()` samples a DEM at each segment's first and last
vertex, divides the height difference by the segment length in metres, and adds
a `slope_class` column for the EMEP lookup. With `keep=True` it also retains
`from_height`, `to_height`, `dh` and the raw `slope`.

## Rules

- The DEM may have a different CRS; endpoints are reprojected to the raster CRS.
- A table without a CRS, a DEM without a CRS while the table has one, an empty
  table, non-LineString geometries and a missing `dist_km` column are rejected.
- A segment outside the raster, a nodata height or an empty geometry is treated
  as flat (`slope_class = 0.0`) and emits `SlopeWarning`.
- Raw slopes are bucketed with R's `fcase` boundaries. Values above `0.07` map
  to the final `-0.06` class; exactly `0.07` stays in the preceding `0.06` class.

```python
from gtfs2emis_py.gtfs.slope import slope_class_europe_emep

classified = slope_class_europe_emep(table, "dem.tif", keep=True)
```
