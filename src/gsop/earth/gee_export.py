"""Earth Engine side of the Earth feature build (``gsop s2 earth --source gee``).

STATUS: written against the Earth Engine Python API but **never executed against live Earth Engine**: no
login existed on the build machine when this was written (issue B01). Everything that does not touch
Earth Engine (chunk planning, task ledger, retries, band lists) is unit tested; the Earth Engine
expressions are not. ``gsop check-operators`` (issue B24) and the small-region export (issue B17) are the
first real tests and must pass before any global export.

Every function takes the initialised ``ee`` module as its first argument (``gee_auth.ensure_ee()``), so
this module imports without Earth Engine installed or logged in.

The terrain operator is defined once in ``docs/SPEC.md`` (section Operators) and implemented twice: in
numpy (``operators.py``) and here. Settings come from ``config/features.yaml`` (``operators`` block).
"""

from __future__ import annotations

import math
from typing import Any

from gsop.config import Config

NASADEM_ID = "NASA/NASADEM_HGT/001"
NASADEM_BAND = "elevation"
COPERNICUS_ID = "COPERNICUS/DEM/GLO30_2024_1"
COPERNICUS_BAND = "DEM"

# Horn (1981) weights. dz/dx = sum(KX * z) / (8 * dx); dz/dy = sum(KY * z) / (8 * dy).
HORN_KX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
HORN_KY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def lonlat_box_around(lat: float, lon: float, size_km: float) -> list[float]:
    """``[west, south, east, north]`` of a box about ``size_km`` wide and tall centred on a point.

    Plain spherical approximation (111.195 km per degree of latitude); good enough to choose an export
    region. Not used for scoring.
    """
    half_lat = (size_km / 2) / 111.195
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    half_lon = half_lat / cos_lat
    return [lon - half_lon, max(lat - half_lat, -90.0), lon + half_lon, min(lat + half_lat, 90.0)]


def dem_mosaic(ee: Any, cfg: Config) -> Any:
    """One elevation image: NASADEM inside the configured latitude range, Copernicus GLO-30 outside it.

    Seam rule (documented in SPEC): the switch is a hard cut at the configured latitudes; there is no
    blending. Issue B07 measures whether that creates a step.
    """
    south, north = cfg.features.operators.nasadem_lat_range
    nasa = ee.Image(NASADEM_ID).select(NASADEM_BAND).rename("elevation")
    cop_col = ee.ImageCollection(COPERNICUS_ID).select(COPERNICUS_BAND)
    cop_proj = cop_col.first().select(0).projection()
    cop = cop_col.mosaic().setDefaultProjection(cop_proj).rename("elevation")
    lat = ee.Image.pixelLonLat().select("latitude")
    in_nasa = lat.gte(south).And(lat.lte(north))
    merged = ee.ImageCollection([cop.updateMask(in_nasa.Not()), nasa.updateMask(in_nasa)]).mosaic()
    return merged.setDefaultProjection(nasa.projection()).rename("elevation")


def resample_mean(ee: Any, image: Any, cfg: Config, scale_m: float, max_pixels: int = 4096) -> Any:
    """Block-mean ``image`` to ``scale_m`` metres on the grid CRS (``reduceResolution`` mean)."""
    return (
        image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels)
        .reproject(crs=cfg.features.operators.grid_crs, scale=scale_m)
    )


def horn_slope_deg(ee: Any, dem: Any, cell_m: float) -> Any:
    """Horn slope in degrees on a ``cell_m`` grid. Cells with any missing 3x3 neighbour are masked."""
    kx = ee.Kernel.fixed(3, 3, HORN_KX, 1, 1, False)
    ky = ee.Kernel.fixed(3, 3, HORN_KY, 1, 1, False)
    valid = dem.mask().reduceNeighborhood(reducer=ee.Reducer.min(), kernel=ee.Kernel.square(1, "pixels"))
    filled = dem.unmask(0)
    gx = filled.convolve(kx).divide(8.0 * cell_m)
    gy = filled.convolve(ky).divide(8.0 * cell_m)
    slope = gx.pow(2).add(gy.pow(2)).sqrt().atan().multiply(180.0 / math.pi)
    return slope.updateMask(valid).rename("slope")


def tile_cells(cfg: Config, cell_m: float) -> int:
    ops = cfg.features.operators
    return int(round(ops.tile_m / cell_m)) ** 2


def slope_bands(ee: Any, cfg: Config) -> Any:
    """Bands ``slope_med`` and ``slope_p90`` (degrees) per 6 km tile, from slope at the 500 m baseline.

    DEM -> block mean to 500 m -> Horn slope -> median and 90th percentile over the tile's 12 x 12 cells.
    A tile with more than ``max_nan_fraction`` missing cells is masked (never a partial answer).
    """
    ops = cfg.features.operators
    dem500 = resample_mean(ee, dem_mosaic(ee, cfg), cfg, ops.baseline_m)
    slope = horn_slope_deg(ee, dem500, ops.baseline_m).reproject(crs=ops.grid_crs, scale=ops.baseline_m)
    n = tile_cells(cfg, ops.baseline_m)
    pct = slope.reduceResolution(reducer=ee.Reducer.percentile([50, 90]), maxPixels=n + 16)
    count = slope.reduceResolution(reducer=ee.Reducer.count(), maxPixels=n + 16)
    enough = count.gte(math.ceil((1.0 - ops.max_nan_fraction) * n))
    return pct.rename(["slope_med", "slope_p90"]).updateMask(enough)


def _enough_cells(ee: Any, image: Any, cfg: Config, cell_m: float, max_pixels: int) -> Any:
    """Mask that is 1 where a tile has enough valid cells (more than ``max_nan_fraction`` missing: 0)."""
    n = tile_cells(cfg, cell_m)
    count = image.reduceResolution(reducer=ee.Reducer.count(), maxPixels=max_pixels)
    return count.gte(math.ceil((1.0 - cfg.features.operators.max_nan_fraction) * n))


def relief_band(ee: Any, cfg: Config) -> Any:
    """Band ``relief`` (m): max minus min of the 500 m cells inside the 6 km tile."""
    ops = cfg.features.operators
    dem500 = resample_mean(ee, dem_mosaic(ee, cfg), cfg, ops.baseline_m)
    n = tile_cells(cfg, ops.baseline_m)
    hi = dem500.reduceResolution(reducer=ee.Reducer.max(), maxPixels=n + 16)
    lo = dem500.reduceResolution(reducer=ee.Reducer.min(), maxPixels=n + 16)
    return hi.subtract(lo).rename("relief").updateMask(_enough_cells(ee, dem500, cfg, ops.baseline_m, n + 16))


def roughness_baseline_m(cfg: Config, column: str) -> float:
    """Baseline in metres of ``rough_a`` / ``rough_b``: lives in config/features.yaml, never in code."""
    return float(cfg.features.columns[column].extra["baseline_m"])


def rough_band(ee: Any, cfg: Config, column: str) -> Any:
    """Band ``rough_a`` or ``rough_b`` (m): RMS height after plane removal over the 6 km tile.

    The DEM is block-averaged to the column's baseline B (100 m or 1 km), then inside each tile a plane
    ``z = a + b*x + c*y`` is fitted and removed and the RMS of the residual is taken. On a full square
    grid with tile-centred coordinates ``x`` and ``y`` are orthogonal to each other and to the constant, so
    the least-squares plane has a closed form and no regression is needed::

        mean_sq_resid = E[z^2] - E[z]^2 - E[x z]^2 / E[x^2] - E[y z]^2 / E[y^2],   E[x^2] = (n^2 - 1) / 12

    where ``n`` is the number of cells along a tile side and every ``E`` is a tile mean (computed with
    ``reduceResolution(mean)``). This is exact for a tile with no missing cells; for a tile with some
    missing cells it is an approximation (the numpy operator fits the plane by least squares on the valid
    cells). Tiles with too many missing cells are masked.
    """
    ops = cfg.features.operators
    baseline = roughness_baseline_m(cfg, column)
    n = int(round(ops.tile_m / baseline))
    z = resample_mean(ee, dem_mosaic(ee, cfg), cfg, baseline).select(0).rename("z")
    px = ee.Image.pixelCoordinates(z.projection())
    centre = (n - 1) / 2.0

    def local(band: str) -> Any:  # index of the cell inside its tile, centred; floor-mod so it is never negative
        v = px.select(band)
        return v.subtract(v.divide(n).floor().multiply(n)).subtract(centre)

    stack = ee.Image.cat([z, z.pow(2), local("x").multiply(z), local("y").multiply(z)]).rename(["z", "z2", "xz", "yz"])
    mean = stack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=n * n + 16)
    ex2 = (n * n - 1) / 12.0
    resid = (
        mean.select("z2").subtract(mean.select("z").pow(2))
        .subtract(mean.select("xz").pow(2).divide(ex2))
        .subtract(mean.select("yz").pow(2).divide(ex2))
    )
    rms = resid.max(0).sqrt().rename(column)
    return rms.updateMask(_enough_cells(ee, z, cfg, baseline, n * n + 16))
