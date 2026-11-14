"""Fallback route without Earth Engine: ``gsop s2 earth --source local`` (issue B21).

The user supplies already-downloaded rasters (a global elevation grid, MODIS and ASTER products from NASA
Earthdata, ...) and a small YAML file ``local_sources.yaml`` in the same folder. This module resamples them onto
the 6 km EPSG:6933 lattice and writes ONE chunk GeoTIFF with the same band names an Earth Engine export has, so the
rest of the pipeline (``build_parquet``, QA, manifest) is identical for both routes. **Nothing is downloaded here.**

``local_sources.yaml``::

    region: [west, south, east, north]        # degrees; required (a global local build is too big to do by accident)
    bands:
      elevation:   {path: relief_grid.tif, band: 1, scale: 1.0, offset: 0.0}   # metres
      land_frac:   {path: land_mask.tif}                                        # 0..1 share of land (required)
      t_mean:      {path: t_mean.tif}                                           # any exported band name can be given
      ...

Terrain from a coarse DEM: ``elevation_mean`` and ``relief`` are built (relief from the maximum and minimum of the
native cells in each tile, so it **underestimates** when the cells are coarse); ``slope_med``, ``slope_p90``,
``rough_a`` and ``rough_b`` are built only if the DEM cell size is at most the 500 m baseline (roughness: at most
its own baseline); otherwise they are **unavailable** (all NaN) and the missing-data gate acts. A band that is not
supplied is unavailable too. The reasons are written to ``local_report.json`` next to the chunk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.earth import gee_export as G
from gsop.earth.build_parquet import FEATURE_COLUMNS  # noqa: F401  (documentation of what the parquet expects)

SOURCES_FILE = "local_sources.yaml"
METRES_PER_DEGREE = 111195.0
DEM_BANDS = ("slope_med", "slope_p90", "rough_a", "rough_b")
# supplied-band name -> chunk band name
PASS_THROUGH = {
    "t_mean": "t_mean", "t_coldest": "t_coldest", "t_low_obs": "t_low_obs", "ti_proxy": "ti_proxy",
    "ti_proxy_fallback": "ti_proxy_fallback", "water_index": "water_index", "dryness": "dryness",
    "dust_index_raw": "dust_index_raw", "silica_index": "silica_index", "gate_water_ice_frac": "gate_water_ice_frac",
    "gate_ndvi_med": "gate_ndvi_med", "gate_built_crop_frac": "gate_built_crop_frac", "land_frac": "land_frac",
}


class LocalSourceError(ValueError):
    pass


@dataclass
class BandSource:
    path: Path
    band: int = 1
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class LocalSources:
    region: list[float]
    bands: dict[str, BandSource]
    folder: Path


def load_sources(folder: Path) -> LocalSources:
    folder = Path(folder)
    spec_path = folder / SOURCES_FILE
    if not spec_path.is_file():
        raise LocalSourceError(f"{SOURCES_FILE} not found in {folder}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    region = raw.get("region")
    if not (isinstance(region, list) and len(region) == 4 and all(isinstance(v, (int, float)) for v in region)):
        raise LocalSourceError(f"{SOURCES_FILE}: region must be [west, south, east, north] in degrees")
    if not isinstance(raw.get("bands"), dict) or not raw["bands"]:
        raise LocalSourceError(f"{SOURCES_FILE}: bands must be a non-empty mapping")
    allowed = {"elevation", *PASS_THROUGH}
    bands: dict[str, BandSource] = {}
    for name, spec in raw["bands"].items():
        if name not in allowed:
            raise LocalSourceError(f"{SOURCES_FILE}: unknown band '{name}' (allowed: {', '.join(sorted(allowed))})")
        if not isinstance(spec, dict) or "path" not in spec:
            raise LocalSourceError(f"{SOURCES_FILE}: band '{name}' needs a path")
        path = folder / spec["path"]
        if not path.is_file():
            raise LocalSourceError(f"{SOURCES_FILE}: band '{name}': file not found: {path}")
        bands[name] = BandSource(path, int(spec.get("band", 1)), float(spec.get("scale", 1.0)), float(spec.get("offset", 0.0)))
    if "land_frac" not in bands:
        raise LocalSourceError(f"{SOURCES_FILE}: a land_frac band is required (it decides which tiles are land)")
    return LocalSources([float(v) for v in region], bands, folder)


def native_cell_m(path: Path, region: list[float]) -> float:
    """Native cell size in metres (the larger of x and y); degrees are converted at the region's mid latitude."""
    with rasterio.open(path) as ds:
        if ds.crs and ds.crs.is_geographic:
            lat = (region[1] + region[3]) / 2
            return max(abs(ds.transform.a) * METRES_PER_DEGREE * math.cos(math.radians(lat)), abs(ds.transform.e) * METRES_PER_DEGREE)
        return max(abs(ds.transform.a), abs(ds.transform.e))


def resample_to_grid(src: BandSource, dst_transform, shape: tuple[int, int], crs: str, method: Resampling) -> np.ndarray:
    """Resample one band onto the destination 6 km grid. Nodata and masked cells become NaN."""
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(src.path) as ds:
        data = ds.read(src.band, masked=True).astype("float32").filled(np.nan) * src.scale + src.offset
        reproject(data, out, src_transform=ds.transform, src_crs=ds.crs, dst_transform=dst_transform, dst_crs=crs,
                  resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def build_local_chunk(folder: Path, cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Write ``out_dir/local_chunk.tif`` (+ ``local_report.json``) from the user's rasters. Returns the report."""
    sources = load_sources(folder)
    (chunk,) = G.plan_chunks(cfg, sources.region)
    ops = cfg.features.operators
    shape = (chunk.height_px, chunk.width_px)
    transform = from_origin(chunk.x0, chunk.y1, ops.tile_m, ops.tile_m)
    all_bands, _ = G.planned_bands(cfg)
    bands: dict[str, np.ndarray] = {}
    reasons: dict[str, str] = {}

    for out_name, key in PASS_THROUGH.items():
        if key in sources.bands:
            bands[out_name] = resample_to_grid(sources.bands[key], transform, shape, ops.grid_crs, Resampling.average)

    if "elevation" in sources.bands:
        dem = sources.bands["elevation"]
        cell = native_cell_m(dem.path, sources.region)
        bands["elevation_mean"] = resample_to_grid(dem, transform, shape, ops.grid_crs, Resampling.average)
        hi = resample_to_grid(dem, transform, shape, ops.grid_crs, Resampling.max)
        lo = resample_to_grid(dem, transform, shape, ops.grid_crs, Resampling.min)
        bands["relief"] = hi - lo
        reasons["relief"] = (f"built from the max and min of native cells of {cell:.0f} m: underestimates relief when cells are coarser than "
                             f"the {ops.baseline_m:.0f} m baseline") if cell > ops.baseline_m else ""
        for name in ("slope_med", "slope_p90"):
            if cell > ops.baseline_m:
                reasons[name] = f"unavailable: DEM cell {cell:.0f} m is coarser than the {ops.baseline_m:.0f} m baseline"
        for name in ("rough_a", "rough_b"):
            baseline = G.roughness_baseline_m(cfg, name)
            if cell > baseline:
                reasons[name] = f"unavailable: DEM cell {cell:.0f} m is coarser than the {baseline:.0f} m baseline"
        if cell <= ops.baseline_m:
            for name in DEM_BANDS:  # buildable in principle from this DEM, but no tile operator is wired into the local route yet
                reasons[name] = "unavailable: local route has no tile operator wired in yet (use operators.py on windows; see SPEC)"
    else:
        reasons["relief"] = reasons["elevation_mean"] = "unavailable: no elevation raster supplied"
        for name in DEM_BANDS:
            reasons[name] = "unavailable: no elevation raster supplied"

    unavailable: dict[str, str] = {}
    for name in all_bands:
        if name not in bands or reasons.get(name, "").startswith("unavailable"):
            bands[name] = np.full(shape, np.nan, dtype="float32")
            unavailable[name] = reasons.get(name) or "unavailable: not supplied"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "local_chunk.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=shape[0], width=shape[1], count=len(all_bands), dtype="float32",
                       crs=ops.grid_crs, transform=transform, nodata=-9999.0) as ds:
        for i, name in enumerate(all_bands, start=1):
            ds.write(np.where(np.isnan(bands[name]), -9999.0, bands[name]).astype("float32"), i)
            ds.set_band_description(i, name)
    report = {
        "route": "local (coarser than Earth Engine: terrain features are coarse or unavailable)",
        "region": sources.region,
        "bands": all_bands,
        "unavailable": unavailable,
        "notes": {k: v for k, v in reasons.items() if v and not v.startswith("unavailable")},
        "sources": {k: str(v.path) for k, v in sources.bands.items()},
    }
    atomic_write_text(out_dir / "local_report.json", json.dumps(report, indent=2) + "\n")
    return report
