"""s2 targets --body mars: Mars target profiles. Terrain at Jezero (issue B28); THEMIS/TES (B29), SWIM water_index and Arcadia
(B30), temperature, air_density, insolation (B31) are added by those issues.

Terrain source: the USGS Astrogeology MOLA DEM mosaic ``Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`` (int16, metres above the
areoid, equirectangular on the IAU 2000 sphere of radius 3,396,190 m, 463.09 m per pixel, nodata -32768). Only a small window is
read over HTTP (``rasterio``); the 2 GB file is never downloaded.

The file's cells are NOT square in metres: an equirectangular pixel is 463.09 m north-south but only 463.09 * cos(latitude)
east-west (439 m at Jezero). Applying the Horn slope to it as if it were square would be wrong, so the window is **warped to a
local Lambert azimuthal equal-area grid centred on the target at exactly 500 m** (bilinear interpolation at the 500 m cell centres,
with exact pyproj coordinates; area-averaging uneven 463-to-500 m weights and GDAL's approximate warp transformer both add
slope noise of several percent, bilinear with exact coordinates is exact on a plane), the same baseline as
Earth and Moon. Then ``gsop.operators`` runs unchanged. What this cannot fix: MOLA itself is an interpolated
463 m product (track gaps of 1 to 2 km are filled), so Mars slope and roughness have less fine-scale information than the
5 m LOLA and 30 m Earth data; and ``rough_a`` (100 m baseline) cannot be met at 500 m cells: its effective baseline is 500 m.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from gsop.atomic import atomic_write_bytes, atomic_write_text
from gsop.config import Config
from gsop.targets.moon import TERRAIN_COLUMNS, terrain_profile

MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
MARS_RADIUS_M = 3_396_190.0
TARGET_CELL_M = 500.0
SOURCE_TEXT = "MOLA DEM 463 m (MOLA Team; USGS Astrogeology Mars_MGS_MOLA_DEM_mosaic_global_463m.tif)"
EVIDENCE_GRADE = "A"  # measured; but gaps between tracks of 1-2 km are interpolated


def local_crs(lat: float, lon_east: float) -> CRS:
    return CRS.from_proj4(f"+proj=laea +lat_0={lat} +lon_0={lon_east} +x_0=0 +y_0=0 +a={MARS_RADIUS_M} +b={MARS_RADIUS_M} +units=m +no_defs")


def _open_path(source: str) -> str:
    return f"/vsicurl/{source}" if str(source).startswith("http") else str(source)


def read_window(source: str, lat: float, lon_east: float, window_km: float, cache_dir: Optional[Path] = None) -> dict:
    """15 km (``window_km``) DEM window centred on the target on a local 500 m equal-area grid. Windowed read only."""
    key = hashlib.sha256(f"{source}|{lat}|{lon_east}|{window_km}|{TARGET_CELL_M}".encode()).hexdigest()[:16]
    if cache_dir is not None:
        npy, meta = Path(cache_dir) / f"mola_{key}.npy", Path(cache_dir) / f"mola_{key}.json"
        if npy.is_file() and meta.is_file():
            info = json.loads(meta.read_text(encoding="utf-8"))
            info["dem"] = np.load(npy)
            return info
    n = int(round(window_km * 1000.0 / TARGET_CELL_M))
    dst_crs = local_crs(lat, lon_east)
    # centres of the destination cells in the local equal-area grid (row 0 at the north)
    half = window_km * 500.0
    cx = -half + (np.arange(n) + 0.5) * TARGET_CELL_M
    cy = half - (np.arange(n) + 0.5) * TARGET_CELL_M
    gx, gy = np.meshgrid(cx, cy)
    with rasterio.Env(GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="3", CPL_VSIL_CURL_CHUNK_SIZE="4194304", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(_open_path(source)) as ds:
            sphere = CRS.from_proj4(f"+proj=longlat +a={MARS_RADIUS_M} +b={MARS_RADIUS_M} +no_defs")
            to_src = Transformer.from_crs(sphere, ds.crs, always_xy=True)
            x, y = to_src.transform(((lon_east + 180.0) % 360.0) - 180.0, lat)
            margin = int(math.ceil((window_km * 1000.0) / ds.res[0])) + 6  # source pixels each side of the target
            col, row = int((x - ds.bounds.left) / ds.res[0]), int((ds.bounds.top - y) / ds.res[1])
            win = Window(max(col - margin, 0), max(row - margin, 0), 2 * margin, 2 * margin)
            src = ds.read(1, window=win, masked=True).astype("float64").filled(np.nan)
            wt = ds.window_transform(win)
            # EXACT coordinates (pyproj), not GDAL's warp: GDAL's approximate transformer allows 1/8 pixel of position error,
            # which on a 6 degree plane is 6 m of elevation and shows up as +-5% noise in slope.
            lon, latg = Transformer.from_crs(dst_crs, sphere, always_xy=True).transform(gx, gy)
            sx, sy = to_src.transform(lon, latg)
            cols = (sx - wt.c) / wt.a - 0.5
            rows = (sy - wt.f) / wt.e - 0.5
            native = float(ds.res[0])
    # bilinear interpolation of the source at the 500 m cell centres; any NaN neighbour makes the result NaN
    dem = map_coordinates(src, [rows, cols], order=1, mode="constant", cval=np.nan, prefilter=False).astype("float32")
    info = {
        "cell_m": TARGET_CELL_M, "native_cell_m": native, "x0": -window_km * 500.0, "y_top": window_km * 500.0,
        "crs": dst_crs.to_wkt(), "source": source, "shape": [n, n],
        "sha256": hashlib.sha256(np.ascontiguousarray(dem).tobytes()).hexdigest(),
    }
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        import io

        buf = io.BytesIO()
        np.save(buf, dem)
        atomic_write_bytes(npy, buf.getvalue())
        atomic_write_text(meta, json.dumps(info, indent=2) + "\n")
    info["dem"] = dem
    return info


def mars_terrain_rows(cfg: Config, source: str = MOLA_URL, cache_dir: Optional[Path] = None) -> list[dict]:
    """One row per (target, terrain column) for the scored Mars targets (Arcadia has no coordinates and is skipped)."""
    rows = []
    ts = cfg.target_settings
    for name, target in cfg.targets("mars").items():
        if target.lat is None:  # decision D10: Arcadia is not guessed
            continue
        info = read_window(source, target.lat, target.lon_e, ts.target_window_km, cache_dir)
        prof = terrain_profile(info["dem"], info["cell_m"], cfg)
        window = (f"{ts.target_window_km:g} km window centred on the target, {ts.tile_stride_km:g} km stride, "
                  f"{cfg.features.operators.tile_m / 1000:g} km tiles; MOLA {info['native_cell_m']:.0f} m cells interpolated to a local {info['cell_m']:g} m equal-area grid")
        for col in TERRAIN_COLUMNS:
            p = prof[col]
            rows.append({
                "target": name, "column": col, "value": p["value"], "spread": p["spread"], "source": SOURCE_TEXT,
                "evidence_grade": EVIDENCE_GRADE, "window": window, "unavailable": not np.isfinite(p["value"]),
                "note": (f"{p['n_valid']} of {p['n_tiles']} tiles valid; effective baselines {prof['effective_baseline_m']}"
                         + ("; rough_a baseline is 500 m, not 100 m (MOLA cannot resolve it)" if col == "rough_a" else "")
                         + f"; window sha256 {info['sha256'][:16]}"),
            })
    return rows
