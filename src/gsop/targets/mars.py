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


def read_window(source: str, lat: float, lon_east: float, window_km: float, cache_dir: Optional[Path] = None,
                cell_m: float = TARGET_CELL_M) -> dict:
    """``window_km`` square window centred on the target, resampled to a local equal-area grid of ``cell_m`` metres (default 500 m, the
    terrain baseline; THEMIS uses its native 100 m). Windowed read only."""
    key = hashlib.sha256(f"{source}|{lat}|{lon_east}|{window_km}|{cell_m}".encode()).hexdigest()[:16]
    if cache_dir is not None:
        npy, meta = Path(cache_dir) / f"mola_{key}.npy", Path(cache_dir) / f"mola_{key}.json"
        if npy.is_file() and meta.is_file():
            info = json.loads(meta.read_text(encoding="utf-8"))
            info["dem"] = np.load(npy)
            return info
    n = int(round(window_km * 1000.0 / cell_m))
    dst_crs = local_crs(lat, lon_east)
    # centres of the destination cells in the local equal-area grid (row 0 at the north)
    half = window_km * 500.0
    cx = -half + (np.arange(n) + 0.5) * cell_m
    cy = half - (np.arange(n) + 0.5) * cell_m
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
        "cell_m": cell_m, "native_cell_m": native, "x0": -window_km * 500.0, "y_top": window_km * 500.0,
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
        # target elevation (metres above the areoid): the input of the Mars target check (decision D6), not a scored column
        from gsop.targets.moon import _summarise

        n, step = int(round(cfg.features.operators.tile_m / info["cell_m"])), int(round(ts.tile_stride_km * 1000.0 / info["cell_m"]))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a tile with no valid cell gives NaN (and is skipped by the summary)
            elev = np.array([np.nanmean(info["dem"][r : r + n, c : c + n]) for r in range(0, info["dem"].shape[0] - n + 1, step)
                             for c in range(0, info["dem"].shape[1] - n + 1, step)], dtype="float64")
        p = _summarise(elev, ts.spread_iqr_divisor)
        rows.append({
            "target": name, "column": "elevation_mean", "value": p["value"], "spread": p["spread"], "source": SOURCE_TEXT,
            "evidence_grade": EVIDENCE_GRADE, "window": window, "unavailable": not np.isfinite(p["value"]),
            "note": "metres above the areoid (MOLA); input of the Mars target check (D6), not a scored column",
        })
    return rows


# ===========================================================================================================
# THEMIS thermal inertia, TES, and what cannot be built (issue B29)
# ===========================================================================================================
THEMIS_BASE = ("https://asc-astropedia.s3.us-west-2.amazonaws.com/Mars/Odyssey/THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/"
               "THEMIS_TI_Mosaic_Quant_{band}_100mpp.cub")
THEMIS_CELL_M = 100.0
TI_COLUMN = "thermal_inertia_themis"  # raw THEMIS thermal inertia in J m-2 K-1 s-1/2; NOT a scored column (see SPEC)


def themis_tile_name(lat: float, lon_east: float) -> str:
    """Name of the 60 x 30 degree THEMIS quantitative mosaic tile, for example ``00N060E`` (0-30 N, 60-120 E).

    Only the northern naming pattern (``00N``, ``30N``) was seen on the catalog pages; other latitude bands raise.
    """
    if not 0.0 <= lat < 60.0:
        raise ValueError("only the tiles at 0-30 N and 30-60 N were verified; extend after checking the catalog names")
    return f"{int(lat // 30) * 30:02d}N{int((lon_east % 360.0) // 60) * 60:03d}E"


def themis_url(lat: float, lon_east: float) -> str:
    return THEMIS_BASE.format(band=themis_tile_name(lat, lon_east))


def _tile_values(grid: np.ndarray, cell_m: float, tile_m: float, stride_m: float) -> np.ndarray:
    n, step = int(round(tile_m / cell_m)), int(round(stride_m / cell_m))
    meds = []
    for r in range(0, grid.shape[0] - n + 1, step):
        for c in range(0, grid.shape[1] - n + 1, step):
            block = grid[r : r + n, c : c + n]
            meds.append(float(np.nanmedian(block)) if np.isfinite(block).mean() >= 0.8 else np.nan)
    return np.array(meds)


def mars_thermal_rows(cfg: Config, themis_source=themis_url, cache_dir: Optional[Path] = None) -> list[dict]:
    """THEMIS thermal inertia at Jezero, and explicit ``unavailable`` rows for ``ti_proxy``, ``dust_index`` and ``silica_index``."""
    from gsop.targets.moon import _summarise

    ts = cfg.target_settings
    tile = cfg.features.operators.tile_m
    rows: list[dict] = []
    for name, target in cfg.targets("mars").items():
        if target.lat is None:
            continue
        window = f"{ts.target_window_km:g} km window, {ts.tile_stride_km:g} km stride, {tile / 1000:g} km tiles"
        info = read_window(themis_source(target.lat, target.lon_e), target.lat, target.lon_e, ts.target_window_km, cache_dir, cell_m=THEMIS_CELL_M)
        vals = _tile_values(info["dem"].astype("float64"), THEMIS_CELL_M, tile, ts.tile_stride_km * 1000.0)
        p = _summarise(vals, ts.spread_iqr_divisor)
        rows.append({
            "target": name, "column": TI_COLUMN, "value": p["value"], "spread": p["spread"],
            "source": "THEMIS quantitative thermal inertia mosaic (Fergason et al. 2006; USGS Astropedia THEMIS_TI_Mosaic_Quant_" + themis_tile_name(target.lat, target.lon_e) + "_100mpp.cub)",
            "evidence_grade": "A", "window": window + f"; {info['shape'][1]}x{info['shape'][0]} px of 100 m", "unavailable": not np.isfinite(p["value"]),
            "note": "raw thermal inertia J m-2 K-1 s-1/2 (stated absolute accuracy about 20%); used for the D6 Mars target check, NOT a scored column (see below)",
        })
        for col, why in (
            ("ti_proxy", "unavailable: Earth ti_proxy is a day-night temperature SWING in K (high inertia = small swing); THEMIS gives thermal inertia itself (high inertia = large value), "
                         "so the direction and units differ, and no thermal model is available to convert one into the other"),
            ("dust_index", "unavailable: no TES dust cover index product was found (only the 8 pixel/degree TES bolometric albedo map, which is albedo, not a dust index); "
                           "Earth dust_index is MERRA-2 atmospheric dust optical thickness scaled with Earth percentiles, so a Mars value would not be comparable"),
            ("silica_index", "unavailable: no TES silica or surface-type map was found on the catalog pages searched; the Earth silica_index is the ASTER Quartz Index "
                             "and its Mars equivalent has not been defined or located"),
        ):
            rows.append({"target": name, "column": col, "value": float("nan"), "spread": float("nan"), "source": "TES / THEMIS (not built)",
                         "evidence_grade": "C", "window": window, "unavailable": True, "note": why})
    return rows


# ===========================================================================================================
# Mars water_index from SWIM (issue B30, decisions D7 and D10)
# ===========================================================================================================
SWIM_CI_URL = "https://swim.psi.edu/output/NorthernHemisphere/Northern/Composite/Northern_Ci.tif"
SWIM_ARCADIA_FOOTPRINT_URL = "https://swim.psi.edu/output/PilotStudy/Arcadia/Composite/Ci.tif"
SWIM_CELL_M = 3000.0


def consistency_to_water_index(c) -> np.ndarray:
    """SWIM combined ice consistency C in [-1, +1] (+1 consistent with ice, -1 inconsistent) -> water_index in [0, 1]: (C + 1) / 2."""
    return np.clip((np.asarray(c, dtype="float64") + 1.0) / 2.0, 0.0, 1.0)


def swim_water_index_rows(cfg: Config, source: str = SWIM_CI_URL, cache_dir: Optional[Path] = None) -> list[dict]:
    """Mars ``water_index`` = ONE composite, the SWIM combined ice consistency rescaled to 0..1 (decision D7).

    The SWIM combined map is itself the "SWIM equation" combination of the neutron (water-equivalent hydrogen, including its depth
    to a lower layer), thermal (TES and THEMIS thermal inertia), radar surface power, geomorphology and radar dielectric maps. Ice
    depth, ice consistency and thermal inertia are therefore **already inside this one number and are not added again**: counting
    them separately is exactly what D7 forbids. Value = median over the window's 6 km tiles of ``(C + 1) / 2``, spread = IQR / 1.349.
    The product has a 3 km pixel, so a 6 km tile holds only 2 x 2 pixels.
    """
    from gsop.targets.moon import _summarise

    ts = cfg.target_settings
    tile = cfg.features.operators.tile_m
    rows: list[dict] = []
    for name, target in cfg.targets("mars").items():
        if target.lat is None:
            continue
        info = read_window(source, target.lat, target.lon_e, ts.target_window_km, cache_dir, cell_m=SWIM_CELL_M)
        wi = consistency_to_water_index(info["dem"])
        vals = _tile_values(wi, SWIM_CELL_M, tile, ts.tile_stride_km * 1000.0)
        p = _summarise(vals, ts.spread_iqr_divisor)
        rows.append({
            "target": name, "column": "water_index", "value": p["value"], "spread": p["spread"],
            "source": "SWIM 1.0 combined ice consistency, northern hemisphere (Morgan et al. 2021; swim.psi.edu Northern_Ci.tif), rescaled (C + 1) / 2",
            "evidence_grade": "B",
            "window": f"{ts.target_window_km:g} km window, {ts.tile_stride_km:g} km stride, {tile / 1000:g} km tiles; SWIM pixel 3 km ({info['shape'][1]}x{info['shape'][0]} sampled px)",
            "unavailable": not np.isfinite(p["value"]),
            "note": "ONE composite (D7): ice depth, consistency and thermal inertia are all inside the SWIM combined consistency and not counted again; cross-instrument proxy",
        })
    return rows


def swim_arcadia_footprint(source: str = SWIM_ARCADIA_FOOTPRINT_URL) -> dict:
    """Footprint of the SWIM pilot-study raster for Arcadia, as INFORMATION ONLY (it is never written to targets.yaml).

    It is the extent of a study swath (about 20 degrees of longitude by 60 degrees of latitude), not the position of a site, and the
    SWIM maps contain no site centre; the midpoint would be an arbitrary number presented as data, which decision D10 forbids.
    """
    with rasterio.Env(GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="3", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(_open_path(source)) as ds:
            b = ds.bounds
            r = MARS_RADIUS_M
            return {
                "lon_west": math.degrees(b.left / r), "lon_east": math.degrees(b.right / r),
                "lat_south": math.degrees(b.bottom / r), "lat_north": math.degrees(b.top / r),
                "pixel_m": float(ds.res[0]), "source": source,
                "is_a_site_centre": False,
            }


# ===========================================================================================================
# Temperature (unavailable), air_density and insolation (issue B31)
# ===========================================================================================================
def _local_tile_centres(window_km: float, tile_m: float, stride_m: float) -> np.ndarray:
    """(x, y) centres in the local grid (metres from the window centre) of the tiles of a ``window_km`` square."""
    n = int(round((window_km * 1000.0 - tile_m) / stride_m)) + 1
    start = -window_km * 500.0 + tile_m / 2.0
    return np.array([(start + i * stride_m, start + j * stride_m) for j in range(n) for i in range(n)])


def mars_climate_rows(cfg: Config, mola_source: str = MOLA_URL, cache_dir: Optional[Path] = None) -> list[dict]:
    """``air_density``, ``insol_annual``, ``sun_max_elev`` (built) and ``t_mean``, ``t_coldest`` (unavailable) at the Mars targets.

    * ``air_density``: the exponential law of ``gsop.features_misc`` (constants from ``features.yaml``, Mars Fact Sheet) at the mean MOLA
      elevation (above the areoid) of each tile; median over tiles. Same function as Earth, no constants in code.
    * ``insol_annual`` and ``sun_max_elev``: ``gsop.illumination`` with the Mars obliquity and irradiance from ``features.yaml``, at the
      latitude of each tile centre (flat ground, no terrain shadowing).
    * ``t_mean``, ``t_coldest``: unavailable, see the note.
    """
    from gsop.features_misc import air_density
    from gsop.illumination import compute_illumination
    from gsop.targets.moon import _summarise

    ts = cfg.target_settings
    tile = cfg.features.operators.tile_m
    ad = cfg.features.air_density
    rows: list[dict] = []
    for name, t in cfg.targets("mars").items():
        if t.lat is None:
            continue
        window = f"{ts.target_window_km:g} km window, {ts.tile_stride_km:g} km stride, {tile / 1000:g} km tiles"
        info = read_window(mola_source, t.lat, t.lon_e, ts.target_window_km, cache_dir)
        dem = info["dem"].astype("float64")
        n, step = int(round(tile / info["cell_m"])), int(round(ts.tile_stride_km * 1000.0 / info["cell_m"]))
        elev = np.array([np.nanmean(dem[r : r + n, c : c + n]) for r in range(0, dem.shape[0] - n + 1, step) for c in range(0, dem.shape[1] - n + 1, step)])
        ok = np.isfinite(elev)
        rho = np.full(elev.shape, np.nan)
        rho[ok] = air_density(elev[ok], ad["mars"], ad["universal_gas_constant_j_mol_k"])
        p = _summarise(rho, ts.spread_iqr_divisor)
        rows.append({
            "target": name, "column": "air_density", "value": p["value"], "spread": p["spread"],
            "source": "gsop.features_misc exponential law (NASA Mars Fact Sheet constants in features.yaml) at MOLA elevation above the areoid",
            "evidence_grade": "B", "window": window, "unavailable": not np.isfinite(p["value"]),
            "note": "kg/m3; the pressure datum (stated 'at mean radius' vs MOLA areoid) is an open TODO in features.yaml",
        })
        # insolation at the tile-centre latitudes
        centres = _local_tile_centres(ts.target_window_km, tile, ts.tile_stride_km * 1000.0)
        from pyproj import Transformer

        sphere = CRS.from_proj4(f"+proj=longlat +a={MARS_RADIUS_M} +b={MARS_RADIUS_M} +no_defs")
        _, lat = Transformer.from_crs(local_crs(t.lat, t.lon_e), sphere, always_xy=True).transform(centres[:, 0], centres[:, 1])
        ill = compute_illumination(np.asarray(lat), cfg.features.illumination.bodies["mars"], cfg.features.illumination)
        for col in ("insol_annual", "sun_max_elev"):
            q = _summarise(ill[col], ts.spread_iqr_divisor)
            rows.append({
                "target": name, "column": col, "value": q["value"], "spread": q["spread"],
                "source": "computed: gsop.illumination (analytic, flat ground) with the Mars obliquity and irradiance from features.yaml",
                "evidence_grade": "B", "window": window, "unavailable": not np.isfinite(q["value"]),
                "note": "W/m2 annual mean on flat ground" if col == "insol_annual" else "degrees; geometry only, circular-orbit approximation (Mars eccentricity about 0.09)",
            })
        for col in ("t_mean", "t_coldest"):
            rows.append({
                "target": name, "column": col, "value": float("nan"), "spread": float("nan"),
                "source": "Mars climate data (none verified)", "evidence_grade": "C", "window": window, "unavailable": True,
                "note": "unavailable: no gridded annual-mean or coldest-month surface-temperature file was found (one search: only single-time TES snapshot images and "
                        "thermal-inertia maps). The Mars Climate Database is a model with its own access route that was not verified or used",
            })
    return rows
