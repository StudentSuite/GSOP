"""s2 targets --body moon: Moon target profiles. This file holds the LOLA terrain part (issue B25); temperature and silica
(B26) and illumination and PSR water_index (B27) are added by those issues.

Terrain source: the Barker et al. (2021) site-specific LOLA DEMs on NASA PGDA, 5 m per pixel GeoTIFFs in a south polar
stereographic grid (sphere radius 1,737,400 m, metres): ``Site01`` is Connecting Ridge, ``Site04`` is Shackleton rim
(site numbering of Mazarico et al. 2011, from the PGDA site list). Each tile is 16 km square (3200 x 3200 px). Only a
window is read (``rasterio`` on the URL, or a local file): the full-planet DEM is never downloaded.

Target value = median over the 6 km tiles (3 km stride) inside a 15 km window centred on the target; spread = IQR / 1.349.
Terrain features come from ``gsop.operators`` (the one operator shared with Earth and Mars). The window is read without a
halo, so border tiles lose their outermost slope cells (at most 1 of 12 per side, under the 20% missing-cell limit).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window

from gsop import operators as O
from gsop.atomic import atomic_write_bytes, atomic_write_text
from gsop.config import Config

PGDA_BASE = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp"
SITE_FOR_TARGET = {"connecting_ridge_c1": "Site01", "shackleton_rim": "Site04"}
TERRAIN_COLUMNS = ("slope_med", "slope_p90", "relief", "rough_a", "rough_b")
MOON_SPHERE = "+proj=longlat +a=1737400 +b=1737400 +no_defs"
SOURCE_TEXT = "LOLA 5 m DEM, Barker et al. 2021 (NASA PGDA {site}_final_adj_5mpp_surf.tif)"
EVIDENCE_GRADE = "A"  # measured (but about 90% of the 5 m pixels are interpolated between LOLA tracks, Barker 2021)


def lola_url(site: str) -> str:
    return f"{PGDA_BASE}/{site}/{site}_final_adj_5mpp_surf.tif"


def target_xy(lat: float, lon_east: float, crs: Any) -> tuple[float, float]:
    """Target position in the DEM's own (polar stereographic, metres) coordinates."""
    to_dem = Transformer.from_crs(CRS.from_proj4(MOON_SPHERE), CRS.from_user_input(crs), always_xy=True)
    x, y = to_dem.transform(((lon_east + 180.0) % 360.0) - 180.0, lat)
    return float(x), float(y)


def _open_path(source: str) -> str:
    return f"/vsicurl/{source}" if str(source).startswith("http") else str(source)


def read_window(source: str, lat: float, lon_east: float, window_km: float, cache_dir: Optional[Path] = None) -> dict:
    """Read a ``window_km`` square DEM window centred on the target, shifted to stay inside the file. Windowed read only.

    Returns ``{"dem", "cell_m", "x0", "y_top", "crs", "bounds", "source", "sha256"}``. With ``cache_dir`` the window is cached as
    ``.npy`` plus a ``.json`` sidecar so reruns are offline.
    """
    key = hashlib.sha256(f"{source}|{lat}|{lon_east}|{window_km}".encode()).hexdigest()[:16]
    if cache_dir is not None:
        npy, meta = Path(cache_dir) / f"lola_{key}.npy", Path(cache_dir) / f"lola_{key}.json"
        if npy.is_file() and meta.is_file():
            info = json.loads(meta.read_text(encoding="utf-8"))
            info["dem"] = np.load(npy)
            return info
    # Larger HTTP chunks and more retries: with GDAL's defaults a strip read over HTTP failed once with a TIFF strip error
    # (2026-11-14); the same read succeeded with these settings. Same source, no mirror.
    with rasterio.Env(GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="3", CPL_VSIL_CURL_CHUNK_SIZE="4194304",
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(_open_path(source)) as ds:
            x, y = target_xy(lat, lon_east, ds.crs)
            half = window_km * 500.0
            left, top = x - half, y + half
            n = int(round(window_km * 1000.0 / ds.res[0]))
            n_w, n_h = min(n, ds.width), min(n, ds.height)
            # centred on the target where possible; shifted (not shrunk) to stay inside the file, so the window keeps its size
            col0 = min(max(int(round((left - ds.bounds.left) / ds.res[0])), 0), ds.width - n_w)
            row0 = min(max(int(round((ds.bounds.top - top) / ds.res[1])), 0), ds.height - n_h)
            dem = ds.read(1, window=Window(col0, row0, n_w, n_h), masked=True).astype("float64")
            scale, offset = (ds.scales[0] or 1.0), (ds.offsets[0] or 0.0)  # PDS maps store scaled integers; the DEM has scale 1
            dem = (dem * scale + offset).filled(np.nan)
            info = {
                "cell_m": float(ds.res[0]), "x0": ds.bounds.left + col0 * ds.res[0], "y_top": ds.bounds.top - row0 * ds.res[1],
                "crs": ds.crs.to_wkt(), "bounds": list(ds.bounds), "source": source, "target_xy": [x, y],
                "shape": [int(dem.shape[0]), int(dem.shape[1])],
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


def terrain_profile(dem: np.ndarray, cell_m: float, cfg: Config) -> dict:
    """Median and IQR/1.349 of each terrain feature over the tiles of ``dem`` (tile and stride from ``targets.yaml``)."""
    ops = cfg.features.operators
    ts = cfg.target_settings
    feats = O.tile_features(
        dem, cell_m, tile_m=ops.tile_m, baseline_m=ops.baseline_m,
        rough_baselines_m=(float(cfg.features.columns["rough_a"].extra["baseline_m"]), float(cfg.features.columns["rough_b"].extra["baseline_m"])),
        stride_m=ts.tile_stride_km * 1000.0, max_nan_fraction=ops.max_nan_fraction,
    )
    out = {}
    for col in TERRAIN_COLUMNS:
        v = feats[col][np.isfinite(feats[col])]
        if v.size == 0:
            out[col] = {"value": float("nan"), "spread": float("nan"), "n_tiles": int(feats[col].size), "n_valid": 0}
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        out[col] = {"value": float(med), "spread": float((q3 - q1) / ts.spread_iqr_divisor), "n_tiles": int(feats[col].size), "n_valid": int(v.size)}
    out["effective_baseline_m"] = feats["effective_baseline_m"]
    return out


def moon_terrain_rows(cfg: Config, source_for=lola_url, cache_dir: Optional[Path] = None) -> list[dict]:
    """One row per (target, terrain column): the format ``gsop s2 targets`` (issue B32) assembles."""
    rows = []
    window_km = cfg.target_settings.target_window_km
    for name, target in cfg.targets("moon").items():
        site = SITE_FOR_TARGET.get(name)
        if site is None or target.lat is None:
            continue
        info = read_window(source_for(site), target.lat, target.lon_e, window_km, cache_dir)
        prof = terrain_profile(info["dem"], info["cell_m"], cfg)
        window = (f"{window_km:g} km window centred on the target, {cfg.target_settings.tile_stride_km:g} km stride, "
                  f"{cfg.features.operators.tile_m / 1000:g} km tiles; read {info['shape'][1]}x{info['shape'][0]} px of {info['cell_m']:g} m from {site}")
        for col in TERRAIN_COLUMNS:
            p = prof[col]
            rows.append({
                "target": name, "column": col, "value": p["value"], "spread": p["spread"],
                "source": SOURCE_TEXT.format(site=site), "evidence_grade": EVIDENCE_GRADE, "window": window,
                "unavailable": not np.isfinite(p["value"]),
                "note": f"{p['n_valid']} of {p['n_tiles']} tiles valid; effective baseline {prof['effective_baseline_m']}",
            })
    return rows


# ===========================================================================================================
# Illumination targets and PSR water_index (issue B27)
# ===========================================================================================================
LPSR_URL = "https://pgda.gsfc.nasa.gov/data/MoonIllumination/LPSR_85S_060M_201608.TIF"
PSR_SHADOW_THRESHOLD = 0.5  # scaled LPSR values are exactly 0.0 or 1.0; 1.0 = permanent shadow (verified, see SPEC)
GLASER_KEYS = ("lit_surface_pct", "lit_2m_pct", "max_shadow_h_surface", "max_shadow_h_2m", "nearest_psr_m")


def polar_xy_to_lat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Latitude (degrees, south negative) of south polar stereographic points (sphere radius 1,737,400 m)."""
    rho = np.hypot(x, y)
    return -90.0 + np.degrees(2.0 * np.arctan(rho / (2.0 * 1_737_400.0)))


def _summarise(values: np.ndarray, divisor: float) -> dict:
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if not v.size:
        return {"value": float("nan"), "spread": float("nan"), "n_tiles": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"value": float(med), "spread": float((q3 - q1) / divisor), "n_tiles": int(v.size)}


def moon_illumination_rows(cfg: Config) -> list[dict]:
    """``sun_max_elev`` and ``sun_low_share`` at the two Moon targets (rule in docs/SPEC.md, issue B27).

    RULE. The two columns describe the Sun's geometry in the sky (highest elevation over the year, share of daylight below 5
    degrees), so they are computed by the shared analytic module ``gsop.illumination`` with the Moon obliquity, at the centres
    of the window's 6 km tiles (median, IQR / 1.349). Glaser et al. (2018) numbers (``lit_surface_pct``, ``lit_2m_pct``, shadow
    hours, nearest PSR distance, from ``targets.yaml``) measure how much of the time terrain leaves the site lit. They are a
    different quantity from both columns, so they are **not converted** into either; they are carried in each row's note as the
    evidence a reader can compare with. Receiver height 2 m applies to Glaser's ``*_2m`` values only.
    """
    from gsop.illumination import compute_illumination
    from gsop.targets.diviner import polar_xy_m, tile_edges

    ts = cfg.target_settings
    body = cfg.features.illumination.bodies["moon"]
    rows = []
    for name, t in cfg.targets("moon").items():
        if t.lat is None:
            continue
        cx, cy = polar_xy_m(t.lat, t.lon_e)
        tile = cfg.features.operators.tile_m
        corners = tile_edges(cx, cy, ts.target_window_km * 1000, tile, ts.tile_stride_km * 1000)
        centres = np.array([(x0 + tile / 2, y0 + tile / 2) for x0, y0 in corners])
        lat = polar_xy_to_lat(centres[:, 0], centres[:, 1])
        ill = compute_illumination(lat, body, cfg.features.illumination)
        glaser = ", ".join(f"{k}={t.attributes[k]}" for k in GLASER_KEYS if k in t.attributes)
        for col in ("sun_max_elev", "sun_low_share"):
            p = _summarise(ill[col], ts.spread_iqr_divisor)
            rows.append({
                "target": name, "column": col, "value": p["value"], "spread": p["spread"],
                "source": "computed: gsop.illumination (analytic, flat horizon) with the Moon obliquity from features.yaml",
                "evidence_grade": "B",
                "window": f"{ts.target_window_km:g} km window, {ts.tile_stride_km:g} km stride, {tile / 1000:g} km tiles (centre latitudes)",
                "unavailable": not np.isfinite(p["value"]),
                "note": f"geometry only, no terrain shadowing. Glaser 2018 (not converted, receiver height {ts.receiver_height_m:g} m for *_2m): {glaser}",
            })
    return rows


def psr_share_rows(cfg: Config, source: str = LPSR_URL, cache_dir: Optional[Path] = None) -> list[dict]:
    """``water_index`` (Moon) = share of the tile in permanent shadow, from the LOLA PSR map (Mazarico et al. 2011, 2016 release)."""
    from gsop.targets.diviner import polar_xy_m, tile_edges

    ts = cfg.target_settings
    tile = cfg.features.operators.tile_m
    rows = []
    for name, t in cfg.targets("moon").items():
        if t.lat is None:
            continue
        info = read_window(source, t.lat, t.lon_e, ts.target_window_km, cache_dir)
        psr = info["dem"]  # scaled values: 0.0 sunlit at some time, 1.0 permanently shadowed
        cell = info["cell_m"]
        shadow = np.where(np.isfinite(psr), (psr > PSR_SHADOW_THRESHOLD).astype("float64"), np.nan)
        shares = []
        for x0, y0 in tile_edges(*info["target_xy"], ts.target_window_km * 1000, tile, ts.tile_stride_km * 1000):
            c0, r0 = int(round((x0 - info["x0"]) / cell)), int(round((info["y_top"] - (y0 + tile)) / cell))
            n = int(round(tile / cell))
            block = shadow[max(r0, 0): r0 + n, max(c0, 0): c0 + n]
            shares.append(float(np.nanmean(block)) if block.size and np.isfinite(block).any() else np.nan)
        p = _summarise(np.array(shares), ts.spread_iqr_divisor)
        rows.append({
            "target": name, "column": "water_index", "value": p["value"], "spread": p["spread"],
            "source": "LOLA permanent shadow map LPSR_85S_060M_201608 (Mazarico et al. 2011; NASA PGDA)", "evidence_grade": "B",
            "window": f"{ts.target_window_km:g} km window, {ts.tile_stride_km:g} km stride, {tile / 1000:g} km tiles; {psr.shape[1]}x{psr.shape[0]} px of {cell:g} m",
            "unavailable": not np.isfinite(p["value"]),
            "note": f"share of tile in permanent shadow (modelled from LOLA topography over an 18.6-year cycle); Glaser nearest_psr_m={t.attributes.get('nearest_psr_m')} not used because the map is reachable",
        })
    return rows
