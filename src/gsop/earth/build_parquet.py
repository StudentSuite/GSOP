"""Build ``data/processed/earth_features_6km.parquet`` and ``grid.json`` from the exported chunk rasters (issue B18).

* Reads each chunk raster in row blocks (never more cells at once than the ``gsop doctor`` chunk budget).
* One row per **land** tile. ``tile_id = iy * nx + ix`` on the global 6 km lattice, so ids are the same whichever
  chunks or test region were exported.
* Nodata becomes NaN. Illumination (``sun_max_elev``, ``sun_low_share``, ``insol_annual``) and ``air_density`` are
  computed here from latitude and ``elevation_mean`` with the shared numpy code; ``dust_index_raw`` is scaled to
  0..1 with land-tile percentiles (``gsop.dust``).
* Flags: ``t_low_obs``, ``ti_proxy_fallback``, and one ``<feature>_unavailable`` boolean per feature (True where the
  value is NaN). Features that were not exported at all (for example ``water_index``) are all-NaN and listed in
  ``grid.json`` under ``unavailable_features``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer

from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.dust import normalise_dust
from gsop.earth import gee_export as G
from gsop.features_misc import air_density
from gsop.illumination import compute_illumination

FEATURE_COLUMNS = [
    "slope_med", "slope_p90", "relief", "rough_a", "rough_b", "sun_max_elev", "sun_low_share", "insol_annual",
    "t_mean", "t_coldest", "ti_proxy", "dust_index", "water_index", "silica_index", "air_density",
]
RASTER_FEATURES = {  # parquet column <- raster band, for bands that are copied as they are
    "slope_med": "slope_med", "slope_p90": "slope_p90", "relief": "relief", "rough_a": "rough_a", "rough_b": "rough_b",
    "t_mean": "t_mean", "t_coldest": "t_coldest", "ti_proxy": "ti_proxy", "water_index": "water_index",
    "silica_index": "silica_index",
}
FLAG_BANDS = {"t_low_obs": "t_low_obs", "ti_proxy_fallback": "ti_proxy_fallback"}


def gate_band_map(cfg: Config) -> dict[str, str]:
    """parquet gate column <- raster band, from ``features.yaml`` ``gate_layers[*].band``."""
    return {name: spec["band"] for name, spec in cfg.features.gate_layers.items()}


@dataclass
class GridInfo:
    x0: int  # west edge of the global lattice (m, EPSG:6933)
    y_top: int  # north edge
    nx: int
    ny: int
    tile_m: int


def global_grid(cfg: Config) -> GridInfo:
    chunks = G.plan_chunks(cfg)
    tile = int(cfg.features.operators.tile_m)
    x0, x1 = min(c.x0 for c in chunks), max(c.x1 for c in chunks)
    y0, y1 = min(c.y0 for c in chunks), max(c.y1 for c in chunks)
    return GridInfo(x0, y1, (x1 - x0) // tile, (y1 - y0) // tile, tile)


def _read_band(ds, name: str, window) -> np.ndarray:
    names = [d or f"band_{i + 1}" for i, d in enumerate(ds.descriptions)]
    if name not in names:
        return None
    return ds.read(names.index(name) + 1, window=window, masked=True).astype("float64").filled(np.nan)


def block_rows(width: int, max_cells: int) -> int:
    return max(1, int(max_cells // max(width, 1)))


def scan_dust_limits(rasters: Sequence[Path], cfg: Config, grid: GridInfo, max_cells: int) -> Optional[tuple[float, float]]:
    """Pass 1: percentile limits of ``dust_index_raw`` over land tiles of every raster."""
    from gsop.dust import percentile_limits

    lo_p, hi_p = cfg.features.earth_build["dust_percentiles"]
    land_min = cfg.features.earth_build["land_min_fraction"]
    values: list[np.ndarray] = []
    for path in rasters:
        with rasterio.open(path) as ds:
            step = block_rows(ds.width, max_cells)
            for r0 in range(0, ds.height, step):
                win = rasterio.windows.Window(0, r0, ds.width, min(step, ds.height - r0))
                dust, land = _read_band(ds, "dust_index_raw", win), _read_band(ds, "land_frac", win)
                if dust is None or land is None:
                    continue
                v = dust[(land >= land_min) & np.isfinite(dust)]
                if v.size:
                    values.append(v)
    if not values:
        return None
    return percentile_limits(np.concatenate(values), lo_p, hi_p)


def build_block(ds, win, cfg: Config, grid: GridInfo, to_lonlat: Transformer, dust_limits: Optional[tuple[float, float]]) -> Optional[pd.DataFrame]:
    land_frac = _read_band(ds, "land_frac", win)
    if land_frac is None:
        raise ValueError(f"{ds.name}: band land_frac is missing (needed to decide which tiles are land)")
    land = land_frac >= cfg.features.earth_build["land_min_fraction"]
    if not land.any():
        return None
    rows, cols = np.nonzero(land)
    tf = ds.window_transform(win)
    x = tf.c + (cols + 0.5) * tf.a
    y = tf.f + (rows + 0.5) * tf.e
    lon, lat = to_lonlat.transform(x, y)
    ix = np.rint((x - grid.x0) / grid.tile_m - 0.5).astype("int64")
    iy = np.rint((grid.y_top - y) / grid.tile_m - 0.5).astype("int64")
    data: dict[str, np.ndarray] = {
        "tile_id": iy * grid.nx + ix, "lat": lat, "lon": lon, "x_m": x, "y_m": y, "land": np.ones(rows.size, dtype=bool),
    }
    for col, band in RASTER_FEATURES.items():
        arr = _read_band(ds, band, win)
        data[col] = (arr[land] if arr is not None else np.full(rows.size, np.nan)).astype("float32")
    # dust: scaled with the land-tile percentile limits
    raw = _read_band(ds, "dust_index_raw", win)
    if raw is not None and dust_limits is not None:
        data["dust_index"] = normalise_dust(raw[land], *dust_limits).astype("float32")
    else:
        data["dust_index"] = np.full(rows.size, np.nan, dtype="float32")
    # illumination from latitude (flat ground, Earth constants)
    ill = compute_illumination(lat, cfg.features.illumination.bodies["earth"], cfg.features.illumination)
    for k, v in ill.items():
        data[k] = v.astype("float32")
    # air density from the mean elevation of the tile
    elev = _read_band(ds, "elevation_mean", win)
    if elev is not None:
        e = elev[land]
        ok = np.isfinite(e) & (e <= cfg.features.air_density["earth"]["max_elevation_m"])
        rho = np.full(e.shape, np.nan)
        rho[ok] = air_density(e[ok], cfg.features.air_density["earth"], cfg.features.air_density["universal_gas_constant_j_mol_k"])
        data["air_density"] = rho.astype("float32")
    else:
        data["air_density"] = np.full(rows.size, np.nan, dtype="float32")
    data["elevation_mean"] = (elev[land] if elev is not None else np.full(rows.size, np.nan)).astype("float32")
    # gate layers and flags
    for col, band in gate_band_map(cfg).items():
        arr = _read_band(ds, band, win)
        data[col] = (arr[land] if arr is not None else np.full(rows.size, np.nan)).astype("float32")
    for col, band in FLAG_BANDS.items():
        arr = _read_band(ds, band, win)
        data[col] = (np.nan_to_num(arr[land], nan=0.0) > 0.5) if arr is not None else np.zeros(rows.size, dtype=bool)
    df = pd.DataFrame(data)
    for col in FEATURE_COLUMNS:
        df[f"{col}_unavailable"] = df[col].isna()
    return df


def build_earth_parquet(
    rasters: Sequence[Path],
    cfg: Config,
    out_dir: Path,
    max_cells: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Pass 1 (dust limits), pass 2 (rows to parquet in blocks), then ``grid.json``. Returns the grid info dict."""
    from gsop.doctor import chunk_cells, collect

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters = [Path(r) for r in rasters]
    if not rasters:
        raise ValueError("no raster files given")
    grid = global_grid(cfg)
    if max_cells is None:
        info = collect()
        max_cells = max(1, chunk_cells(int(info["ram_free_gb"] * 1e9), len(FEATURE_COLUMNS) + 8, cfg.run.chunk_ram_fraction))
    to_lonlat = Transformer.from_crs(cfg.features.operators.grid_crs, "EPSG:4326", always_xy=True)
    dust_limits = scan_dust_limits(rasters, cfg, grid, max_cells)
    log(f"[build] dust limits (land-tile percentiles): {dust_limits}")

    parquet_path = out_dir / "earth_features_6km.parquet"
    tmp_path = out_dir / ".earth_features_6km.parquet.tmp"
    writer = None
    n_rows, nan_counts, seen_bands = 0, {c: 0 for c in FEATURE_COLUMNS}, set()
    for path in rasters:
        with rasterio.open(path) as ds:
            seen_bands.update(d for d in ds.descriptions if d)
            step = block_rows(ds.width, max_cells)
            for r0 in range(0, ds.height, step):
                win = rasterio.windows.Window(0, r0, ds.width, min(step, ds.height - r0))
                df = build_block(ds, win, cfg, grid, to_lonlat, dust_limits)
                if df is None:
                    continue
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                writer.write_table(table)
                n_rows += len(df)
                for c in FEATURE_COLUMNS:
                    nan_counts[c] += int(df[c].isna().sum())
    if writer is None:
        raise ValueError("no land tiles found in the given rasters")
    writer.close()
    tmp_path.replace(parquet_path)

    unavailable = sorted(c for c in FEATURE_COLUMNS if nan_counts[c] == n_rows)
    warn = cfg.features.earth_build["nan_warn_fraction"]
    for c in FEATURE_COLUMNS:
        share = nan_counts[c] / n_rows
        if c in unavailable:
            log(f"[build] {c}: UNAVAILABLE (all {n_rows:,} tiles are NaN)")
        elif share > warn:
            log(f"[build] WARNING {c}: {share:.1%} of land tiles are NaN (limit {warn:.0%})")
    grid_info = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": cfg.features.operators.grid_crs,
        "cell_size_m": grid.tile_m,
        "origin": {"x_west_m": grid.x0, "y_north_m": grid.y_top},
        "shape": {"ny": grid.ny, "nx": grid.nx},
        "tile_id": "iy * nx + ix, iy counted from the north edge, ix from the west edge",
        "rows": n_rows,
        "columns": [str(c) for c in pq.ParquetFile(parquet_path).schema_arrow.names],
        "feature_columns": FEATURE_COLUMNS,
        "unavailable_features": unavailable,
        "nan_share": {c: round(nan_counts[c] / n_rows, 6) for c in FEATURE_COLUMNS},
        "raster_bands_seen": sorted(seen_bands),
        "dust_limits": None if dust_limits is None else {"low": dust_limits[0], "high": dust_limits[1]},
        "config_hash": cfg.config_hash(["features", "run"]),
        "datasets": {
            "dem_nasadem": G.NASADEM_ID, "dem_copernicus": G.COPERNICUS_ID, "temperature": G.MODIS_LST_ID,
            "ndvi": G.MODIS_NDVI_ID, "land_cover": G.MODIS_LANDCOVER_ID, "silica": G.ASTER_GED_ID,
            "dust": G.MERRA2_AER_ID, "precipitation_aridity": G.TERRACLIMATE_ID,
            "climatology_years": [cfg.features.temperature.start_year, cfg.features.temperature.end_year],
            "land_cover_year": cfg.features.land_cover["year"],
        },
    }
    atomic_write_text(out_dir / "grid.json", json.dumps(grid_info, indent=2) + "\n")
    log(f"[build] wrote {parquet_path} ({n_rows:,} land tiles, {len(grid_info['columns'])} columns) and grid.json")
    return grid_info


def roundtrip_check(parquet_path: Path, rasters: Sequence[Path], cfg: Config, n: int = 5, seed: int = 0) -> list[dict]:
    """Pick ``n`` random tiles, re-read their raster pixels and compare with the parquet values. Raises on mismatch."""
    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    picks = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
    results = []
    for _, row in picks.iterrows():
        found = False
        for path in rasters:
            with rasterio.open(path) as ds:
                r, c = ds.index(row["x_m"], row["y_m"])
                if not (0 <= r < ds.height and 0 <= c < ds.width):
                    continue
                for col, band in RASTER_FEATURES.items():
                    arr = _read_band(ds, band, rasterio.windows.Window(c, r, 1, 1))
                    expected = np.nan if arr is None else arr[0, 0]
                    got = row[col]
                    if not (np.isnan(expected) and np.isnan(got)) and not np.isclose(expected, got, rtol=1e-5, equal_nan=False):
                        raise AssertionError(f"tile {row['tile_id']} {col}: raster {expected} != parquet {got}")
                found = True
                results.append({"tile_id": int(row["tile_id"]), "raster": str(path), "row": r, "col": c})
                break
        if not found:
            raise AssertionError(f"tile {row['tile_id']} not found in any raster")
    return results
