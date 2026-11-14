import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop.config import load_config
from gsop.earth import build_parquet as B
from gsop.earth import gee_export as G
from gsop.earth.sanity import SANITY_RANGES

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")
TILE = 6000.0
BANDS, _ = G.planned_bands(CFG)


def make_raster(path, x0, y_top, h=8, w=10, seed=0, ocean_rows=2):
    """Synthetic chunk raster on the global lattice with every exported band and some ocean rows."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for b in BANDS:
        lo, hi, _ = SANITY_RANGES[b]
        arrays[b] = rng.uniform(lo, min(hi, lo + 1.0) if b not in ("t_mean", "t_coldest") else 300, (h, w)).astype("float32")
    arrays["land_frac"][:] = 1.0
    arrays["land_frac"][:ocean_rows] = 0.0  # the northern rows are ocean
    arrays["elevation_mean"] = rng.uniform(0, 3000, (h, w)).astype("float32")
    arrays["t_low_obs"] = (rng.uniform(size=(h, w)) > 0.8).astype("float32")
    arrays["ti_proxy_fallback"][:] = 0.0
    arrays["slope_med"][3, 3] = np.nan  # a missing value
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=len(BANDS), dtype="float32", crs="EPSG:6933",
                       transform=from_origin(x0, y_top, TILE, TILE), nodata=-9999.0) as ds:
        for i, b in enumerate(BANDS, start=1):
            ds.write(np.where(np.isnan(arrays[b]), -9999.0, arrays[b]).astype("float32"), i)
            ds.set_band_description(i, b)
    return arrays


@pytest.fixture
def built(tmp_path):
    x0, y_top = 600000.0, 300000.0
    a = make_raster(tmp_path / "a.tif", x0, y_top, seed=1)
    b = make_raster(tmp_path / "b.tif", x0 + 10 * TILE, y_top, seed=2)  # the chunk to the east
    out = tmp_path / "processed"
    logs = []
    info = B.build_earth_parquet([tmp_path / "a.tif", tmp_path / "b.tif"], CFG, out, max_cells=25, log=logs.append)
    return tmp_path, out, info, (a, b), logs


def test_rows_columns_and_land_filter(built):
    tmp, out, info, (a, b), _ = built
    df = pd.read_parquet(out / "earth_features_6km.parquet")
    assert len(df) == info["rows"] == 2 * 6 * 10  # 2 ocean rows of 8 dropped in each chunk
    assert df.tile_id.is_unique and df.land.all()
    for col in B.FEATURE_COLUMNS + ["tile_id", "lat", "lon", "x_m", "y_m", "t_low_obs", "ti_proxy_fallback",
                                    "water_ice_fraction", "ndvi_median", "builtup_cropland_fraction", "elevation_mean"]:
        assert col in df.columns, col
    assert all(f"{c}_unavailable" in df.columns for c in B.FEATURE_COLUMNS)
    assert df.slope_med.dtype == np.float32 and df.lat.dtype == np.float64


def test_tile_ids_follow_the_global_lattice_and_lat_lon_are_sane(built):
    _, out, info, _, _ = built
    df = pd.read_parquet(out / "earth_features_6km.parquet")
    nx = info["shape"]["nx"]
    ix, iy = df.tile_id % nx, df.tile_id // nx
    grid = B.global_grid(CFG)
    assert np.allclose(grid.x0 + (ix + 0.5) * 6000, df.x_m) and np.allclose(grid.y_top - (iy + 0.5) * 6000, df.y_m)
    assert df.lat.between(-90, 90).all() and df.lon.between(-180, 180).all()
    assert 0 < (df.lat.abs() < 20).mean() <= 1  # 300 km north of the equator at x about 600 km: near lat 2.7, lon 6


def test_nodata_becomes_nan_and_unavailable_flags(built):
    _, out, info, _, logs = built
    df = pd.read_parquet(out / "earth_features_6km.parquet")
    assert df.slope_med.isna().sum() == 2  # one NaN pixel per chunk, both on land rows
    assert df.slope_med_unavailable.sum() == 2
    assert (df.water_index.isna()).all() and "water_index" in info["unavailable_features"]
    assert any("water_index: UNAVAILABLE" in m for m in logs)
    assert "slope_med" not in info["unavailable_features"]


def test_derived_columns_illumination_density_dust_and_flags(built):
    _, out, info, _, _ = built
    df = pd.read_parquet(out / "earth_features_6km.parquet")
    assert df.dust_index.between(0, 1).all() and df.dust_index.min() == 0.0 and df.dust_index.max() == 1.0
    assert info["dust_limits"]["low"] < info["dust_limits"]["high"]
    assert np.allclose(df.sun_max_elev, 90.0 - np.maximum(df.lat.abs() - 23.44, 0.0), atol=1e-3)
    assert df.sun_low_share.between(0, 1).all() and (df.insol_annual > 0).all()
    assert df.air_density.between(0.5, 1.3).all()  # 0-3000 m
    assert df.t_low_obs.dtype == bool and df.t_low_obs.any() and not df.ti_proxy_fallback.any()


def test_grid_json(built):
    _, out, info, _, _ = built
    on_disk = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    assert on_disk["crs"] == "EPSG:6933" and on_disk["cell_size_m"] == 6000
    assert on_disk["shape"]["nx"] > 5000 and on_disk["shape"]["ny"] > 2000
    assert on_disk["datasets"]["dem_nasadem"] == G.NASADEM_ID and on_disk["datasets"]["climatology_years"] == [2015, 2024]
    assert on_disk["rows"] == 120 and len(on_disk["columns"]) == len(set(on_disk["columns"]))
    assert "raster_bands_seen" in on_disk and "config_hash" in on_disk


def test_roundtrip_matches_the_rasters_and_detects_corruption(built, tmp_path):
    tmp, out, _, _, _ = built
    rasters = [tmp / "a.tif", tmp / "b.tif"]
    res = B.roundtrip_check(out / "earth_features_6km.parquet", rasters, CFG, n=5, seed=3)
    assert len(res) == 5
    df = pd.read_parquet(out / "earth_features_6km.parquet")
    df.loc[df.index[0], "relief"] = df.loc[df.index[0], "relief"] + 10
    bad = tmp_path / "bad.parquet"
    df.to_parquet(bad)
    with pytest.raises(AssertionError, match="relief"):
        B.roundtrip_check(bad, rasters, CFG, n=len(df), seed=0)


def test_memory_blocks_never_exceed_the_budget():
    assert B.block_rows(width=1000, max_cells=25_000) == 25
    assert B.block_rows(width=1000, max_cells=10) == 1


def test_a_raster_without_land_frac_is_refused(tmp_path):
    p = tmp_path / "x.tif"
    with rasterio.open(p, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:6933",
                       transform=from_origin(600000.0, 300000.0, TILE, TILE)) as ds:
        ds.write(np.zeros((2, 2), dtype="float32"), 1)
        ds.set_band_description(1, "slope_med")
    with pytest.raises(ValueError, match="land_frac"):
        B.build_earth_parquet([p], CFG, tmp_path / "o", max_cells=10, log=lambda m: None)
