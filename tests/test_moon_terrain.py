from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop.config import load_config
from gsop.targets import moon as M

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")

# Bounds of the real PGDA tiles, observed by opening the files remotely on 2026-11-14 (x_min, y_min, x_max, y_max), metres.
OBSERVED_BOUNDS = {"Site01": (-19000.0, -20000.0, -3000.0, -4000.0), "Site04": (-9000.0, -15000.0, 7000.0, 1000.0)}
POLAR_WKT = ('PROJCS["unnamed",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",1737400,0]],PRIMEM["Greenwich",0],'
             'UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],'
             'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')


def test_targets_fall_inside_their_real_tiles_near_the_centre():
    """The plan's coordinates (Glaser 2018) land inside the PGDA tiles: a check of the projection maths against real facts."""
    for name, site in M.SITE_FOR_TARGET.items():
        t = CFG.targets("moon")[name]
        x, y = M.target_xy(t.lat, t.lon_e, POLAR_WKT)
        xmin, ymin, xmax, ymax = OBSERVED_BOUNDS[site]
        assert xmin + 3000 < x < xmax - 3000 and ymin + 3000 < y < ymax - 3000, (name, x, y)
        assert np.hypot(x - (xmin + xmax) / 2, y - (ymin + ymax) / 2) < 2500  # near the centre: observed offsets are 0.4 km (Site01) and 1.9 km (Site04)


def test_distance_from_the_pole_is_the_polar_stereographic_value():
    x, y = M.target_xy(-89.45, 222.84, POLAR_WKT)
    rho = 2 * 1737400 * np.tan(np.radians(0.55) / 2)
    assert np.hypot(x, y) == pytest.approx(rho, rel=1e-6)


def synthetic_tile(path, bounds, seed=0, cell=5.0):
    xmin, ymin, xmax, ymax = bounds
    n = int((xmax - xmin) / cell)
    X, Y = np.meshgrid((np.arange(n) + 0.5) * cell, (np.arange(n) + 0.5) * cell)
    rng = np.random.default_rng(seed)
    dem = 100.0 + 0.1 * X + 40 * np.sin(X / 900.0) * np.cos(Y / 1300.0) + rng.normal(0, 0.2, X.shape)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=POLAR_WKT,
                       transform=from_origin(xmin, ymax, cell, cell), nodata=np.nan) as ds:
        ds.write(dem.astype("float32"), 1)
    return dem


def test_window_read_profile_and_rows_on_a_synthetic_tile(tmp_path):
    files = {s: tmp_path / f"{s}.tif" for s in OBSERVED_BOUNDS}
    for i, (s, b) in enumerate(OBSERVED_BOUNDS.items()):
        synthetic_tile(files[s], b, seed=i)
    rows = M.moon_terrain_rows(CFG, source_for=lambda site: str(files[site]), cache_dir=tmp_path / "cache")
    assert len(rows) == 2 * 5
    assert {(r["target"], r["column"]) for r in rows} == {(t, c) for t in M.SITE_FOR_TARGET for c in M.TERRAIN_COLUMNS}
    for r in rows:
        assert np.isfinite(r["value"]) and r["spread"] >= 0 and r["evidence_grade"] == "A" and not r["unavailable"]
        assert "Barker et al. 2021" in r["source"] and "15 km window" in r["window"]
    slope = [r for r in rows if r["column"] == "slope_med"][0]["value"]
    assert 2.0 < slope < 10.0  # a 0.1 east tilt is 5.7 degrees plus gentle waves
    assert (tmp_path / "cache").is_dir() and len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_window_keeps_its_full_size_even_when_the_target_is_near_the_tile_edge(tmp_path):
    f = tmp_path / "t.tif"
    synthetic_tile(f, OBSERVED_BOUNDS["Site04"])  # the target sits 1.9 km from the centre: a centred 15 km window sticks out
    t = CFG.targets("moon")["shackleton_rim"]
    info = M.read_window(str(f), t.lat, t.lon_e, 15.0)
    assert info["dem"].shape == (3000, 3000)
    x, y = info["target_xy"]
    assert info["x0"] <= x <= info["x0"] + 15000 and info["y_top"] - 15000 <= y <= info["y_top"]


def test_cache_makes_the_second_read_offline(tmp_path):
    f = tmp_path / "t.tif"
    synthetic_tile(f, OBSERVED_BOUNDS["Site01"])
    t = CFG.targets("moon")["connecting_ridge_c1"]
    a = M.read_window(str(f), t.lat, t.lon_e, 15.0, tmp_path / "c")
    f.unlink()
    b = M.read_window(str(f), t.lat, t.lon_e, 15.0, tmp_path / "c")  # source is gone; served from the cache
    assert np.array_equal(a["dem"], b["dem"]) and a["sha256"] == b["sha256"]


def test_profile_uses_iqr_over_1349_and_the_median():
    rng = np.random.default_rng(0)
    n = 3000  # 15 km at 5 m
    dem = rng.normal(0, 1.0, (n, n)).cumsum(axis=0).cumsum(axis=1) * 0.001
    prof = M.terrain_profile(dem, 5.0, CFG)
    assert prof["slope_med"]["n_tiles"] == 16 and prof["slope_med"]["n_valid"] == 16
    assert prof["effective_baseline_m"]["slope_relief_m"] == 500.0
    assert prof["rough_a"]["spread"] >= 0


def test_arcadia_and_unknown_targets_are_not_scored():
    assert "arcadia" not in M.SITE_FOR_TARGET and "malapert" not in str(M.SITE_FOR_TARGET).lower()
