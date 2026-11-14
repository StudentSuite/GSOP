from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop.config import load_config
from gsop.targets import mars as MA

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
R = MA.MARS_RADIUS_M
WKT = ('PROJCS["SimpleCylindrical Mars",GEOGCS["GCS_Mars",DATUM["D_Mars",SPHEROID["Mars",3396190,0]],PRIMEM["Reference_Meridian",0],'
       'UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],'
       'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')


def write_ti(path, value_fn, lat0=18.4386, lon0=77.5031, half_deg=0.8, cell=100.0, nodata=-3.4028226550889045e+38):
    import math

    half_m = half_deg * math.radians(1.0) * R
    xc, yc = R * math.radians(lon0), R * math.radians(lat0)
    n = int(2 * half_m / cell)
    xs, ys = xc - half_m + (np.arange(n) + 0.5) * cell, yc + half_m - (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    arr = value_fn(np.degrees(X / R), np.degrees(Y / R)).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=WKT,
                       transform=from_origin(xc - half_m, yc + half_m, cell, cell), nodata=nodata) as ds:
        ds.write(arr, 1)


def test_tile_names_follow_the_catalog_pattern_and_jezero_is_00N060E():
    assert MA.themis_tile_name(18.4386, 77.5031) == "00N060E"
    assert MA.themis_tile_name(35.0, 130.0) == "30N120E" and MA.themis_tile_name(10.0, 5.0) == "00N000E"
    assert MA.themis_url(18.4386, 77.5031).endswith("THEMIS_TI_Mosaic_Quant_00N060E_100mpp.cub")
    with pytest.raises(ValueError):
        MA.themis_tile_name(-20.0, 10.0)  # southern names were not verified


def test_themis_value_is_the_median_over_tiles_and_the_three_other_columns_are_unavailable(tmp_path):
    f = tmp_path / "ti.tif"
    write_ti(f, lambda lon, lat: np.full(lon.shape, 357.0))
    rows = MA.mars_thermal_rows(CFG, lambda lat, lon: str(f), tmp_path / "c")
    by = {r["column"]: r for r in rows}
    assert set(by) == {MA.TI_COLUMN, "ti_proxy", "dust_index", "silica_index"} and {r["target"] for r in rows} == {"jezero"}
    assert by[MA.TI_COLUMN]["value"] == pytest.approx(357.0) and by[MA.TI_COLUMN]["spread"] == pytest.approx(0.0, abs=1e-6)
    assert not by[MA.TI_COLUMN]["unavailable"] and by[MA.TI_COLUMN]["evidence_grade"] == "A"
    for col in ("ti_proxy", "dust_index", "silica_index"):
        assert by[col]["unavailable"] and np.isnan(by[col]["value"]) and by[col]["note"].startswith("unavailable:")
    assert "direction and units differ" in by["ti_proxy"]["note"] and "not a dust index" in by["dust_index"]["note"]
    assert "SWING" in by["ti_proxy"]["note"]


def test_spread_and_nodata_handling(tmp_path):
    f = tmp_path / "ti.tif"
    # a west-to-east gradient 300..400 and a nodata hole in the north-west corner
    def fn(lon, lat):
        z = 300.0 + 100.0 * (lon - lon.min()) / (lon.max() - lon.min())
        z[:20, :20] = -3.4028226550889045e+38
        return z
    write_ti(f, fn)
    rows = MA.mars_thermal_rows(CFG, lambda lat, lon: str(f))
    ti = [r for r in rows if r["column"] == MA.TI_COLUMN][0]
    assert 340.0 < ti["value"] < 360.0 and ti["spread"] > 0 and not ti["unavailable"]


def test_all_nodata_is_unavailable(tmp_path):
    f = tmp_path / "ti.tif"
    write_ti(f, lambda lon, lat: np.full(lon.shape, -3.4028226550889045e+38))
    ti = [r for r in MA.mars_thermal_rows(CFG, lambda lat, lon: str(f)) if r["column"] == MA.TI_COLUMN][0]
    assert ti["unavailable"] and np.isnan(ti["value"])


def test_the_mars_gate_input_exists_for_the_d6_check():
    # D6: the target thermal inertia must be at least min_thermal_inertia; the THEMIS row is the input for that check
    assert CFG.gates.thresholds["mars_target_check"]["min_thermal_inertia"] == 100
