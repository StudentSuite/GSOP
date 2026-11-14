import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop.config import load_config
from gsop.targets import mars as MA

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
R = MA.MARS_RADIUS_M
EQUIRECT_WKT = ('PROJCS["Equirectangular Mars",GEOGCS["GCS_Mars",DATUM["D_Mars",SPHEROID["Mars_localRadius",3396190,0]],PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],'
                'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')
CELL = 463.0935415503709


def write_mola(path, elevation_fn, lat0=18.4386, lon0=77.5031, half_deg=1.5, nodata_box=None):
    """A synthetic MOLA tile in the real layout (int16, equirectangular metres, 463.09 m pixels, nodata -32768)."""
    half_m = half_deg * math.radians(1.0) * R
    x_c, y_c = R * math.radians(lon0), R * math.radians(lat0)
    n = int(2 * half_m / CELL)
    xs = x_c - half_m + (np.arange(n) + 0.5) * CELL
    ys = y_c + half_m - (np.arange(n) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    lon, lat = np.degrees(X / R), np.degrees(Y / R)
    z = elevation_fn(lon, lat).astype("float64")
    dn = np.rint(z).astype("int16")
    if nodata_box:
        dn[nodata_box] = -32768
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="int16", crs=EQUIRECT_WKT,
                       transform=from_origin(x_c - half_m, y_c + half_m, CELL, CELL), nodata=-32768) as ds:
        ds.write(dn, 1)


def plane_in_true_metres(slope_deg, lat0=18.4386, lon0=77.5031):
    """Elevation of a plane that rises eastward at ``slope_deg`` measured in TRUE metres on the ground."""
    g = math.tan(math.radians(slope_deg))
    return lambda lon, lat: 1000.0 + g * (R * np.cos(np.radians(lat0)) * np.radians(lon - lon0))


@pytest.mark.parametrize("angle", [3.0, 6.0])
def test_east_west_cells_are_corrected_for_latitude(tmp_path, angle):
    """Equirectangular cells are about 439 m wide at Jezero. A plane tilted 'angle' degrees east must come out as 'angle'."""
    f = tmp_path / "mola.tif"
    write_mola(f, plane_in_true_metres(angle))
    rows = MA.mars_terrain_rows(CFG, str(f), cache_dir=tmp_path / "c")
    by = {r["column"]: r for r in rows}
    assert by["slope_med"]["value"] == pytest.approx(angle, rel=0.01)  # int16 rounding of metres adds a little noise
    assert by["slope_p90"]["value"] == pytest.approx(angle, rel=0.01)
    # a naive reading (treating the pixels as square 463 m cells) would give atan(tan(angle) / cos(lat)): not what we got
    naive = math.degrees(math.atan(math.tan(math.radians(angle)) / math.cos(math.radians(18.4386))))
    assert abs(by["slope_med"]["value"] - naive) > 0.05 * angle


def test_rows_are_for_jezero_only_with_grades_and_the_baseline_note(tmp_path):
    f = tmp_path / "mola.tif"
    write_mola(f, plane_in_true_metres(4.0))
    rows = MA.mars_terrain_rows(CFG, str(f))
    assert {r["target"] for r in rows} == {"jezero"}  # Arcadia has no coordinates (D10)
    assert {r["column"] for r in rows} == set(MA.TERRAIN_COLUMNS) | {"elevation_mean"}  # elevation feeds the Mars target check (D6)
    for r in rows:
        assert r["evidence_grade"] == "A" and not r["unavailable"] and "MOLA DEM 463 m" in r["source"]
        assert "500 m equal-area grid" in r["window"]
        if r["column"] in MA.TERRAIN_COLUMNS:
            assert "sha256" in r["note"]
    elev = [r for r in rows if r["column"] == "elevation_mean"][0]
    assert "areoid" in elev["note"] and elev["value"] == pytest.approx(1000.0, abs=5.0)  # the synthetic plane is 1000 m at the target
    ra = [r for r in rows if r["column"] == "rough_a"][0]
    assert "rough_a baseline is 500 m, not 100 m" in ra["note"]
    assert [r for r in rows if r["column"] == "relief"][0]["value"] == pytest.approx(math.tan(math.radians(4.0)) * 11 * 500.0, rel=0.05)


def test_window_has_the_expected_size_and_nodata_becomes_nan(tmp_path):
    f = tmp_path / "mola.tif"
    write_mola(f, plane_in_true_metres(4.0), nodata_box=(slice(0, None), slice(0, None)))  # everything nodata
    info = MA.read_window(str(f), 18.4386, 77.5031, 15.0)
    assert info["dem"].shape == (30, 30) and np.isnan(info["dem"]).all()
    rows = MA.mars_terrain_rows(CFG, str(f))
    assert all(r["unavailable"] for r in rows)


def test_same_inputs_give_the_same_fingerprint_for_the_cross_os_comparison(tmp_path):
    f = tmp_path / "mola.tif"
    write_mola(f, plane_in_true_metres(4.0))
    a = MA.read_window(str(f), 18.4386, 77.5031, 15.0)
    b = MA.read_window(str(f), 18.4386, 77.5031, 15.0)
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64


def test_local_grid_is_equal_area_centred_on_the_target():
    crs = MA.local_crs(18.4386, 77.5031)
    from pyproj import Transformer

    x, y = Transformer.from_crs(
        "+proj=longlat +a=3396190 +b=3396190 +no_defs", crs, always_xy=True).transform(77.5031, 18.4386)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
