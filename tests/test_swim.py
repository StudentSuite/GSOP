from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from gsop.config import load_config
from gsop.targets import mars as MA

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")
R = MA.MARS_RADIUS_M
WKT = ('PROJCS["Mars_Equidistant_Cylindrical",GEOGCS["GCS_Mars_2000_Sphere",DATUM["D_IAU/IAG_2000",SPHEROID["IAU/IAG_2000",3396190,0]],'
       'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
       'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')


def write_swim(path, fn, west_m=-10669440.0, north_m=3555000.0, w=7113, h=1185, nodata=-30.0):
    arr = fn(np.indices((h, w))).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=WKT,
                       transform=from_origin(west_m, north_m, 3000.0, 3000.0), nodata=nodata) as ds:
        ds.write(arr, 1)


def test_consistency_rescaling_is_one_composite_in_0_1():
    assert MA.consistency_to_water_index(-1.0) == 0.0 and MA.consistency_to_water_index(1.0) == 1.0
    assert MA.consistency_to_water_index(0.0) == 0.5
    assert MA.consistency_to_water_index(0.58) == pytest.approx(0.79)
    assert np.isnan(MA.consistency_to_water_index(np.nan))


def test_jezero_row_uses_the_combined_consistency_only(tmp_path):
    f = tmp_path / "ci.tif"
    write_swim(f, lambda ij: np.full(ij[0].shape, 0.2))
    rows = MA.swim_water_index_rows(CFG, str(f), tmp_path / "c")
    assert [(r["target"], r["column"]) for r in rows] == [("jezero", "water_index")]
    r = rows[0]
    assert r["value"] == pytest.approx(0.6) and r["spread"] == pytest.approx(0.0, abs=1e-6) and not r["unavailable"]
    assert "ONE composite (D7)" in r["note"] and "not counted again" in r["note"] and r["evidence_grade"] == "B"
    assert "Northern_Ci.tif" in r["source"]


def test_nodata_gives_unavailable(tmp_path):
    f = tmp_path / "ci.tif"
    write_swim(f, lambda ij: np.full(ij[0].shape, -30.0))
    assert MA.swim_water_index_rows(CFG, str(f))[0]["unavailable"]


def test_arcadia_is_never_given_coordinates_from_the_swim_footprint(tmp_path):
    f = tmp_path / "pilot.tif"
    write_swim(f, lambda ij: np.zeros(ij[0].shape), west_m=-10195248.0, w=394, h=1185)
    fp = MA.swim_arcadia_footprint(str(f))
    assert fp["is_a_site_centre"] is False and fp["lon_west"] == pytest.approx(-172.0, abs=0.01) and fp["lat_north"] == pytest.approx(59.975, abs=0.01)
    assert 0 < fp["lon_east"] - fp["lon_west"] < 25 and fp["lat_north"] - fp["lat_south"] > 55  # a swath, not a site
    arcadia = CFG.targets("mars")["arcadia"]
    assert arcadia.lat is None and arcadia.lon_e is None and arcadia.status == "unscored_no_target_data"
    raw = yaml.safe_load((ROOT / "config" / "targets.yaml").read_text(encoding="utf-8"))
    assert raw["mars"]["arcadia"]["lat"] is None and raw["mars"]["arcadia"]["lon_e"] is None
    assert not any(r["target"] == "arcadia" for r in MA.swim_water_index_rows(CFG, str(f)))
