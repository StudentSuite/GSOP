import ast
import math
from pathlib import Path

import numpy as np
import pytest

from gsop import illumination as I
from gsop.config import load_config
from gsop.features_misc import air_density
from gsop.targets import mars as MA
import test_mars_terrain as T  # synthetic MOLA writer

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def test_rows_for_jezero_and_the_unavailable_temperatures(tmp_path):
    f = tmp_path / "mola.tif"
    T.write_mola(f, lambda lon, lat: np.full(lon.shape, -2500.0))  # flat at -2.5 km above the areoid
    rows = MA.mars_climate_rows(CFG, str(f), tmp_path / "c")
    by = {(r["target"], r["column"]): r for r in rows}
    assert {c for _, c in by} == {"air_density", "insol_annual", "sun_max_elev", "t_mean", "t_coldest"} and {t for t, _ in by} == {"jezero"}
    mars = CFG.features.illumination.bodies["mars"]
    ad = CFG.features.air_density
    expected_rho = float(air_density(-2500.0, ad["mars"], ad["universal_gas_constant_j_mol_k"]))
    assert by[("jezero", "air_density")]["value"] == pytest.approx(expected_rho, rel=1e-6)
    assert expected_rho > float(air_density(0.0, ad["mars"], ad["universal_gas_constant_j_mol_k"]))  # lower ground, denser air
    lat = CFG.targets("mars")["jezero"].lat
    assert by[("jezero", "sun_max_elev")]["value"] == pytest.approx(float(I.sun_max_elev(lat, mars.obliquity_deg)[0]), abs=0.05)
    assert by[("jezero", "sun_max_elev")]["value"] == pytest.approx(90.0, abs=0.05)  # 18.4 N is inside the tropics of an obliquity 25.19 planet
    flat = float(I.insol_annual(lat, mars.obliquity_deg, mars.solar_irradiance_w_m2)[0])
    assert by[("jezero", "insol_annual")]["value"] == pytest.approx(flat, rel=0.01) and 150 < flat < 250
    for col in ("t_mean", "t_coldest"):
        r = by[("jezero", col)]
        assert r["unavailable"] and np.isnan(r["value"]) and r["note"].startswith("unavailable: no gridded")


def test_air_density_responds_to_elevation_with_the_shared_function(tmp_path):
    f = tmp_path / "mola.tif"
    T.write_mola(f, lambda lon, lat: np.full(lon.shape, 3000.0))
    high = [r for r in MA.mars_climate_rows(CFG, str(f)) if r["column"] == "air_density"][0]["value"]
    f2 = tmp_path / "mola2.tif"
    T.write_mola(f2, lambda lon, lat: np.full(lon.shape, -3000.0))
    low = [r for r in MA.mars_climate_rows(CFG, str(f2)) if r["column"] == "air_density"][0]["value"]
    assert low > high and low / high == pytest.approx(math.exp(6000.0 / CFG.features.air_density["mars"]["scale_height_m"]), rel=1e-6)


def test_no_physical_constant_is_in_the_new_code():
    src = (Path(MA.__file__)).read_text(encoding="utf-8")
    block = src[src.index("def mars_climate_rows"):]
    tree = ast.parse(block)
    numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    assert numbers <= {0, 1, 2, 1000.0, 500.0, 2.0}, numbers  # unit conversions and indexing only
