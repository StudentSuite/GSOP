import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop.earth.region_check import check_raster, format_report

ROOT = Path(__file__).resolve().parents[1]


def write_raster(path, bands: dict, crs="EPSG:6933", pixel=6000.0, x0=6000.0 * 1000, y0=6000.0 * 500, nodata=-9999.0):
    data = list(bands.values())
    h, w = data[0].shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=len(data), dtype="float32", crs=crs,
                       transform=from_origin(x0, y0, pixel, pixel), nodata=nodata) as ds:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            ds.write(np.where(np.isnan(arr), nodata, arr).astype("float32"), i)
            ds.set_band_description(i, name)


def good_bands():
    rng = np.random.default_rng(0)
    a = lambda lo, hi: rng.uniform(lo, hi, (10, 12))  # noqa: E731
    return {"slope_med": a(0, 20), "relief": a(0, 800), "t_mean": a(270, 305), "gate_ndvi_med": a(-0.1, 0.7)}


def test_good_raster_passes(tmp_path):
    p = tmp_path / "ok.tif"
    write_raster(p, good_bands())
    rep = check_raster(p, ["slope_med", "relief", "t_mean", "gate_ndvi_med"])
    assert rep.ok and rep.crs == "EPSG:6933" and rep.pixel_size == (6000.0, 6000.0) and rep.shape == (10, 12)
    assert all(b.valid_share == 1.0 for b in rep.bands)
    assert "RESULT: ok" in format_report(rep)


def test_problems_are_found(tmp_path):
    bands = good_bands()
    bands["slope_med"] = bands["slope_med"] + 95  # unit error: all above 90 degrees
    bands["relief"] = np.full((10, 12), np.nan)  # all nodata
    p = tmp_path / "bad.tif"
    write_raster(p, bands, crs="EPSG:6933", pixel=5000.0, x0=5001.0)
    rep = check_raster(p, ["slope_med", "relief", "t_mean", "gate_ndvi_med", "silica_index"])
    text = " ".join(rep.problems + [x for b in rep.bands for x in b.problems])
    assert not rep.ok
    for needle in ("pixel size", "not a multiple", "missing bands: silica_index", "outside the plausible range", "no valid pixels"):
        assert needle in text, needle


def test_wrong_crs_and_unexpected_band(tmp_path):
    p = tmp_path / "crs.tif"
    write_raster(p, good_bands(), crs="EPSG:3857", pixel=6000.0)
    rep = check_raster(p, ["slope_med", "relief", "t_mean"])
    assert any("CRS is EPSG:3857" in x for x in rep.problems) and any("unexpected bands: gate_ndvi_med" in x for x in rep.problems)


def test_ee_probe_check_raster_command(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("ee_probe", ROOT / "scripts" / "ee_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from gsop.config import load_config
    from gsop.earth import gee_export

    cfg = load_config(ROOT / "config")
    bands, _ = gee_export.planned_bands(cfg)
    full = {}
    rng = np.random.default_rng(1)
    from gsop.earth.sanity import SANITY_RANGES

    for b in bands:
        lo, hi, _ = SANITY_RANGES[b]
        full[b] = rng.uniform(lo, min(hi, lo + 1.0), (4, 5))
    p = tmp_path / "all.tif"
    write_raster(p, full)
    assert mod.main(["--check-raster", str(p)]) == 0
    assert "RESULT: ok" in capsys.readouterr().out
