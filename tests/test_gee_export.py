import math
from pathlib import Path
from unittest.mock import MagicMock

from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_operator_settings_loaded_from_config():
    ops = CFG.features.operators
    assert (ops.grid_crs, ops.tile_m, ops.baseline_m, ops.slope_method) == ("EPSG:6933", 6000, 500, "horn")
    assert ops.nasadem_lat_range == (-56, 60)
    assert G.tile_cells(CFG, ops.baseline_m) == 144


def test_lonlat_box_is_about_the_right_size():
    w, s, e, n = G.lonlat_box_around(19.98, 76.51, 600)
    assert abs((n - s) * 111.195 - 600) < 1
    assert abs((e - w) * 111.195 * math.cos(math.radians(19.98)) - 600) < 1
    assert w < 76.51 < e and s < 19.98 < n


def test_horn_kernels_match_the_spec():
    assert sum(map(sum, G.HORN_KX)) == 0 and sum(map(sum, G.HORN_KY)) == 0
    assert [r[2] for r in G.HORN_KX] == [1, 2, 1] and [c for c in G.HORN_KY[2]] == [1, 2, 1]


def test_slope_bands_builds_the_documented_chain():
    ee = MagicMock()
    out = G.slope_bands(ee, CFG)
    # the DEM sources and the seam rule
    ee.Image.assert_any_call(G.NASADEM_ID)
    ee.ImageCollection.assert_any_call(G.COPERNICUS_ID)
    # 500 m resampling by mean, tile reduction by percentiles 50 and 90
    ee.Reducer.mean.assert_called()
    ee.Reducer.percentile.assert_called_with([50, 90])
    assert out is not None
    # Horn kernel is 3x3, un-normalised
    args = ee.Kernel.fixed.call_args_list[0][0]
    assert args[:2] == (3, 3) and args[-1] is False


def test_spec_documents_the_slope_operator():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    for needle in ("Horn (1981)", "dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * dx)", "hard cut",
                   "reduceResolution", "12 x 12", "max_nan_fraction"):
        assert needle in text, needle


def _closed_form_rms(block):
    """The closed-form plane-removed RMS used in the Earth Engine expression, in numpy."""
    import numpy as np

    n = block.shape[0]
    idx = np.arange(n) - (n - 1) / 2.0
    y, x = np.meshgrid(idx, idx, indexing="ij")
    ex2 = (n * n - 1) / 12.0
    resid = (np.mean(block**2) - np.mean(block) ** 2 - np.mean(x * block) ** 2 / ex2 - np.mean(y * block) ** 2 / ex2)
    return float(np.sqrt(max(resid, 0.0)))


def _lstsq_rms(block):
    import numpy as np

    n = block.shape[0]
    idx = np.arange(n) - (n - 1) / 2.0
    y, x = np.meshgrid(idx, idx, indexing="ij")
    a = np.column_stack([np.ones(block.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(a, block.ravel(), rcond=None)
    return float(np.sqrt(np.mean((block.ravel() - a @ coef) ** 2)))


def test_closed_form_plane_removal_equals_least_squares():
    import numpy as np

    rng = np.random.default_rng(0)
    for n in (6, 12, 60):
        block = rng.normal(size=(n, n)) * 10 + np.add.outer(np.arange(n) * 3.0, np.arange(n) * -2.0)
        assert abs(_closed_form_rms(block) - _lstsq_rms(block)) < 1e-6 * max(1.0, _lstsq_rms(block))
    plane = np.add.outer(np.arange(12) * 3.0, np.arange(12) * -2.0)
    assert _closed_form_rms(plane) < 1e-6


def test_roughness_baselines_come_from_config_not_code():
    assert G.roughness_baseline_m(CFG, "rough_a") == 100
    assert G.roughness_baseline_m(CFG, "rough_b") == 1000


def test_relief_and_rough_build_with_the_documented_reducers():
    ee = MagicMock()
    G.relief_band(ee, CFG)
    ee.Reducer.max.assert_called()
    ee.Reducer.min.assert_called()
    ee = MagicMock()
    G.rough_band(ee, CFG, "rough_b")
    ee.Image.pixelCoordinates.assert_called()
    ee.Reducer.mean.assert_called()
    # 1 km baseline: 6 cells per tile side; maxPixels = 6*6 + 16


def test_lst_qa_rule_and_config_window():
    assert G.lst_qa_is_good(0) and G.lst_qa_is_good(0b11111100)  # only bits 0-1 matter
    assert not G.lst_qa_is_good(1) and not G.lst_qa_is_good(2) and not G.lst_qa_is_good(3)
    t = CFG.features.temperature
    assert (t.start_year, t.end_year, t.lst_scale_kelvin) == (2015, 2024, 0.02)
    assert t.min_valid_obs_total > 0 and t.min_valid_obs_per_month > 0


def test_temperature_bands_use_the_documented_collection_and_window():
    ee = MagicMock()
    G.temperature_bands(ee, CFG)
    ee.ImageCollection.assert_any_call(G.MODIS_LST_ID)
    ee.ImageCollection.return_value.filterDate.assert_called_with("2015-01-01", "2025-01-01")
    assert ee.Filter.calendarRange.call_count == 12  # one per calendar month
