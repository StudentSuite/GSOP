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
