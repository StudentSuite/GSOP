from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gsop import gate_layers as GL
from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
LC = CFG.features.land_cover


def test_class_numbers_match_what_was_read_from_the_catalog_page():
    assert LC["water_ice_classes"] == [15, 17]
    assert LC["builtup_cropland_classes"] == [12, 13, 14]
    assert LC["ndvi_good_summary_qa"] == [0, 1]


def test_lake_and_farmland_show_up_in_the_fractions():
    # a Lonar-like 12x12 tile: a lake in one corner, farmland around it, scrub elsewhere
    tile = np.full((12, 12), 7.0)  # open shrubland
    tile[:3, :3] = 17  # crater lake
    tile[3:9, :] = 12  # cropland
    tile[9:11, 0:4] = 13  # a village
    assert GL.class_fraction(tile, LC["water_ice_classes"]) == pytest.approx(9 / 144)
    assert GL.class_fraction(tile, LC["builtup_cropland_classes"]) == pytest.approx((6 * 12 - 0 + 8) / 144)
    assert np.isnan(GL.class_fraction(np.full((3, 3), np.nan), [17]))


def test_median_ndvi_ignores_missing_cells():
    assert GL.median_ndvi([0.1, 0.5, np.nan, 0.3]) == pytest.approx(0.3)
    assert np.isnan(GL.median_ndvi([np.nan]))


def test_bands_are_named_as_the_issue_says_and_thresholds_are_not_applied():
    layers = CFG.features.gate_layers
    assert layers["water_ice_fraction"]["band"] == "gate_water_ice_frac"
    assert layers["ndvi_median"]["band"] == "gate_ndvi_med"
    assert layers["builtup_cropland_fraction"]["band"] == "gate_built_crop_frac"
    ee = MagicMock()
    G.gate_bands(ee, CFG)
    ee.ImageCollection.assert_any_call(G.MODIS_LANDCOVER_ID)
    ee.ImageCollection.assert_any_call(G.MODIS_NDVI_ID)
    ee.Reducer.median.assert_called()
    src = (Path(G.__file__)).read_text(encoding="utf-8")
    gate_src = src[src.index("def gate_bands"):]
    assert "cfg.gates" not in gate_src and "water_ice_fraction" not in gate_src  # reads no gate threshold
