from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gsop import silica
from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_quartz_index_formula_and_direction():
    # quartz-like spectrum: low at bands 10 and 12, high at 11  -> high QI
    quartz = silica.quartz_index(0.85, 0.95, 0.88)
    # basalt-like (flat-ish spectrum, no dip) -> QI near 1
    basalt = silica.quartz_index(0.95, 0.95, 0.95)
    # clay/vegetated terrain: emissivity increasing smoothly -> QI below 1
    veg = silica.quartz_index(0.96, 0.97, 0.98)
    assert quartz == pytest.approx(0.95**2 / (0.85 * 0.88))
    assert quartz > veg and quartz > basalt
    assert basalt == pytest.approx(1.0)


def test_missing_or_nonpositive_emissivity_is_nan():
    out = silica.quartz_index([0.9, 0.0, np.nan], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1]) and np.isnan(out[2])


def test_earth_engine_builder_uses_aster_ged_and_masks_water():
    ee = MagicMock()
    G.silica_bands(ee, CFG)
    ee.Image.assert_any_call(G.ASTER_GED_ID)
    selected = [c.args[0] for c in ee.Image.return_value.select.call_args_list]
    assert {"emissivity_band10", "emissivity_band11", "emissivity_band12", "land_water_map"} <= set(selected)
    assert CFG.features.silica_index.aster_emissivity_scale == 0.001


def test_spec_names_moon_and_mars_equivalents_with_the_same_direction():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "Diviner" in text and "TES" in text
    assert "higher = more silica-rich" in text.lower()
