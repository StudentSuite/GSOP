from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gsop import dust
from gsop.config import ConfigError, load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_normalisation_uses_data_percentiles_not_an_invented_cap():
    raw = np.linspace(0.0, 1.0, 1001)
    lo, hi = dust.percentile_limits(raw)
    assert lo == pytest.approx(0.01) and hi == pytest.approx(0.99)
    out = dust.normalise_dust(raw, lo, hi)
    assert out.min() == 0.0 and out.max() == 1.0 and np.all(np.diff(out) >= 0)
    assert np.isnan(dust.normalise_dust([np.nan], lo, hi)[0])
    with pytest.raises(ValueError):
        dust.percentile_limits([np.nan])
    with pytest.raises(ValueError):
        dust.percentile_limits([2.0, 2.0, 2.0])


def test_unavailable_dust_leaves_ti_proxy_alone_in_the_mars_family():
    assert CFG.family_columns("mars", "thermal_inertia") == ("ti_proxy", "dust_index")
    assert CFG.family_columns("mars", "thermal_inertia", unavailable=["dust_index"]) == ("ti_proxy",)
    with pytest.raises(ConfigError, match="every column is unavailable"):
        CFG.family_columns("mars", "composition", unavailable=["silica_index"])
    with pytest.raises(ConfigError, match="unknown family"):
        CFG.family_columns("moon", "elevation_density")


def test_merra_builder_uses_dust_extinction_band_and_noon_filter():
    ee = MagicMock()
    G.dust_bands(ee, CFG)
    ee.ImageCollection.assert_any_call(G.MERRA2_AER_ID)
    ee.Filter.calendarRange.assert_called_with(12, 12, "hour")
    ee.ImageCollection.return_value.filterDate.return_value.filter.return_value.select.assert_called_with("DUEXTTAU")


def test_spec_states_the_dust_status():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "### Earth `dust_index`" in text and "uses `ti_proxy` **alone**" in text
