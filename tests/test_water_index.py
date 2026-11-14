from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gsop import water_index as W
from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_dryness_direction_and_bounds():
    # tropical wet: P well above PET -> 0; hyperarid: P ~ 0 -> 1; semi-arid in between
    d = W.dryness([2500, 5, 300], [1400, 2000, 900], ai_cap=1.0)
    assert d[0] == 0.0 and d[1] == pytest.approx(1 - 5 / 2000) and d[2] == pytest.approx(1 - 300 / 900)
    assert np.isnan(W.dryness(100, 0)) and np.isnan(W.dryness(np.nan, 500))


def test_cold_deserts_score_high_and_tropical_wet_low():
    cold_desert = W.water_index(0.9, W.dryness(150, 700))
    hot_desert = W.water_index(0.0, W.dryness(30, 2400))
    tropical_wet = W.water_index(0.0, W.dryness(2800, 1300))
    assert cold_desert > 0.8 and cold_desert > hot_desert > tropical_wet
    assert tropical_wet == 0.0


def test_missing_in_either_layer_is_nan_not_zero():
    assert np.isnan(W.water_index(np.nan, 0.5))
    assert np.isnan(W.water_index(0.5, np.nan))
    assert W.water_index(np.nan, 0.6, outside_domain="zero") == pytest.approx(0.3)  # explicit policy only
    assert np.isnan(W.water_index(0.2, np.nan, outside_domain="zero"))  # dryness missing is still NaN
    with pytest.raises(ValueError):
        W.water_index(0.1, 0.1, outside_domain="maybe")


def test_unavailable_while_no_permafrost_asset_is_configured():
    status, reason = G.water_index_status(CFG)
    assert status == "unavailable" and "Anay" in reason
    with pytest.raises(RuntimeError, match="water_index unavailable"):
        G.water_index_bands(MagicMock(), CFG)


def test_earth_engine_builder_when_a_permafrost_asset_is_given():
    from dataclasses import replace

    wi = replace(CFG.features.water_index, permafrost_asset="users/example/permafrost", permafrost_outside_domain="zero")
    from types import SimpleNamespace

    cfg = SimpleNamespace(features=replace(CFG.features, water_index=wi))
    ee = MagicMock()
    G.water_index_bands(ee, cfg)
    ee.Image.assert_any_call("users/example/permafrost")
    ee.ImageCollection.assert_any_call(G.TERRACLIMATE_ID)


def test_water_index_is_flagged_as_a_proxy_in_features_and_spec():
    assert CFG.features.columns["water_index"].proxy is True
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "cross-instrument proxy" in text
