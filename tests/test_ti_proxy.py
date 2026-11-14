from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gsop import ti_proxy as T
from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_driest_window_wraps_around_the_year():
    pr = [50, 50, 50, 50, 50, 50, 50, 50, 50, 5, 5, 5]
    assert T.driest_window_start(pr, 3) == (9, 15.0)
    pr = [2, 3, 80, 80, 80, 80, 80, 80, 80, 80, 1, 2]  # Nov, Dec, Jan wrap
    start, total = T.driest_window_start(pr, 3)
    assert start == 10 and total == 1 + 2 + 2


def test_monsoon_tile_uses_its_dry_season_swing():
    pr = [5, 5, 5, 10, 40, 200, 400, 350, 200, 40, 10, 5]
    swing = [18, 19, 21, 22, 15, 9, 7, 7, 9, 14, 17, 18]  # large swing when dry, small when wet
    value, fallback = T.ti_proxy(pr, swing, swing)
    assert not fallback
    start, _ = T.driest_window_start(pr)
    window = [swing[(start + k) % 12] for k in range(3)]
    assert value == pytest.approx(np.mean(window))
    assert value > np.median(swing)  # dry-season swing is larger than the annual median here


def test_hot_desert_high_swing_humid_forest_low_swing():
    desert = T.ti_proxy([0] * 12, [28] * 12, [28] * 40)
    forest = T.ti_proxy([220, 200, 230, 250, 260, 240, 200, 190, 210, 240, 250, 230], [6] * 12, [6] * 40)
    assert desert == (28.0, False)  # no rain at all: dry season trivially defined
    assert forest[0] == 6.0 and forest[1] is True  # evenly wet: no dry season, fallback to the median
    assert desert[0] > forest[0]


def test_fallback_when_precip_missing_or_swing_missing():
    nan_pr = [np.nan] * 12
    assert T.ti_proxy(nan_pr, [10] * 12, [10, 12, 14]) == (12.0, True)
    pr = [0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    swing = [np.nan, 10, 10] + [5] * 9  # a dry month has no valid swing
    value, fb = T.ti_proxy(pr, swing, [5, 5, 10])
    assert fb is True and value == 5.0
    assert np.isnan(T.ti_proxy(pr, swing, [])[0])


def test_seasonality_threshold_is_the_config_value():
    pr = [30] * 9 + [10, 10, 10]  # driest 3 months = 30, average window = 3 * mean(pr) = 3*25 = 75 -> ratio 0.4
    assert T.dry_season_defined(pr, 3, ratio_max=0.5)
    assert not T.dry_season_defined(pr, 3, ratio_max=0.3)
    t = CFG.features.temperature
    assert (t.dry_months, t.dry_season_ratio_max) == (3, 0.5)
    with pytest.raises(ValueError):
        T.driest_window_start([1, 2, 3])


def test_earth_engine_builder_uses_the_same_sources_and_window():
    ee = MagicMock()
    G.ti_proxy_bands(ee, CFG)
    ee.ImageCollection.assert_any_call(G.MODIS_LST_ID)
    ee.ImageCollection.assert_any_call(G.TERRACLIMATE_ID)
    assert ee.Filter.calendarRange.call_count == 24  # 12 calendar months for precipitation and 12 for swing
