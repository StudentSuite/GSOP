import inspect

import numpy as np

from gsop import operators as O


def test_public_api_is_documented_and_stable():
    for name in ("baseline_factor", "effective_baseline_m", "block_mean", "horn_slope_deg", "plane_rms", "tile_features"):
        fn = getattr(O, name)
        assert inspect.getdoc(fn), name
    sig = inspect.signature(O.tile_features)
    assert list(sig.parameters) == ["dem", "cell_m", "tile_m", "baseline_m", "rough_baselines_m", "stride_m", "max_nan_fraction"]


def test_constant_dem_runs_end_to_end():
    out = O.tile_features(np.full((26, 26), 100.0), cell_m=500.0, tile_m=6000.0, rough_baselines_m=(500.0, 1000.0))
    assert out["slope_med"].shape == (2, 2) or out["slope_med"].size > 0
