import ast
from pathlib import Path

import numpy as np
import pytest

from gsop import gates as G
from gsop.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def block(**over):
    base = {"water_ice_fraction": [0.0], "ndvi_median": [0.0], "builtup_cropland_fraction": [0.0], "slope_med": [1.0], "slope_p90": [2.0], "relief": [10.0]}
    base.update({k: np.atleast_1d(v) for k, v in over.items()})
    return {k: np.asarray(v, dtype="float64") for k, v in base.items()}


def test_no_gate_threshold_is_hard_coded_in_src():
    for name in ("gates.py",):
        tree = ast.parse((ROOT / "src" / "gsop" / name).read_text(encoding="utf-8"))
        numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
        assert numbers <= {0, 0.0, 1, 1.0, 2, 4, 8, 16}, numbers  # 1,2,4,8,16 are the bit values; no 0.5, 0.2, 30, ...


def test_a_clean_tile_passes_everything_and_reasons_use_the_config_codes():
    r = G.evaluate_gates(CFG, "moon", block(), ["slope_med", "relief"])
    assert r.reasons[0] == 0 and r.passed(True)[0] and G.decode(0, CFG) == []
    assert G.reason_codes(CFG) == {"water_ice": "WATER_OR_ICE", "vegetation": "NDVI_HIGH", "builtup_cropland": "BUILTUP_OR_CROPLAND",
                                   "landing_mobility_slope": "LANDING_SLOPE", "missing_data": "MISSING_DATA"}


def test_thresholds_come_from_gates_yaml():
    t = CFG.gates.thresholds
    just_over = block(water_ice_fraction=t["water_ice_fraction"] + 1e-6, ndvi_median=t["ndvi_median"] + 1e-6, builtup_cropland_fraction=t["builtup_cropland_fraction"] + 1e-6)
    r = G.evaluate_gates(CFG, "moon", just_over, ["slope_med"])
    assert G.decode(int(r.reasons[0]), CFG) == ["WATER_OR_ICE", "NDVI_HIGH", "BUILTUP_OR_CROPLAND"]


def test_chunk_safety_same_result_for_any_split():
    rng = np.random.default_rng(0)
    n = 1000
    cols = {"water_ice_fraction": rng.uniform(0, 1, n), "ndvi_median": rng.uniform(-0.1, 0.6, n), "builtup_cropland_fraction": rng.uniform(0, 0.5, n),
            "slope_med": rng.uniform(0, 20, n), "slope_p90": rng.uniform(0, 40, n)}
    cols["slope_med"][::50] = np.nan
    whole = G.evaluate_gates(CFG, "moon", cols, ["slope_med"])
    parts = [G.evaluate_gates(CFG, "moon", {k: v[i : i + 137] for k, v in cols.items()}, ["slope_med"]) for i in range(0, n, 137)]
    assert np.array_equal(whole.reasons, np.concatenate([p.reasons for p in parts])) and np.array_equal(whole.near, np.concatenate([p.near for p in parts]))


def test_counts_per_reason_and_totals():
    cols = {"water_ice_fraction": np.array([0.0, 0.9, 0.9, 0.0]), "ndvi_median": np.array([0.0, 0.0, 0.5, 0.0]),
            "builtup_cropland_fraction": np.zeros(4), "slope_med": np.array([1.0, 1.0, 1.0, np.nan]), "slope_p90": np.ones(4)}
    r = G.evaluate_gates(CFG, "moon", cols, ["slope_med"])
    c = G.counts_by_reason(r, CFG)
    assert c == {"WATER_OR_ICE": 2, "NDVI_HIGH": 1, "BUILTUP_OR_CROPLAND": 0, "MISSING_DATA": 1, "PASSED": 1, "REJECTED": 3}
    assert G.counts_by_reason(r, CFG, applies_landing_gate=True)["LANDING_SLOPE"] == 0
