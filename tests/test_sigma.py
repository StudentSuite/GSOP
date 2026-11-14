import json
import logging
from pathlib import Path

import numpy as np
import pytest

from gsop import similarity as S
from gsop.config import load_config
from gsop.targets import TargetProfile

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
IQR_VALUES = np.arange(0.0, 101.0)  # percentiles 25 and 75 are 25 and 75: IQR = 50, so 0.25 * IQR = 12.5


def test_each_of_the_three_terms_can_win():
    a = S.compute_sigma("c", target_spread=30.0, meas_unc=2.0, earth_values_gated=IQR_VALUES)
    b = S.compute_sigma("c", target_spread=1.0, meas_unc=20.0, earth_values_gated=IQR_VALUES)
    c = S.compute_sigma("c", target_spread=1.0, meas_unc=2.0, earth_values_gated=IQR_VALUES)
    assert (a.winner, a.sigma) == ("target_spread", 30.0)
    assert (b.winner, b.sigma) == ("measurement_uncertainty", 20.0)
    assert (c.winner, c.sigma) == ("iqr", 12.5) and c.iqr_term == pytest.approx(12.5)


def test_iqr_is_taken_over_finite_gated_values_only():
    vals = np.concatenate([IQR_VALUES, [np.nan, np.inf, -np.inf]])
    assert S.compute_sigma("c", 0, 0, vals).iqr_term == pytest.approx(12.5)
    assert S.compute_sigma("c", 0, 0, [np.nan, np.nan]).iqr_term == 0.0 and S.compute_sigma("c", 0, 0, []).sigma == S.SIGMA_MIN


def test_the_multiplier_is_an_argument_taken_from_config():
    assert CFG.run.sigma_iqr_multiplier == 0.25
    assert S.compute_sigma("c", 0, 0, IQR_VALUES, iqr_multiplier=0.5).iqr_term == pytest.approx(25.0)


def test_nan_target_spread_counts_as_zero_and_all_zero_uses_the_guard(caplog):
    with caplog.at_level(logging.WARNING, logger="gsop.similarity"):
        t = S.compute_sigma("flat", float("nan"), 0.0, np.full(10, 5.0))
    assert t.sigma == S.SIGMA_MIN and t.target_spread == 0.0 and any("all three terms are 0" in r.message for r in caplog.records)


def test_measurement_uncertainty_uses_the_larger_sourced_side_and_todo_is_zero_with_one_warning(caplog):
    S._WARNED.clear()
    # Moon slope_med: target side sourced 2.5 deg (Barker 2021), Earth TODO -> 2.5
    assert S.measurement_uncertainty(CFG, "moon", "slope_med") == 2.5
    # Earth t_mean sourced 3 K, Moon TODO -> 3
    assert S.measurement_uncertainty(CFG, "moon", "t_mean") == 3.0
    # Mars relief: target side sourced 3 m
    assert S.measurement_uncertainty(CFG, "mars", "relief") == 3.0
    with caplog.at_level(logging.WARNING, logger="gsop.similarity"):
        assert S.measurement_uncertainty(CFG, "mars", "slope_med") == 0.0  # Mars TODO and Earth TODO
        assert S.measurement_uncertainty(CFG, "mars", "slope_med") == 0.0
        assert S.measurement_uncertainty(CFG, "mars", "water_index") == 0.0
    msgs = [r.message for r in caplog.records if "no sourced measurement uncertainty" in r.message]
    assert len(msgs) == 2 and "slope_med" in msgs[0] and "water_index" in msgs[1]  # once per column
    from gsop.config import ConfigError

    with pytest.raises(ConfigError):
        S.measurement_uncertainty(CFG, "moon", "nonsense")


def test_table_vector_and_json(tmp_path):
    prof = TargetProfile("t", {"slope_med": 5.0, "slope_p90": 12.0, "relief": 100.0}, {"slope_med": 0.5, "slope_p90": 1.0, "relief": 10.0}, frozenset())
    setup = S.make_setup(CFG, "moon", prof, policy="drop_and_renormalise")
    earth = {"slope_med": IQR_VALUES, "slope_p90": IQR_VALUES * 2, "relief": IQR_VALUES * 0.1}
    table = S.build_sigma_table(CFG, setup, earth)
    assert set(table) == {"slope_med", "slope_p90", "relief"}
    # hand calculation: slope_med: spread 0.5, unc 2.5, 0.25 * IQR(0..100) = 12.5  -> iqr wins, 12.5
    assert (table["slope_med"].winner, table["slope_med"].sigma) == ("iqr", 12.5)
    # slope_p90: IQR of 0..200 is 100 -> 25; spread 1; unc 2.5 -> iqr wins, 25
    assert (table["slope_p90"].winner, table["slope_p90"].sigma) == ("iqr", 25.0)
    # relief: spread 10; unc 0.5 (Barker height error); IQR of 0..10 is 5 -> 1.25 -> spread wins, 10
    assert (table["relief"].winner, table["relief"].sigma) == ("target_spread", 10.0)
    assert np.array_equal(S.sigma_vector(setup, table), [table[c].sigma for c in setup.columns])
    path = S.write_sigma_table(tmp_path / "sigma_table.json", {"t": table}, CFG.run.sigma_iqr_multiplier)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["iqr_multiplier"] == 0.25 and data["targets"]["t"]["relief"]["winner"] == "target_spread"
    assert set(data["targets"]["t"]["relief"]) == {"column", "sigma", "target_spread", "measurement_uncertainty", "iqr_term", "winner"}
