import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import test_mc_weights as TW
from gsop import montecarlo as MC
from gsop import similarity as S
from gsop.config import load_config
from gsop.targets import TargetProfile

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def with_unc(target_unc, bias_sd):
    return replace(TW.setup(), target_unc=np.array(target_unc, dtype=float), earth_bias_sd=np.array(bias_sd, dtype=float))


def test_with_zero_uncertainty_the_jitter_is_the_identity():
    s = with_unc([0, 0], [0, 0])
    for i in range(50):
        assert np.array_equal(MC.sample_data_shift(s, i), np.zeros(2))
        w, t = MC.run_params(s, i)
        assert np.array_equal(t, s.target_values) and w.sum() == pytest.approx(1.0)


def test_target_jitter_and_earth_bias_are_independent_normals_with_the_stated_sd():
    s = with_unc([2.0, 0.0], [0.0, 3.0])
    shifts = np.array([MC.sample_data_shift(s, i) for i in range(20_000)])
    assert shifts[:, 0].std() == pytest.approx(2.0, rel=0.03) and abs(shifts[:, 0].mean()) < 0.06
    assert shifts[:, 1].std() == pytest.approx(3.0, rel=0.03)  # the Earth bias enters as an equal-size shift of the target
    both = MC.sample_data_shift(with_unc([2.0, 2.0], [3.0, 3.0]), 0)
    only_t = MC.sample_data_shift(with_unc([2.0, 2.0], [0.0, 0.0]), 0)
    only_b = MC.sample_data_shift(with_unc([0.0, 0.0], [3.0, 3.0]), 0)
    assert np.allclose(both, only_t + only_b)  # same two normals each time: terms are additive and the stream is stable


def test_the_shift_equals_moving_the_earth_values_the_other_way():
    """(x + b - t') / sigma with t' = t + e equals (x - (t + e - b)) / sigma: one target shift covers both terms."""
    rng = np.random.default_rng(0)
    x, t, sigma, e, b = rng.normal(size=5), 1.0, 2.0, 0.3, -0.4
    direct = np.exp(-0.5 * (((x + b) - (t + e)) / sigma) ** 2)
    shifted = np.exp(-0.5 * ((x - (t + e - b)) / sigma) ** 2)
    assert np.allclose(direct, shifted)


def test_weights_and_data_use_different_streams():
    s = with_unc([1.0, 1.0], [1.0, 1.0])
    w1 = MC.sample_weights(s, 7)
    # changing the data uncertainty must not change the weights of the same run, and vice versa
    s2 = with_unc([5.0, 5.0], [9.0, 9.0])
    assert np.array_equal(w1, MC.sample_weights(s2, 7))
    s3 = replace(s, weight_lo=s.weight_lo * 0.5)
    assert np.array_equal(MC.sample_data_shift(s, 7), MC.sample_data_shift(s3, 7))


def test_reproducible_for_any_run_index_in_any_order():
    s = with_unc([1.0, 2.0], [0.5, 0.5])
    first = {i: MC.run_params(s, i) for i in (5, 0, 99)}
    again = {i: MC.run_params(s, i) for i in (99, 5, 0)}
    for i in first:
        assert np.array_equal(first[i][0], again[i][0]) and np.array_equal(first[i][1], again[i][1])


def test_todo_uncertainty_means_no_jitter_logged_once(caplog):
    MC._JITTER_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="gsop.montecarlo"):
        assert MC.target_uncertainty_sd(CFG, "mars", "slope_med") == 0.0
        assert MC.target_uncertainty_sd(CFG, "mars", "slope_med") == 0.0
        assert MC.earth_bias_sd(CFG, "relief") == 0.0
        assert MC.earth_bias_sd(CFG, "relief") == 0.0
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 2 and "slope_med" in msgs[0] and "relief" in msgs[1]
    # sourced target-side values are used: Moon slope 2.5 deg (Barker 2021), Mars relief 3 m (MOLA)
    assert MC.target_uncertainty_sd(CFG, "moon", "slope_med") == 2.5 and MC.target_uncertainty_sd(CFG, "mars", "relief") == 3.0
    assert all(MC.earth_bias_sd(CFG, c) == 0.0 for c in CFG.features.columns)  # none of the Earth biases has a source today


def test_make_mc_setup_from_a_score_setup():
    prof = TargetProfile("t", {"slope_med": 5.0, "slope_p90": 12.0, "relief": 100.0}, {"slope_med": 0.5, "slope_p90": 1.0, "relief": 10.0}, frozenset())
    ss = S.make_setup(CFG, "moon", prof, policy="drop_and_renormalise")
    with pytest.raises(ValueError, match="no sigma"):
        MC.make_mc_setup(CFG, ss, "base")
    ss = ss.with_sigma(np.ones(len(ss.columns)))
    m = MC.make_mc_setup(CFG, ss, "landing_mobility")
    assert m.families == ("slope", "relief") and m.master_seed == 20261114
    assert np.allclose(m.weight_lo, [0.21, 0.07]) and np.allclose(m.weight_hi, [0.39, 0.18]) and np.allclose(m.multipliers, [2.0, 2.0])  # landing: slope and relief x2
    assert np.array_equal(m.target_unc, [2.5, 2.5, 0.5]) and np.array_equal(m.earth_bias_sd, np.zeros(3))
    assert MC.make_mc_setup(CFG, ss, "base", master_seed=5).master_seed == 5
