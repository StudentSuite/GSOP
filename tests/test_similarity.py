"""Similarity behaves sanely on trivial inputs (issue C09).

HAND CALCULATION used by test_three_tile_hand_calculation. Three columns, sigma = 1 for all, target = (0, 0, 0). Family 1 = columns 1 and 2
(weight 0.7), family 2 = column 3 (weight 0.3). With e^-0.5 = 0.6065306597 and e^-2 = 0.1353352832:

  tile A x = (0, 0, 0): s = (1, 1, 1);                       S1 = 1,                         S2 = 1               -> 100 * (0.7 + 0.3)            = 100
  tile B x = (1, 0, 0): s = (0.6065306597, 1, 1);            S1 = (0.6065306597 + 1) / 2 = 0.80326533, S2 = 1   -> 100 * (0.562285731 + 0.3)  = 86.2285731
  tile C x = (2, 2, 1): s = (0.1353352832, 0.1353352832, 0.6065306597);
                                                              S1 = 0.1353352832,              S2 = 0.6065306597    -> 100 * (0.0947346982 + 0.1819591979) = 27.66938961
"""

from pathlib import Path

import numpy as np
import pytest

import synth
from gsop import similarity as S
from gsop import weights as W
from gsop.config import load_config
from gsop.targets import TargetProfile

CFG = load_config(Path(__file__).resolve().parents[1] / "config")

TARGET = np.zeros(3)
SIGMA = np.ones(3)
FAMS = (np.array([0, 1]), np.array([2]))
WEIGHTS = np.array([0.7, 0.3])


def score(x, sigma=SIGMA, target=TARGET, weights=WEIGHTS):
    return S.score_tiles(np.asarray(x, dtype=np.float32), target, sigma, FAMS, weights, return_family=True)


def test_exact_match_gives_one_per_column_and_100_overall():
    r = score([[0, 0, 0]])
    assert np.array_equal(S.column_similarity(np.zeros((1, 3), np.float32), TARGET, SIGMA), np.ones((1, 3), np.float32))
    assert r.score[0] == pytest.approx(100.0, abs=1e-4) and np.allclose(r.family, 1.0)


def test_far_values_give_about_zero_without_nan_or_underflow_ties_at_the_top():
    r = score([[40, 40, 40], [1e4, -1e4, 1e8], [1e30, 1e30, 1e30], [0, 0, 0.5]])
    assert np.isfinite(r.score).all() and (r.score[:3] < 1e-20).all() and r.score[3] > 90
    # on the synthetic grid: all finite, and the best scores are distinct (the far tail may tie, the top never does)
    g = synth.make_grid(20_000, seed=3, body="moon", cfg=CFG)
    t = synth.target_profile("moon", CFG)
    prof = TargetProfile("synthetic", dict(zip(t.column, t.value)), dict(zip(t.column, t.spread)), frozenset())
    setup = S.make_setup(CFG, "moon", prof)
    sigma = np.maximum(setup.target_spread, [synth.BACKGROUND[c][1] * 0.25 for c in setup.columns])
    w = S.renormalise(W.weight_vector(W.derive_weights(CFG, "moon")), setup.families)
    sc = S.score_tiles(g.df[list(setup.columns)].to_numpy("float32"), setup.target_values, sigma, setup.family_cols, w).score
    ok = g.df[list(setup.columns)].notna().all(axis=1).to_numpy()
    assert np.isfinite(sc[ok]).all() and not np.isfinite(sc[~ok]).any()
    # the analog and the three gate traps are identical to the target by construction, so they legitimately tie at the top;
    # every other tile must get its own score
    planted_copies = {g.truth["analog"], *g.truth["gate_trap"].values()}
    other = ok & ~g.df.tile_id.isin(planted_copies).to_numpy()
    top = np.sort(sc[other])[::-1][:20]
    assert len(np.unique(top)) == 20 and sc[g.df.tile_id.to_numpy() == g.truth["analog"]][0] >= top[0]


def test_sigma_floor_gives_a_slightly_off_tile_a_positive_score_where_target_spread_is_zero():
    iqr_values = np.arange(0.0, 101.0)  # IQR 50 -> sigma floor 12.5
    sigma_floor = S.compute_sigma("c", target_spread=0.0, meas_unc=0.0, earth_values_gated=iqr_values).sigma
    off = [[1.0, 1.0, 1.0]]
    with_floor = S.score_tiles(np.asarray(off, np.float32), TARGET, np.full(3, sigma_floor), FAMS, WEIGHTS).score[0]
    no_floor = S.score_tiles(np.asarray(off, np.float32), TARGET, np.full(3, S.SIGMA_MIN), FAMS, WEIGHTS).score[0]
    assert sigma_floor == 12.5 and with_floor > 99.0 and no_floor < 1e-20  # without a floor, one unit off scores nothing


def test_three_tile_hand_calculation():
    r = score([[0, 0, 0], [1, 0, 0], [2, 2, 1]])
    assert r.score == pytest.approx([100.0, 86.2285731, 27.66938961], abs=1e-3)
    assert r.family[1] == pytest.approx([0.80326533, 1.0], abs=1e-6)
    assert r.family[2] == pytest.approx([0.1353352832, 0.6065306597], abs=1e-6)


def test_weights_that_change_between_calls_change_the_result_as_expected():
    r1 = score([[1, 0, 0]], weights=np.array([1.0, 0.0])).score[0]
    r2 = score([[1, 0, 0]], weights=np.array([0.0, 1.0])).score[0]
    assert r1 == pytest.approx(80.326533, abs=1e-3) and r2 == pytest.approx(100.0, abs=1e-4)


def test_missing_columns_propagate_as_missing_not_zero():
    r = score([[0, np.nan, 0], [0, 0, np.nan], [0, 0, 0]])
    assert np.isnan(r.score[0]) and np.isnan(r.score[1]) and r.score[2] == pytest.approx(100, abs=1e-4)
    assert np.isnan(r.family[0, 0]) and r.family[0, 1] == pytest.approx(1.0)  # only the family with the missing column is missing
    assert np.isnan(r.family[1, 1]) and r.family[1, 0] == pytest.approx(1.0)
    # a missing value is never silently turned into a zero similarity
    assert not (r.score[:2] == 0).any()


def test_scores_stay_between_0_and_100_for_random_input():
    x = np.random.default_rng(0).normal(size=(5000, 3)) * 3
    sc = score(x).score
    assert (sc >= 0).all() and (sc <= 100.0001).all()
