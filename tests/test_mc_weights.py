from pathlib import Path

import numpy as np
import pytest

from gsop import montecarlo as MC

CFG_RANGES = None


def setup(lo=(0.1, 0.2, 0.05), hi=(0.2, 0.4, 0.1), mult=(1.0, 1.0, 1.0), seed=20261114):
    k = 2
    return MC.MCSetup(
        body="moon", target="t", profile="base", columns=("a", "b"), families=("f1", "f2", "f3"),
        family_cols=(np.array([0]), np.array([1]), np.array([0, 1])), target_values=np.zeros(k), sigma=np.ones(k),
        weight_lo=np.array(lo), weight_hi=np.array(hi), multipliers=np.array(mult), target_unc=np.zeros(k), earth_bias_sd=np.zeros(k), master_seed=seed)


def test_run_seed_is_child_i_of_spawn_for_any_n_and_streams_are_its_children():
    for n in (5, 50, 2000):
        children = np.random.SeedSequence(20261114).spawn(n)
        for i in (0, 1, n - 1):
            mine = MC.run_seed(20261114, i)
            assert mine.spawn_key == children[i].spawn_key and np.array_equal(mine.generate_state(4), children[i].generate_state(4))
            grand = children[i].spawn(2)
            for s in (0, 1):
                assert np.array_equal(MC.run_seed(20261114, i, s).generate_state(4), grand[s].generate_state(4))
    # the two streams of one run differ, and different runs differ
    a, b = MC.run_seed(1, 3, 0).generate_state(2), MC.run_seed(1, 3, 1).generate_state(2)
    assert not np.array_equal(a, b) and not np.array_equal(MC.run_seed(1, 3, 0).generate_state(2), MC.run_seed(1, 4, 0).generate_state(2))


def test_same_run_index_gives_the_same_weights_always_and_nothing_global_is_touched():
    s = setup()
    a = MC.sample_weights(s, 17)
    np.random.seed(0)
    _ = np.random.rand(5)  # touching the global generator must not matter
    b = MC.sample_weights(s, 17)
    assert np.array_equal(a, b)
    assert not np.array_equal(MC.sample_weights(s, 17), MC.sample_weights(s, 18))
    assert not np.array_equal(MC.sample_weights(s, 17), MC.sample_weights(setup(seed=1), 17))  # master seed matters


def test_10000_draws_stay_inside_each_range_before_renormalisation_and_sum_to_one_after():
    s = setup()
    raws, ws = [], []
    for i in range(10_000):
        w, raw = MC.sample_weights(s, i, return_raw=True)
        raws.append(raw)
        ws.append(w)
    raws, ws = np.array(raws), np.array(ws)
    assert (raws >= s.weight_lo).all() and (raws <= s.weight_hi).all()
    assert np.allclose(ws.sum(axis=1), 1.0, atol=1e-12) and (ws > 0).all()
    # uniform draws: the mean of each raw weight is near the middle of its range (loose check, 10,000 draws)
    assert np.allclose(raws.mean(axis=0), (s.weight_lo + s.weight_hi) / 2, rtol=0.02)


def test_multipliers_are_applied_after_the_first_renormalisation_and_then_renormalised_again():
    s = setup(mult=(2.0, 1.0, 0.5))
    w, raw = MC.sample_weights(s, 3, return_raw=True)
    first = raw / raw.sum()
    expected = first * np.array([2.0, 1.0, 0.5])
    expected = expected / expected.sum()
    assert np.allclose(w, expected) and w.sum() == pytest.approx(1.0)
    # Honest note: renormalising is scale-invariant, so normalise-multiply-normalise equals multiply-normalise. The documented order is kept
    # (it states what the weights mean at each step); the result is the same either way.
    assert np.allclose(w, (raw * np.array([2.0, 1.0, 0.5])) / (raw * np.array([2.0, 1.0, 0.5])).sum())


def test_real_ranges_from_config_are_used_for_the_moon_base_profile():
    from gsop.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config")
    fams = cfg.weights("moon")
    s = setup(lo=[f.range[0] for f in fams.values()], hi=[f.range[1] for f in fams.values()], mult=[1.0] * 8)
    ws = np.array([MC.sample_weights(s, i) for i in range(2000)])
    assert ws.shape == (2000, 8) and np.allclose(ws.sum(axis=1), 1.0)
    assert ws[:, 0].mean() > ws[:, 7].mean() * 5  # slope (range 0.21-0.39) far outweighs thermal_inertia (0.02-0.04)
