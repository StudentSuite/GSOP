import numpy as np
import pytest

import test_mc_weights as TW
from gsop import montecarlo as MC


def small_case(n=1000, runs=40, seed=0, unc=(1.0, 1.0)):
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    s = replace(TW.setup(), target_unc=np.array(unc), sigma=np.array([1.0, 1.5]))
    return x, s, runs


def direct(x, s, runs):
    return np.stack([MC.score_run(x, s, i) for i in range(runs)])  # (runs, n): fine for 1,000 tiles, the thing the engine must never do for 4 million


def test_mean_variance_min_max_equal_a_direct_computation_to_1e_5():
    x, s, runs = small_case()
    stats = MC.run_stats(x, s, 0, runs)
    d = direct(x, s, runs).astype(np.float64)
    assert stats.count == runs
    assert np.allclose(stats.mean, d.mean(axis=0), atol=1e-5) and np.allclose(stats.variance, d.var(axis=0), atol=1e-5)
    assert np.array_equal(stats.minimum, d.min(axis=0).astype(np.float32)) and np.array_equal(stats.maximum, d.max(axis=0).astype(np.float32))


def test_merging_blocks_equals_one_pass():
    x, s, runs = small_case(runs=60)
    whole = MC.run_stats(x, s, 0, runs)
    merged = MC.OnlineStats(len(x))
    for a, b in ((0, 7), (7, 33), (33, 60)):
        merged.merge(MC.run_stats(x, s, a, b))
    assert merged.count == runs
    assert np.allclose(merged.mean, whole.mean, atol=1e-9) and np.allclose(merged.m2, whole.m2, rtol=1e-9, atol=1e-9)
    assert np.array_equal(merged.minimum, whole.minimum) and np.array_equal(merged.maximum, whole.maximum)
    empty = MC.OnlineStats(len(x))
    empty.merge(whole)
    assert np.array_equal(empty.mean, whole.mean)  # merging into an empty set copies
    whole.merge(MC.OnlineStats(len(x)))  # merging an empty set changes nothing
    assert whole.count == runs


def test_save_and_load_round_trip(tmp_path):
    x, s, runs = small_case(runs=10)
    st = MC.run_stats(x, s, 0, runs)
    st.save(tmp_path / "s.npz")
    back = MC.OnlineStats.load(tmp_path / "s.npz")
    assert back.count == st.count and np.array_equal(back.mean, st.mean) and np.array_equal(back.m2, st.m2) and np.array_equal(back.minimum, st.minimum)


def test_a_run_depends_only_on_its_index_not_on_what_ran_before():
    x, s, runs = small_case()
    a = MC.score_run(x, s, 17)
    for i in range(5):
        MC.score_run(x, s, i)
    assert np.array_equal(a, MC.score_run(x, s, 17)) and np.array_equal(MC.score_run(x, s, 17, chunk_rows=33), a)  # chunking never changes a score


def test_jitter_actually_changes_the_scores_between_runs_and_not_with_zero_uncertainty():
    x, s, _ = small_case()
    assert not np.array_equal(MC.score_run(x, s, 1), MC.score_run(x, s, 2))
    from dataclasses import replace

    s0 = replace(s, target_unc=np.zeros(2), earth_bias_sd=np.zeros(2), weight_hi=s.weight_lo)  # no jitter and no weight freedom
    assert np.array_equal(MC.score_run(x, s0, 1), MC.score_run(x, s0, 2))


def test_the_debug_guard_refuses_an_n_by_runs_matrix_and_reports_the_size():
    g = MC.AllocGuard(4_000_000, 2000, enabled=True)
    g.check("fine: one score vector", (4_000_000,))
    g.check("fine: one chunk", (262_144, 14))
    with pytest.raises(AssertionError, match=r"8,000,000,000 elements.*looks like an N x runs matrix"):
        g.check("scores", (4_000_000, 2000))
    MC.AllocGuard(4_000_000, 2000, enabled=False).check("not in debug mode", (4_000_000, 2000))  # silent when debug is off
    x, s, runs = small_case(n=200, runs=3)
    MC.run_stats(x, s, 0, runs, debug=True)  # the real code path never trips it


def test_debug_flag_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("GSOP_MC_DEBUG", raising=False)
    assert not MC.debug_enabled()
    monkeypatch.setenv("GSOP_MC_DEBUG", "1")
    assert MC.debug_enabled()
