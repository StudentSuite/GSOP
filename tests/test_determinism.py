"""Reproducibility is the proof that two machines ran the same experiment (issue C23)."""

import pandas as pd
import pytest

import mc_helpers as H
import stage_inputs as SI
from gsop import montecarlo as MC
from gsop.paths import Paths
from gsop.stages import s3, s4, s5

CFG = H.CFG
FLOAT_TOLERANCE = 1e-6  # the issue's bound for float outputs; the engine meets it with equality (blocks and merge order do not depend on workers)


def case(seed=1, master_seed=None, n=3000):
    return H.synthetic_case("moon", n=n, seed=seed, master_seed=master_seed)


def params(runs=30, block=10):
    return MC.MCParams.from_config(CFG, runs=runs, block_runs=block)


def sim(x, ids, mcs, workers=1, runs=30):
    return MC.simulate(x, ids, mcs, params(runs), workers=workers, log=lambda m: None)


def test_same_seed_same_inputs_twice_gives_identical_counters_and_statistics():
    x, ids, mcs, g = case()
    a, b = sim(x, ids, mcs), sim(x, ids, mcs)
    pd.testing.assert_frame_equal(a.table, b.table, check_exact=True)
    assert a.convergence == b.convergence
    # the raw pieces too: counters (exact) and floating-point statistics
    p = params()
    cps = MC.scaled_checkpoints(p.runs, p.checkpoints)
    blk_a = MC.run_block(x, mcs, 0, 10, p.fractions, MC.segment_starts(cps))
    blk_b = MC.run_block(x, mcs, 0, 10, p.fractions, MC.segment_starts(cps))
    assert all((blk_a.counters[k] == blk_b.counters[k]).all() for k in blk_a.counters)
    assert (blk_a.stats.mean == blk_b.stats.mean).all() and (blk_a.stats.m2 == blk_b.stats.m2).all()


def test_1_vs_4_workers_identical_counters_and_floats_within_1e_6():
    x, ids, mcs, g = case(seed=2)
    one, four = sim(x, ids, mcs, workers=1), sim(x, ids, mcs, workers=4)
    t1, t4 = one.table.set_index("tile_id"), four.table.set_index("tile_id")
    assert t1.index.equals(t4.index)
    for c in ("freq_top1", "freq_top5", "freq_top10", "robust", "is_candidate"):  # counters and flags: exact
        assert (t1[c] == t4[c]).all(), c
    for c in ("mc_mean", "mc_std", "mc_median", "mc_p05", "mc_p95", "rank_pct_p05", "rank_pct_p50", "rank_pct_p95"):
        both = t1[c].notna()
        assert (t1[c].notna() == t4[c].notna()).all() and ((t1[c][both] - t4[c][both]).abs() <= FLOAT_TOLERANCE).all(), c
    assert one.convergence == four.convergence


def test_a_different_master_seed_changes_the_results_sanity():
    x, ids, mcs, g = case(seed=3)
    _, _, mcs2, _ = case(seed=3, master_seed=987654321)
    a, b = sim(x, ids, mcs), sim(x, ids, mcs2)
    assert mcs.master_seed != mcs2.master_seed
    merged = a.table.merge(b.table, on="tile_id", suffixes=("_a", "_b"))
    assert not (merged.mc_mean_a == merged.mc_mean_b).all()  # the jitter and weights differ, so the statistics differ
    assert not a.table.equals(b.table)
    # but the planted analog is found either way
    for t in (a.table, b.table):
        assert bool(t[t.tile_id == g.truth["analog"]].robust.iloc[0])


def test_stage_outputs_are_identical_across_two_runs_and_across_worker_counts(tmp_path):
    outs = []
    for k, workers in enumerate((1, 4, 1)):
        root, g, _ = SI.write_inputs(tmp_path / f"r{k}", "moon", n=2500, seed=5)
        paths = Paths(root)
        s3.run_gates(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon"), log=lambda m: None)
        s4.run_scores(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon") / "gates_passed.parquet", paths.results("moon"), log=lambda m: None)
        s5.run_simulate(CFG, "moon", paths, runs=20, workers=workers, block_runs=10, log=lambda m: None)
        outs.append(paths.results("moon"))
    for name in ("mc_connecting_ridge_c1_base.parquet", "mc_shackleton_rim_power_night.parquet", "scores_base.parquet"):
        a, b, c = (pd.read_parquet(o / name) for o in outs)
        pd.testing.assert_frame_equal(a, b)  # 1 vs 4 workers
        pd.testing.assert_frame_equal(a, c)  # the same configuration twice
