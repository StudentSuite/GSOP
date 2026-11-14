"""The core correctness test: the engine must find the tile we hid (issue C24).

20,000 synthetic tiles, 50 runs, the full s3 -> s4 -> s5 pipeline on real files. The planted analog equals the target on every column; one tile matches
on a single family only; three tiles equal the target but break a gate; three have missing data.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stage_inputs as SI
from gsop.config import load_config
from gsop.paths import Paths
from gsop.stages import s3, s4, s5

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
SECONDS_BUDGET = 60.0


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planted")
    root, g, _ = SI.write_inputs(tmp, "moon", n=20_000, seed=11, away=1.0)
    paths = Paths(root)
    t0 = time.perf_counter()
    s3.run_gates(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon"), log=lambda m: None)
    s4.run_scores(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon") / "gates_passed.parquet", paths.results("moon"), log=lambda m: None)
    s5.run_simulate(CFG, "moon", paths, runs=50, workers=4, log=lambda m: None)
    elapsed = time.perf_counter() - t0
    return paths, g, elapsed


def test_finishes_in_under_a_minute(pipeline):
    paths, g, elapsed = pipeline
    assert elapsed < SECONDS_BUDGET, f"50 runs on 20,000 tiles took {elapsed:.1f} s"


def test_planted_analog_ranks_first_on_the_base_score_and_is_robust(pipeline):
    paths, g, _ = pipeline
    out = paths.results("moon")
    scores = pd.read_parquet(out / "scores_base.parquet")
    for target in ("connecting_ridge_c1", "shackleton_rim"):
        d = scores[scores.target == target].sort_values("score_base", ascending=False)
        assert d.iloc[0].tile_id == g.truth["analog"], target
        mc = pd.read_parquet(out / f"mc_{target}_base.parquet").set_index("tile_id")
        a = mc.loc[g.truth["analog"]]
        assert bool(a.robust) and a.freq_top1 == a.freq_top5 == a.freq_top10 == 1.0
        assert a.rank_pct_p05 > 99.9 and a.mc_p05 > 60  # a near-perfect rank and a high score under every draw


def test_the_tile_matching_one_family_is_not_robust_and_not_even_top_ranked(pipeline):
    paths, g, _ = pipeline
    out = paths.results("moon")
    scores = pd.read_parquet(out / "scores_base.parquet")
    d = scores[scores.target == "connecting_ridge_c1"].set_index("tile_id")
    mc = pd.read_parquet(out / "mc_connecting_ridge_c1_base.parquet").set_index("tile_id")
    ranks = d.score_base.rank(ascending=False)
    for fam, tid in g.truth["single_family"].items():
        assert tid not in mc.index or not bool(mc.loc[tid].robust), fam  # never robust
        assert ranks.loc[tid] > 0.02 * len(d), (fam, ranks.loc[tid])  # and far below the top 2%
        assert d.loc[tid].score_base < 40, fam  # a single family can carry at most its weight (0.30 for slope)


def test_gate_traps_and_missing_data_tiles_are_absent_from_the_passed_set(pipeline):
    paths, g, _ = pipeline
    out = paths.results("moon")
    passed = pd.read_parquet(out / "gates_passed.parquet")
    scores = pd.read_parquet(out / "scores_base.parquet")
    forbidden = set(g.truth["gate_trap"].values()) | set(g.truth["missing"].values())
    assert set(passed.tile_id).isdisjoint(forbidden) and set(scores.tile_id).isdisjoint(forbidden)
    rejected = pd.read_parquet(out / "gates_rejected.parquet")
    assert forbidden <= set(rejected.tile_id)
    for p in out.glob("mc_*.parquet"):
        assert set(pd.read_parquet(p).tile_id).isdisjoint(forbidden), p.name


def test_robust_tiles_are_rare_and_include_the_near_misses_in_order_of_closeness(pipeline):
    paths, g, _ = pipeline
    mc = pd.read_parquet(paths.results("moon") / "mc_connecting_ridge_c1_base.parquet").set_index("tile_id")
    assert 1 <= mc.robust.sum() < 0.03 * 20_000
    near = [mc.loc[t] for t in g.truth["near_miss"]]
    assert all(bool(r.robust) for r in near)  # tiles within 1.5 target spreads are among the robust ones
    assert near[0].mc_mean >= near[1].mc_mean >= near[2].mc_mean  # the closer one scores higher on average
