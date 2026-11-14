import json

import numpy as np
import pytest

import mc_helpers as H
from gsop import montecarlo as MC
from gsop.config import load_config

CFG = H.CFG


def params(runs=40, block=10, **kw):
    return MC.MCParams.from_config(CFG, runs=runs, block_runs=block, **kw)


def test_params_come_from_run_yaml_and_robust_fraction_is_one_of_the_top_fractions():
    p = MC.MCParams.from_config(CFG)
    assert (p.runs, p.fractions, p.robust_fraction, p.robust_min_share, p.candidate_fraction, p.checkpoints) == (2000, (0.01, 0.05, 0.10), 0.05, 0.9, 0.03, (500, 1000, 1500, 2000))
    assert p.robust_index == 1
    with pytest.raises(ValueError, match="not one of top_fractions"):
        MC.MCParams(10, (0.01, 0.10), 0.05, 0.9, 0.03, (10,)).robust_index


def test_checkpoints_and_block_plan():
    assert MC.scaled_checkpoints(2000, (500, 1000, 1500, 2000)) == [500, 1000, 1500, 2000]
    assert MC.scaled_checkpoints(50, (500, 1000, 1500, 2000)) == [12, 25, 38, 50]  # scaled in proportion
    assert MC.scaled_checkpoints(1, (500, 1000, 1500, 2000)) == [1]
    plan = MC.block_plan(2000, 125, [500, 1000, 1500, 2000])
    assert plan[0] == (0, 125) and plan[-1] == (1875, 2000) and len(plan) == 16
    assert all(b > a for a, b in plan) and sum(b - a for a, b in plan) == 2000 and all(plan[i][1] == plan[i + 1][0] for i in range(len(plan) - 1))
    odd = MC.block_plan(50, 20, [12, 25, 38, 50])
    assert odd == [(0, 12), (12, 20), (20, 25), (25, 38), (38, 40), (40, 50)] and all(not (a < c < b) for a, b in odd for c in (12, 25, 38))  # never spans a checkpoint
    assert MC.block_plan(2000, 125, [500, 1000, 1500, 2000]) == MC.block_plan(2000, 125, [500, 1000, 1500, 2000])  # no dependence on workers: no worker argument


def test_planted_analog_is_robust_and_a_single_family_matcher_is_not():
    x, ids, mcs, g = H.synthetic_case("moon", n=20_000, seed=3)
    res = MC.simulate(x, ids, mcs, params(runs=50, block=25), log=lambda m: None)
    t = res.table.set_index("tile_id")
    a = t.loc[g.truth["analog"]]
    assert bool(a.robust) and a.freq_top1 == a.freq_top5 == a.freq_top10 == 1.0 and a.is_candidate
    for tid in g.truth["single_family"].values():
        assert tid not in t.index or not bool(t.loc[tid].robust)
    assert t.robust.sum() < 0.02 * len(ids)  # robust tiles are rare
    assert set(g.truth["gate_trap"].values()).isdisjoint(set(t.index))  # gated out before the Monte Carlo


def test_table_columns_rows_and_types():
    x, ids, mcs, g = H.synthetic_case("moon", n=4000, seed=1)
    res = MC.simulate(x, ids, mcs, params(runs=20, block=10), log=lambda m: None)
    t = res.table
    assert list(t.columns) == ["tile_id", "mc_mean", "mc_std", "freq_top1", "freq_top5", "freq_top10", "robust", "is_candidate", "mc_median", "mc_p05", "mc_p95",
                               "rank_pct_p05", "rank_pct_p50", "rank_pct_p95"]
    assert t.tile_id.is_unique and t.robust.dtype == bool and t.freq_top5.dtype == np.float32
    assert ((t.freq_top1 <= t.freq_top5) & (t.freq_top5 <= t.freq_top10)).all() and t.freq_top10.between(0, 1).all()
    cand = t[t.is_candidate]
    assert cand[["mc_median", "mc_p05", "mc_p95", "rank_pct_p05", "rank_pct_p50", "rank_pct_p95"]].notna().all().all()
    assert t[~t.is_candidate][["mc_median", "rank_pct_p50"]].isna().all().all()
    assert (cand.mc_p05 <= cand.mc_median).all() and (cand.mc_median <= cand.mc_p95).all() and (cand.rank_pct_p05 <= cand.rank_pct_p95).all()
    assert (t[t.robust].freq_top5 >= 0.9).all() and (t[~t.robust].freq_top5 < 0.9).all()
    assert len(t) == int(((t.freq_top10 > 0) | t.is_candidate).sum())  # only tiles that ever reached the top 10% or are candidates are listed


def test_convergence_json_has_four_checkpoints_and_the_last_is_one():
    x, ids, mcs, g = H.synthetic_case("moon", n=5000, seed=2)
    res = MC.simulate(x, ids, mcs, params(runs=80, block=20), log=lambda m: None)
    cps = res.convergence["checkpoints"]
    assert len(cps) == 4 and [c["runs"] for c in cps] == [20, 40, 60, 80]
    assert cps[-1]["jaccard_top1_vs_final"] == 1.0 and all(0 <= c["jaccard_top1_vs_final"] <= 1 for c in cps)
    assert cps[2]["jaccard_top1_vs_final"] >= cps[0]["jaccard_top1_vs_final"] - 0.25  # more runs get closer to the final set (loose)
    json.dumps(res.convergence)  # serialisable


def test_jaccard_and_top1_set_by_hand():
    assert MC.jaccard(np.array([1, 2, 3]), np.array([2, 3, 4])) == pytest.approx(0.5) and MC.jaccard(np.array([]), np.array([])) == 1.0
    counts = np.zeros((10, 3), dtype=np.uint16)
    counts[3, 0], counts[7, 0], counts[7, 1] = 5, 5, 9  # tie on top1 count: tile 7 wins on the 5% count
    assert MC.top1_set(counts, 10, (0.1, 0.5, 1.0)).tolist() == [7]
    counts[3, 1] = 9
    assert MC.top1_set(counts, 10, (0.1, 0.5, 1.0)).tolist() == [3]  # still tied: lower index wins


def test_no_candidate_pass2_columns_when_pass2_is_absent():
    ids = np.arange(5)
    stats = MC.OnlineStats(5)
    stats.update(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    counts = np.zeros((5, 3), dtype=np.uint16)
    counts[4] = 1
    t = MC.build_results_table(ids, stats, counts, np.array([4]), None, MC.MCParams(1, (0.01, 0.05, 0.1), 0.05, 0.9, 0.03, (1,)))
    assert t.tile_id.tolist() == [4] and np.isnan(t.mc_median.iloc[0]) and bool(t.robust.iloc[0]) is True
