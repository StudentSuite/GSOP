import numpy as np
import pytest

import mc_helpers as H
from gsop import montecarlo as MC

FRACS = (0.01, 0.05, 0.10)


def reference_top(score, k):
    """Brute force: sort by (-score, index)."""
    order = sorted(range(len(score)), key=lambda i: (-score[i], i))
    return set(order[:k])


def test_sizes_are_ceil_of_the_fraction():
    assert MC.top_k_sizes(1000, FRACS) == [10, 50, 100] and MC.top_k_sizes(1001, FRACS) == [11, 51, 101] and MC.top_k_sizes(3, FRACS) == [1, 1, 1]


def test_top_sets_match_a_brute_force_ranking_and_are_nested():
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 100, 1000).astype(np.float32)
    sets = MC.top_sets(score, FRACS)
    for s, k in zip(sets, (10, 50, 100)):
        assert set(s.tolist()) == reference_top(score, k) and len(s) == k
        assert (np.diff(score[s]) <= 0).all()  # returned in ranked order
    assert set(sets[0]) <= set(sets[1]) <= set(sets[2])


def test_ties_are_broken_by_the_lower_index_whatever_the_internal_order():
    score = np.zeros(1000, dtype=np.float32)  # everything tied
    sets = MC.top_sets(score, FRACS)
    assert sets[0].tolist() == list(range(10)) and sets[1].tolist() == list(range(50)) and sets[2].tolist() == list(range(100))
    # a tie that straddles the cut: 5 tiles score 9, the cut falls in the middle of the 5 equal ones
    s2 = np.arange(1000, dtype=np.float32)
    s2[[10, 20, 30, 40, 50]] = 990.0  # tile 990 and these five tie at 990; the nine tiles above (991..999) are clear winners
    top10 = MC.top_sets(s2, (0.01,))[0]
    # nine clear winners fill 9 of the 10 places; the tie at the cut has six tiles for ONE place: the lowest index, tile 10, takes it
    assert top10.tolist() == [999, 998, 997, 996, 995, 994, 993, 992, 991, 10]
    s3 = np.zeros(1000, dtype=np.float32)
    s3[500] = 5.0
    s3[[7, 3, 900, 20]] = 1.0  # best is 500, then the tied ones: lower indices first
    assert MC.top_sets(s3, (0.005,))[0].tolist() == [500, 3, 7, 20, 900][:5]
    shuffled = np.random.default_rng(1).permutation(1000)  # same scores, different array order: the rule is about index, so ids matter
    assert MC.top_sets(s3[shuffled], (0.005,))[0].tolist() == sorted(MC.top_sets(s3[shuffled], (0.005,))[0].tolist(), key=lambda i: (-s3[shuffled][i], i))


def test_nan_scores_are_refused():
    s = np.ones(100, dtype=np.float32)
    s[3] = np.nan
    with pytest.raises(ValueError, match="NaN scores cannot be ranked"):
        MC.top_sets(s, FRACS)


def test_counters_accumulate_per_run_and_stay_uint16():
    counters = np.zeros((1000, 3), dtype=np.uint16)
    rng = np.random.default_rng(2)
    total = np.zeros((1000, 3), dtype=np.int64)
    for _ in range(25):
        s = rng.uniform(size=1000).astype(np.float32)
        MC.add_membership(counters, s, FRACS)
        for j, k in enumerate((10, 50, 100)):
            total[list(reference_top(s, k)), j] += 1
    assert counters.dtype == np.uint16 and np.array_equal(counters, total)
    assert MC.MAX_RUNS_UINT16 == 65535 > 2000  # 2,000 runs cannot overflow a uint16 counter


def test_planted_analog_has_frequency_one_in_every_top_set():
    x, ids, mcs, g = H.synthetic_case("moon", n=5000, seed=1)
    runs = 30
    block = MC.run_block(x, mcs, 0, runs, FRACS)
    counters = block.counters[0]
    analog = int(np.flatnonzero(ids == g.truth["analog"])[0])
    assert (counters[analog] == runs).all()  # freq_top1 = freq_top5 = freq_top10 = 1.0
    # the tiles that match on one family only never reach the top 10%
    for tid in g.truth["single_family"].values():
        assert (counters[np.flatnonzero(ids == tid)[0]] == 0).all()
    # nested sets: every tile counted in top1 is also counted in top5 and top10 at least as often
    assert (counters[:, 0] <= counters[:, 1]).all() and (counters[:, 1] <= counters[:, 2]).all()
    assert counters[:, 2].sum() == runs * MC.top_k_sizes(len(ids), FRACS)[2]  # exactly k10 members per run


def test_segments_split_the_counters_at_the_checkpoints():
    x, ids, mcs, g = H.synthetic_case("moon", n=2000, seed=2)
    block = MC.run_block(x, mcs, 0, 10, FRACS, segment_edges=[4, 7])
    assert sorted(block.counters) == [0, 1, 2]
    per_seg_runs = {seg: int(c[:, 2].sum() // MC.top_k_sizes(len(ids), FRACS)[2]) for seg, c in block.counters.items()}
    # an edge is the run index that STARTS the next segment: runs 0-3 | 4-6 | 7-9 (so checkpoints 500, 1000, 1500 give the first 500, 1000, 1500 runs)
    assert per_seg_runs == {0: 4, 1: 3, 2: 3}
    whole = MC.run_block(x, mcs, 0, 10, FRACS)
    assert np.array_equal(sum(block.counters.values()), whole.counters[0])
