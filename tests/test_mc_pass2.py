import numpy as np
import pytest

import mc_helpers as H
from gsop import montecarlo as MC

FRACS = (0.01, 0.05, 0.10)


def pass1(n=4000, runs=30, seed=1, body="moon"):
    x, ids, mcs, g = H.synthetic_case(body, n=n, seed=seed)
    block = MC.run_block(x, mcs, 0, runs, FRACS)
    return x, ids, mcs, g, block, runs


def test_candidate_rule_is_the_union_of_top_by_mean_and_top_by_frequency():
    n = 1000
    mean = np.linspace(0, 1, n)  # tile 999 has the highest mean
    count = np.zeros(n, dtype=np.uint16)
    count[[5, 6, 7]] = 30  # three tiles with low means but top membership
    cand = MC.select_candidates(mean, count, 0.03)  # k = 30
    assert len(cand) == 33 and cand.tolist() == sorted(cand.tolist())
    assert set(range(970, 1000)) <= set(cand.tolist()) and {5, 6, 7} <= set(cand.tolist())
    assert len(MC.select_candidates(mean, np.zeros(n, dtype=np.uint16), 0.03)) == 30  # no frequency signal: just the top 3% by mean


def test_the_planted_analog_is_a_candidate_with_a_near_perfect_rank_range():
    x, ids, mcs, g, block, runs = pass1()
    cand = MC.select_candidates(block.stats.mean, block.counters[0][:, 1], 0.03)
    analog = int(np.flatnonzero(ids == g.truth["analog"])[0])
    assert analog in cand
    store = MC.Pass2Store.__new__(MC.Pass2Store)  # via a temp dir below
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = MC.Pass2Store(d, runs, len(cand))
        MC.pass2_block(x, mcs, cand, 0, runs, store)
        summ = MC.summarise_candidates(store)
    j = int(np.flatnonzero(cand == analog)[0])
    assert summ["rank_pct_p05"][j] > 99.9 and summ["rank_pct_p50"][j] > 99.9 and summ["rank_pct_p95"][j] <= 100.0
    assert summ["mc_p05"][j] <= summ["mc_median"][j] <= summ["mc_p95"][j]


def test_pass2_scores_equal_pass1_scores_for_the_same_runs_and_seeds(tmp_path):
    x, ids, mcs, g, block, runs = pass1(runs=12)
    cand = MC.select_candidates(block.stats.mean, block.counters[0][:, 1], 0.03)
    store = MC.Pass2Store(tmp_path, runs, len(cand))
    MC.pass2_block(x, mcs, cand, 0, runs, store, chunk_rows=777)  # a different chunking than pass 1
    scores, ranks = store.open()
    for i in (0, 5, 11):
        assert np.array_equal(scores[i], MC.score_run(x, mcs, i)[cand])  # bit-identical, not just close
    # mean over runs of the stored candidate scores equals the pass 1 online mean
    assert np.allclose(np.asarray(scores).mean(axis=0, dtype=np.float64), block.stats.mean[cand], atol=1e-4)


def test_percentiles_match_numpys_percentile_on_the_stored_arrays(tmp_path):
    x, ids, mcs, g, block, runs = pass1(runs=25)
    cand = MC.select_candidates(block.stats.mean, block.counters[0][:, 1], 0.02)
    store = MC.Pass2Store(tmp_path, runs, len(cand))
    MC.pass2_block(x, mcs, cand, 0, runs, store)
    scores, ranks = (np.asarray(a) for a in store.open())
    summ = MC.summarise_candidates(store, chunk_cols=7)  # small column blocks exercise the chunked path
    assert np.allclose(summ["mc_median"], np.percentile(scores, 50, axis=0), atol=1e-4) and np.allclose(summ["mc_p05"], np.percentile(scores, 5, axis=0), atol=1e-4)
    assert np.allclose(summ["mc_p95"], np.percentile(scores, 95, axis=0), atol=1e-4)
    r = ranks.astype(np.float64) / MC.RANK_SCALE
    assert np.allclose(summ["rank_pct_p05"], np.percentile(r, 5, axis=0), atol=1e-4) and np.allclose(summ["rank_pct_p95"], np.percentile(r, 95, axis=0), atol=1e-4)


def test_rank_percentile_is_the_share_of_tiles_strictly_below():
    x, ids, mcs, g, block, runs = pass1(n=1000, runs=3)
    score = MC.score_run(x, mcs, 0)
    cand = np.array([int(np.argmax(score)), int(np.argmin(score)), 123])
    s, r = MC.candidate_scores_and_ranks(x, mcs, 0, cand)
    expected = [100.0 * (score < score[c]).sum() / len(score) for c in cand]
    assert np.allclose(r / MC.RANK_SCALE, expected, atol=0.006)  # stored in hundredths of a point
    assert r[0] / MC.RANK_SCALE == pytest.approx(99.9, abs=0.01) and r[1] == 0  # the best tile is above 99.9% of the others, the worst above none


def test_a_finished_block_is_skipped_on_resume(tmp_path):
    x, ids, mcs, g, block, runs = pass1(runs=8)
    cand = MC.select_candidates(block.stats.mean, block.counters[0][:, 1], 0.02)
    store = MC.Pass2Store(tmp_path, runs, len(cand))
    MC.pass2_block(x, mcs, cand, 0, 4, store)
    assert store.block_marker(0, 4).exists() and not store.block_marker(4, 8).exists()
    scores, _ = store.open()
    first = np.array(scores[:4])
    scores[0, 0] = -1.0  # tamper: a skipped block must not be recomputed
    scores.flush()
    MC.pass2_block(x, mcs, cand, 0, 4, store)
    assert np.asarray(store.open()[0])[0, 0] == -1.0
    MC.pass2_block(x, mcs, cand, 4, 8, store)
    assert store.block_marker(4, 8).exists() and np.array_equal(np.asarray(store.open()[0])[1:4], first[1:4])
