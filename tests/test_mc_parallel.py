"""Monte Carlo parallelism, resume and worker-count independence (issue C21). Spawned workers: runs the same on Windows and macOS."""

import numpy as np
import pandas as pd
import pytest

import mc_helpers as H
from gsop import montecarlo as MC
from gsop.parallel import iter_parallel


def case(n=4000, seed=1):
    return H.synthetic_case("moon", n=n, seed=seed)


def params(runs=40, block=10):
    return MC.MCParams.from_config(H.CFG, runs=runs, block_runs=block)


def sim(x, ids, mcs, p, **kw):
    return MC.simulate(x, ids, mcs, p, log=lambda m: None, **kw)


def test_1_and_4_workers_give_identical_results():
    x, ids, mcs, g = case()
    a = sim(x, ids, mcs, params(), workers=1)
    b = sim(x, ids, mcs, params(), workers=4)
    pd.testing.assert_frame_equal(a.table, b.table)  # exactly equal: counters AND float statistics, because blocks and merge order do not depend on workers
    assert a.convergence == b.convergence and a.candidates == b.candidates
    assert b.timing["workers"] == 4 and a.timing["workers"] == 1


def double_plus_shared(task, shared):  # module level: spawned workers must be able to import it
    return task * 2 + int(shared["a"][0])


def test_iter_parallel_yields_in_order_and_bounded_and_matches_serial():
    a = np.array([100.0])
    got = list(iter_parallel(double_plus_shared, range(9), {"a": a}, workers=3, window=2))
    assert got == [i * 2 + 100 for i in range(9)] == list(iter_parallel(double_plus_shared, range(9), {"a": a}, workers=1))


def boom(task, shared):
    if task == 3:
        raise ValueError("worker failed")
    return task


def test_iter_parallel_releases_the_pool_and_shared_memory_on_error():
    from gsop import parallel

    gen = iter_parallel(boom, range(6), {"a": np.ones(10)}, workers=2)
    with pytest.raises(ValueError, match="worker failed"):
        list(gen)
    assert parallel._LIVE == {}


def test_a_crash_midway_resumes_with_only_the_missing_blocks(tmp_path, monkeypatch):
    x, ids, mcs, g = case(n=3000, seed=2)
    p = params(runs=40, block=10)  # blocks (0,10) (10,20) (20,30) (30,40) + checkpoint splits at 10, 20, 30
    reference = sim(x, ids, mcs, p).table
    calls = []
    real = MC.compute_pass1_block

    def flaky(task, shared):
        calls.append((task[1], task[2]))
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated Ctrl-C")  # the third block never finishes
        return real(task, shared)

    monkeypatch.setattr(MC, "compute_pass1_block", flaky)
    with pytest.raises(KeyboardInterrupt):
        sim(x, ids, mcs, p, work_dir=tmp_path / "w")
    saved = sorted(f.name for f in (tmp_path / "w").glob("p1_*.npz"))
    assert saved == ["p1_0_10.npz", "p1_10_20.npz"]  # two blocks were saved before the interrupt
    calls.clear()
    monkeypatch.setattr(MC, "compute_pass1_block", lambda task, shared: (calls.append((task[1], task[2])), real(task, shared))[1])
    res = sim(x, ids, mcs, p, work_dir=tmp_path / "w")
    assert calls == [(20, 30), (30, 40)] and res.timing["resumed_blocks"] == 2  # only the missing blocks ran
    pd.testing.assert_frame_equal(res.table, reference)  # and the answer is the uninterrupted one
    assert not (tmp_path / "w").exists()  # partial files are removed after success


def test_resume_works_with_more_workers_than_the_interrupted_run_had(tmp_path, monkeypatch):
    x, ids, mcs, g = case(n=3000, seed=3)
    p = params(runs=40, block=10)
    reference = sim(x, ids, mcs, p).table
    real = MC.compute_pass1_block
    n = []

    def stop_after_two(task, shared):
        n.append(1)
        if len(n) == 3:
            raise RuntimeError("laptop lid closed")
        return real(task, shared)

    monkeypatch.setattr(MC, "compute_pass1_block", stop_after_two)
    with pytest.raises(RuntimeError):
        sim(x, ids, mcs, p, work_dir=tmp_path / "w", workers=1)
    monkeypatch.setattr(MC, "compute_pass1_block", real)
    res = sim(x, ids, mcs, p, work_dir=tmp_path / "w", workers=3)
    pd.testing.assert_frame_equal(res.table, reference)
    assert res.timing["resumed_blocks"] == 2


def test_pass2_resumes_too(tmp_path, monkeypatch):
    x, ids, mcs, g = case(n=3000, seed=4)
    p = params(runs=30, block=10)
    reference = sim(x, ids, mcs, p).table
    real = MC.pass2_block
    count = []

    def stop(x_, mcs_, cand, a, b, store, chunk_rows=0, debug=None):
        count.append((a, b))
        if len(count) == 2:
            raise RuntimeError("crash in pass 2")
        return real(x_, mcs_, cand, a, b, store, chunk_rows) if chunk_rows else real(x_, mcs_, cand, a, b, store)

    monkeypatch.setattr(MC, "pass2_block", stop)
    with pytest.raises(RuntimeError, match="pass 2"):
        sim(x, ids, mcs, p, work_dir=tmp_path / "w", keep_work=True)
    plan = MC.block_plan(30, 10, MC.scaled_checkpoints(30, p.checkpoints))
    first, second = plan[0], plan[1]
    assert (tmp_path / "w" / "pass2" / f"done_{first[0]}_{first[1]}.txt").exists()
    assert not (tmp_path / "w" / "pass2" / f"done_{second[0]}_{second[1]}.txt").exists()  # the crash came during the second block
    monkeypatch.setattr(MC, "pass2_block", real)
    res = sim(x, ids, mcs, p, work_dir=tmp_path / "w")
    pd.testing.assert_frame_equal(res.table, reference)


def test_partial_results_from_other_settings_are_discarded_not_mixed_in(tmp_path, monkeypatch):
    x, ids, mcs, g = case(n=2000, seed=5)
    p = params(runs=20, block=10)
    real = MC.compute_pass1_block
    n = []

    def stop(task, shared):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("stop")
        return real(task, shared)

    monkeypatch.setattr(MC, "compute_pass1_block", stop)
    with pytest.raises(RuntimeError):
        sim(x, ids, mcs, p, work_dir=tmp_path / "w")
    assert list((tmp_path / "w").glob("p1_*.npz"))
    monkeypatch.setattr(MC, "compute_pass1_block", real)
    other = MC.MCSetup(**{**mcs.__dict__, "master_seed": 7})  # a different master seed
    logs = []
    res = MC.simulate(x, ids, other, p, work_dir=tmp_path / "w", log=logs.append)
    assert any("discarding" in m for m in logs) and res.timing["resumed_blocks"] == 0
    ref = sim(x, ids, other, p).table
    pd.testing.assert_frame_equal(res.table, ref)


def test_block_results_are_independent_of_the_chunk_size_and_worker_count():
    x, ids, mcs, g = case(n=2500, seed=6)
    p = params(runs=20, block=10)
    a = sim(x, ids, mcs, p, workers=1, chunk_rows=2500)
    b = sim(x, ids, mcs, p, workers=2, chunk_rows=313)
    pd.testing.assert_frame_equal(a.table, b.table)
