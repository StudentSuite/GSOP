import os
import pickle
from multiprocessing import shared_memory

import numpy as np
import pytest

from gsop import parallel
from gsop.parallel import SharedArray, run_parallel


def block_sum(task, shared):
    start, stop = task
    return float(shared["x"][start:stop].sum(dtype=np.float64))


def pid_and_writeable(task, shared):
    return os.getpid(), bool(shared["x"].flags.writeable)


def boom(task, shared):
    if task == 3:
        raise ValueError("worker exploded")
    return task


def test_results_are_ordered_and_independent_of_worker_count():
    x = np.random.default_rng(1).normal(size=(10_000, 4)).astype(np.float32)
    tasks = [(i, i + 1000) for i in range(0, 10_000, 1000)]
    one = run_parallel(block_sum, tasks, {"x": x}, workers=1)
    four = run_parallel(block_sum, tasks, {"x": x}, workers=4)
    assert one == four
    assert one[0] == float(x[:1000].sum(dtype=np.float64))


def test_workers_see_read_only_shared_views_in_other_processes():
    x = np.arange(1000, dtype=np.float32)
    out = run_parallel(pid_and_writeable, [0, 1, 2, 3], {"x": x}, workers=2)
    assert {w for _, w in out} == {False}
    assert {pid for pid, _ in out} != {os.getpid()}, "work must run in worker processes"


def test_big_array_is_not_pickled_only_the_spec_is():
    x = np.zeros((2_000_000, 4), dtype=np.float32)  # 32 MB
    with SharedArray(x) as seg:
        assert len(pickle.dumps(seg.spec)) < 500
        assert np.array_equal(seg.array[:10], x[:10])


def test_shared_memory_is_released_after_normal_exit_and_after_exception():
    x = np.ones(1000, dtype=np.float32)
    with SharedArray(x) as seg:
        name = seg.spec.name
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)

    with pytest.raises(RuntimeError, match="parent failed"):
        with SharedArray(x) as seg:
            name = seg.spec.name
            raise RuntimeError("parent failed")
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
    assert parallel._LIVE == {}


def test_worker_exception_propagates_and_leaves_no_segments():
    x = np.ones(1000, dtype=np.float32)
    with pytest.raises(ValueError, match="worker exploded"):
        run_parallel(boom, [0, 1, 2, 3, 4], {"x": x}, workers=2)
    assert parallel._LIVE == {}


def test_default_workers_is_physical_cores():
    import psutil

    assert parallel.default_workers() == (psutil.cpu_count(logical=False) or os.cpu_count())
    with pytest.raises(ValueError):
        run_parallel(block_sum, [(0, 1)], {"x": np.ones(2)}, workers=0)


def test_no_shared_arrays_is_fine():
    assert run_parallel(boom, [0, 1], None, workers=2) == [0, 1]
