"""Memory: a 4,000,000-tile run stays inside the budget `gsop doctor` reports (issue C25).

The slow test generates 4,000,000 synthetic tiles, runs 5 Monte Carlo runs while a thread samples the memory of the process and its workers, and
asserts that what the run ADDED (peak minus the memory before the feature matrix existed) stays under the budget doctor prints (40% of free RAM).
Run once per machine with ``uv run pytest -q --runslow tests/test_memory.py -s`` and paste the printed numbers on the issue.
"""

import logging
import threading
import time

import numpy as np
import psutil
import pytest

import mc_helpers as H
from gsop import doctor
from gsop import montecarlo as MC

N_TILES = 4_000_000


def test_estimate_scales_with_tiles_workers_and_columns():
    base = MC.estimate_peak_bytes(1_000_000, 12)
    assert MC.estimate_peak_bytes(2_000_000, 12) == pytest.approx(2 * base, rel=0.02)
    assert MC.estimate_peak_bytes(1_000_000, 12, workers=4) > base
    assert MC.estimate_peak_bytes(1_000_000, 24) > base
    # at 4 million tiles with the real 4 checkpoints and 8 workers the estimate stays in the low gigabytes
    assert 1.5e9 < MC.estimate_peak_bytes(N_TILES, 14, workers=8) < 8e9
    assert 0.8e9 < MC.estimate_peak_bytes(N_TILES, 12, workers=1, n_checkpoints=5) < 1.1e9  # calibrated on the measured 866 MB


def test_budget_check_logs_warns_and_refuses():
    logs = []
    need = MC.check_memory_budget(1_000_000, 12, 1, 4, available_bytes=10**10, budget_bytes=10**10, log=logs.append)
    assert need > 0 and logs == []
    MC.check_memory_budget(1_000_000, 12, 1, 4, available_bytes=10**10, budget_bytes=1000, log=logs.append)
    assert len(logs) == 1 and "above the budget" in logs[0]
    with pytest.raises(MC.MemoryBudgetError, match="only 100 MB are free"):
        MC.check_memory_budget(N_TILES, 14, 4, 4, available_bytes=100 * 10**6, budget_bytes=10**10)


@pytest.mark.slow
def test_4_million_tiles_stay_inside_the_doctor_budget():
    logging.disable(logging.WARNING)
    try:
        info = doctor.collect()
        budget = info["chunk"]["megabytes"] * 1e6
        x0, _, mcs, _ = H.synthetic_case("moon", n=3000, seed=1)
        k = x0.shape[1]
        proc = psutil.Process()
        before = proc.memory_info().rss
        rng = np.random.default_rng(0)
        x = np.empty((N_TILES, k), dtype=np.float32)
        for i in range(0, N_TILES, 500_000):  # generate in blocks: no float64 temporaries
            x[i : i + 500_000] = rng.standard_normal((min(500_000, N_TILES - i), k), dtype=np.float32) * 3 + mcs.target_values.astype(np.float32)
        ids = np.arange(N_TILES, dtype=np.int64)
        params = MC.MCParams.from_config(H.CFG, runs=5, block_runs=5)
        n_cp = len(MC.scaled_checkpoints(params.runs, params.checkpoints))
        need = MC.estimate_peak_bytes(N_TILES, k, 1, n_cp)
        free = psutil.virtual_memory().available
        if need > budget:
            pytest.skip(f"this machine's doctor budget ({budget / 1e6:,.0f} MB, free RAM {free / 1e6:,.0f} MB) is below the estimate for 4,000,000 tiles "
                        f"({need / 1e6:,.0f} MB): the real run needs more free RAM here")
        peak = [0]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True)))
                time.sleep(0.02)

        th = threading.Thread(target=sample)
        th.start()
        t0 = time.perf_counter()
        try:
            res = MC.simulate(x, ids, mcs, params, workers=1, log=lambda m: None)
        finally:
            stop.set()
            th.join()
        added = peak[0] - before
        print(f"\n[memory] 4,000,000 tiles, {k} columns, 5 runs, 1 worker: feature matrix {x.nbytes / 1e6:,.0f} MB, peak added {added / 1e6:,.0f} MB "
              f"(estimate {need / 1e6:,.0f} MB, doctor budget {budget / 1e6:,.0f} MB, free RAM at start {free / 1e6:,.0f} MB), {time.perf_counter() - t0:.1f} s, "
              f"{len(res.table):,} table rows")
        assert added < budget, f"peak {added / 1e6:,.0f} MB exceeds the doctor budget {budget / 1e6:,.0f} MB: open a fix issue and block the real runs"
        assert added <= need, f"measured {added / 1e6:,.0f} MB is above the estimate {need / 1e6:,.0f} MB: the estimate is not conservative"
    finally:
        logging.disable(logging.NOTSET)


def test_s5_checks_the_memory_before_each_profile(tmp_path, monkeypatch):
    import stage_inputs as SI
    from gsop.paths import Paths
    from gsop.stages import s3, s4, s5

    root, g, _ = SI.write_inputs(tmp_path, "moon", n=1500, seed=1)
    paths = Paths(root)
    s3.run_gates(H.CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon"), log=lambda m: None)
    s4.run_scores(H.CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon") / "gates_passed.parquet", paths.results("moon"), log=lambda m: None)
    monkeypatch.setattr(s5, "memory_numbers", lambda: (1000, 500))  # 1 KB free: nothing can run
    with pytest.raises(MC.MemoryBudgetError, match="only 0 MB are free"):
        s5.run_simulate(H.CFG, "moon", paths, runs=2, workers=1, block_runs=2, log=lambda m: None)
    monkeypatch.setattr(s5, "memory_numbers", lambda: (10**10, 100))  # plenty free but a tiny budget: runs, with a warning
    logs = []
    s5.run_simulate(H.CFG, "moon", paths, runs=2, workers=1, block_runs=2, log=logs.append)
    assert any("above the budget" in m for m in logs)
