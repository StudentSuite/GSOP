"""Benchmark: speedup of gsop.parallel with 1, 4 and 8 workers on a synthetic array.

    uv run python scripts/bench_parallel.py [--rows 1000000] [--cols 15] [--tasks 64]

Workers are spawned (fresh interpreters), so each pays about a second of start-up. Speedup only
appears when the work is much longer than that, as a real 2,000-run Monte Carlo is.

The kernel is a small stand-in for one Monte Carlo batch: Gaussian similarity of every row to a
target, averaged over columns. The feature matrix is shared, not pickled; only (start, stop) pairs
travel to the workers.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gsop.parallel import default_workers, run_parallel


def kernel(task, shared):
    start, stop, seed = task
    x = shared["x"][start:stop]
    rng = np.random.default_rng(seed)
    target = rng.normal(size=x.shape[1]).astype(np.float32)
    best = 0.0
    for _ in range(40):  # enough passes that the work is CPU-bound and outweighs worker start-up
        s = np.exp(-0.5 * (x - target) ** 2).mean(axis=1)
        best = max(best, float(s.max()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--cols", type=int, default=15)
    ap.add_argument("--tasks", type=int, default=64)
    args = ap.parse_args()

    x = np.random.default_rng(0).normal(size=(args.rows, args.cols)).astype(np.float32)
    step = max(1, args.rows // args.tasks)
    tasks = [(i, min(i + step, args.rows), 1000 + n) for n, i in enumerate(range(0, args.rows, step))]
    print(f"array {x.nbytes / 1e6:.0f} MB, {len(tasks)} tasks, default workers {default_workers()}")

    baseline = None
    reference = None
    for workers in (1, 4, 8):
        t0 = time.perf_counter()
        out = run_parallel(kernel, tasks, {"x": x}, workers=workers)
        dt = time.perf_counter() - t0
        baseline = dt if baseline is None else baseline
        reference = out if reference is None else reference
        same = "identical results" if out == reference else "RESULTS DIFFER"
        print(f"workers={workers}: {dt:6.2f} s  speedup x{baseline / dt:4.2f}  ({same})")


if __name__ == "__main__":  # required: Windows and macOS spawn workers that re-import this file
    main()
