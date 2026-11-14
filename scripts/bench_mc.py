"""Time one target x profile Monte Carlo on synthetic tiles (issue C26).

    uv run python scripts/bench_mc.py [--tiles 4000000] [--runs 200] [--workers 8] [--body moon] [--profile-runs 20]

Prints seconds, runs per second and the projection to 2,000 runs. With ``--profile-runs N`` it also runs N runs in this process under cProfile
(works on Windows and macOS) and prints the 25 most expensive functions. The data is Gaussian around the synthetic target (3 sd wide), so the
results table is much bigger than real data would give: treat the time as a bit pessimistic. Quiet machine only: other work skews the numbers.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def make_tiles(n: int, mcs) -> tuple[np.ndarray, np.ndarray]:
    k = len(mcs.target_values)
    rng = np.random.default_rng(0)
    x = np.empty((n, k), dtype=np.float32)
    for i in range(0, n, 500_000):
        x[i : i + 500_000] = rng.standard_normal((min(500_000, n - i), k), dtype=np.float32) * 3 + mcs.target_values.astype(np.float32)
    return x, np.arange(n, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=4_000_000)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0, help="0 = the default (cores - 1)")
    ap.add_argument("--body", default="moon", choices=["moon", "mars"])
    ap.add_argument("--profile-runs", type=int, default=0)
    args = ap.parse_args()
    import mc_helpers as H

    from gsop import montecarlo as MC
    from gsop.parallel import default_workers

    workers = args.workers or default_workers()
    _, _, mcs, _ = H.synthetic_case(args.body, n=3000, seed=1)
    x, ids = make_tiles(args.tiles, mcs)
    print(f"{args.body}: {args.tiles:,} tiles x {x.shape[1]} columns ({x.nbytes / 1e6:.0f} MB), {workers} workers")

    def run(runs: int, w: int):
        params = MC.MCParams.from_config(H.CFG, runs=runs)
        t0 = time.perf_counter()
        MC.simulate(x, ids, mcs, params, work_dir=None, workers=w, log=lambda m: None)
        return time.perf_counter() - t0

    dt = run(args.runs, workers)
    full = dt / args.runs * 2000
    print(f"{args.runs} runs, {workers} workers: {dt:.1f} s = {args.runs / dt:.2f} runs/s -> 2,000 runs about {full / 60:.1f} min")

    if args.profile_runs:
        prof = cProfile.Profile()
        prof.enable()
        run(args.profile_runs, 1)
        prof.disable()
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(25)
        print(out.getvalue())


if __name__ == "__main__":  # required: Windows and macOS spawn workers that re-import this file
    main()
