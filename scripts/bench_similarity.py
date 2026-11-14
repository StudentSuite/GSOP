"""Throughput of the similarity scoring: tiles per second per core (issue C07).

    uv run python scripts/bench_similarity.py [--tiles 2000000]

Scores a synthetic block with the Moon families against the planted target and prints tiles per second. One process, one core of numpy
element-wise work; a loaded machine gives lower numbers, so run it on a quiet one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=2_000_000)
    ap.add_argument("--body", default="moon", choices=["moon", "mars"])
    args = ap.parse_args()
    import synth

    from gsop import similarity as S
    from gsop import weights as W
    from gsop.config import load_config
    from gsop.targets import TargetProfile

    cfg = load_config(ROOT / "config")
    t = synth.target_profile(args.body, cfg)
    profile = TargetProfile("synthetic", dict(zip(t.column, t.value)), dict(zip(t.column, t.spread)), frozenset())
    setup = S.make_setup(cfg, args.body, profile)
    sigma = np.maximum(setup.target_spread, 1.0)
    d = W.derive_weights(cfg, args.body)
    w = S.renormalise({f: x.weight for f, x in d.items()}, setup.families)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(args.tiles, len(setup.columns))).astype(np.float32)
    S.score_tiles(x[:1000], setup.target_values, sigma, setup.family_cols, w)  # warm up
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        S.score_tiles(x, setup.target_values, sigma, setup.family_cols, w)
        best = min(best, time.perf_counter() - t0)
    print(f"{args.body}: {len(setup.columns)} columns, {len(setup.families)} families, {args.tiles:,} tiles in {best:.3f} s = {args.tiles / best:,.0f} tiles/s on one core")


if __name__ == "__main__":
    main()
