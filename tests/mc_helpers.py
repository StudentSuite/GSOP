"""A complete synthetic Monte Carlo case: gated tiles, sigma, set-up. Shared by the engine tests."""

from pathlib import Path

import numpy as np

import synth
from gsop import gates as G
from gsop import montecarlo as MC
from gsop import similarity as S
from gsop.config import load_config
from gsop.targets import TargetProfile

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def synthetic_case(body="moon", n=5000, seed=0, profile="base", cfg=CFG, master_seed=None, away=1.0):
    """Return ``(x float32 gated tiles in tile_id order, tile_ids, MCSetup, grid)`` for the planted-analog grid.

    ``away=1.0`` puts the synthetic target one background standard deviation inside the bulk, so partial matches compete (measured: the analog is in
    every top set of every run while one-family tiles never reach the top 10%; at 2.0 or more one-family tiles start to rank because every background
    tile is far from the target).
    """
    g = synth.make_grid(n, seed=seed, body=body, cfg=cfg, away=away)
    t = synth.target_profile(body, cfg, away)
    prof = TargetProfile("synthetic", dict(zip(t.column, t.value)), dict(zip(t.column, t.spread)), frozenset())
    ss = S.make_setup(cfg, body, prof)
    cols = {c: g.df[c].to_numpy("float64") for c in g.df.columns}
    res = G.evaluate_gates(cfg, body, cols, list(ss.columns))
    keep = res.passed(applies_landing_gate=profile == "landing_mobility")
    df = g.df[keep].sort_values("tile_id")
    x = df[list(ss.columns)].to_numpy("float32")
    table = S.build_sigma_table(cfg, ss, {c: x[:, i] for i, c in enumerate(ss.columns)})
    ss = ss.with_sigma(S.sigma_vector(ss, table))
    mcs = MC.make_mc_setup(cfg, ss, profile, master_seed)
    return x, df["tile_id"].to_numpy("int64"), mcs, g
