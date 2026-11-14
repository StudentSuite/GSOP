import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stage_inputs as SI
from gsop import montecarlo as MC
from gsop.config import load_config
from gsop.paths import Paths
from gsop.runner import Runner
from gsop.stages import s3, s4, s5

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def prepare(tmp, body="moon", n=3000, seed=1, **kw):
    root, g, _ = SI.write_inputs(tmp, body, n=n, seed=seed, **kw)
    paths = Paths(root)
    s3.run_gates(CFG, body, paths.earth_parquet, paths.targets_parquet(body), paths.results(body), log=lambda m: None)
    s4.run_scores(CFG, body, paths.earth_parquet, paths.targets_parquet(body), paths.results(body) / "gates_passed.parquet", paths.results(body), log=lambda m: None)
    return root, paths, g


def test_runs_5_end_to_end_writes_every_target_profile_file(tmp_path):
    root, paths, g = prepare(tmp_path)
    summary = s5.run_simulate(CFG, "moon", paths, runs=5, workers=1, block_runs=5, log=lambda m: None)
    out = paths.results("moon")
    files = sorted(p.name for p in out.glob("mc_*.parquet"))
    expected = sorted(f"mc_{t}_{p}.parquet" for t in ("connecting_ridge_c1", "shackleton_rim") for p in ("base", "landing_mobility", "power_night", "ice_resources"))
    assert files == expected
    conv = json.loads((out / "convergence.json").read_text(encoding="utf-8"))
    assert set(conv) == {f"{t}/{p}" for t in ("connecting_ridge_c1", "shackleton_rim") for p in ("base", "landing_mobility", "power_night", "ice_resources")}
    assert all(len(v["checkpoints"]) >= 1 and v["checkpoints"][-1]["jaccard_top1_vs_final"] == 1.0 for v in conv.values())
    t = summary["timing"]["connecting_ridge_c1/base"]
    assert t["runs"] == 5 and t["tiles"] > 2000 and t["total_seconds"] >= 0 and "pass1_seconds" in t


def test_columns_the_site_needs_are_on_every_row_and_the_analog_is_robust(tmp_path):
    root, paths, g = prepare(tmp_path, n=4000, seed=2)
    s5.run_simulate(CFG, "moon", paths, runs=20, workers=1, block_runs=10, log=lambda m: None)
    df = pd.read_parquet(paths.results("moon") / "mc_connecting_ridge_c1_base.parquet")
    for c in ("target", "profile", "confidence", "fails_own_gate", "own_gate_reasons", "tile_id", "robust", "freq_top5", "mc_median", "rank_pct_p50"):
        assert c in df.columns
    assert (df.target == "connecting_ridge_c1").all() and (df.profile == "base").all() and not df.fails_own_gate.any()
    row = df[df.tile_id == g.truth["analog"]]
    assert len(row) == 1 and bool(row.robust.iloc[0]) and row.freq_top5.iloc[0] == 1.0
    land = pd.read_parquet(paths.results("moon") / "mc_connecting_ridge_c1_landing_mobility.parquet")
    gated = pd.read_parquet(paths.results("moon") / "gates_passed.parquet")
    ok_ids = set(gated[(gated.target == "connecting_ridge_c1") & gated.landing_ok].tile_id)
    assert set(land.tile_id) <= ok_ids  # the landing profile only ever lists tiles whose extra gate passed


def test_mars_carries_confidence_low_and_the_own_gate_label(tmp_path):
    root, paths, g = prepare(tmp_path, body="mars", n=3000, seed=3, mars_elevation=2400.0)
    s5.run_simulate(CFG, "mars", paths, runs=5, workers=1, block_runs=5, log=lambda m: None)
    out = paths.results("mars")
    assert sorted(p.name for p in out.glob("mc_*.parquet")) == sorted(f"mc_jezero_{p}.parquet" for p in ("base", "landing_mobility", "power_night", "ice_resources", "geology_life"))
    for prof, conf in (("landing_mobility", "low"), ("power_night", "low"), ("base", None), ("geology_life", None)):
        df = pd.read_parquet(out / f"mc_jezero_{prof}.parquet")
        assert (df.confidence == conf).all() if conf else df.confidence.isna().all()
        assert df.fails_own_gate.all() and (df.own_gate_reasons == "MARS_TARGET_CHECK:ELEVATION").all()


def test_sigma_is_the_one_s4_wrote(tmp_path, monkeypatch):
    root, paths, g = prepare(tmp_path, n=2000, seed=4)
    seen = []
    real = MC.make_mc_setup
    monkeypatch.setattr(MC, "make_mc_setup", lambda cfg, setup, profile, master_seed=None: (seen.append(setup.sigma.copy()), real(cfg, setup, profile, master_seed))[1])
    s5.run_simulate(CFG, "moon", paths, runs=2, workers=1, block_runs=2, log=lambda m: None)
    table = json.loads((paths.results("moon") / "sigma_table.json").read_text(encoding="utf-8"))["targets"]["connecting_ridge_c1"]
    first = seen[0]
    cols = [c for c in table]  # same column order as the setup uses (insertion order of the sigma table)
    assert np.allclose(first, [table[c]["sigma"] for c in cols])


def test_an_interrupted_run_resumes_at_the_right_profile(tmp_path, monkeypatch):
    root, paths, g = prepare(tmp_path, n=2000, seed=5)
    real = MC.simulate
    calls = []

    def flaky(x, ids, mcs, params, **kw):
        calls.append(f"{mcs.target}/{mcs.profile}")
        if len(calls) == 4:
            raise KeyboardInterrupt("crash at the fourth profile")
        return real(x, ids, mcs, params, **kw)

    monkeypatch.setattr(MC, "simulate", flaky)
    with pytest.raises(KeyboardInterrupt):
        s5.run_simulate(CFG, "moon", paths, runs=4, workers=1, block_runs=2, log=lambda m: None)
    done = sorted(p.name for p in (paths.results("moon") / ".s5").glob("*.json"))
    assert len(done) == 3  # markers for the three finished profiles only
    first_three = calls[:3]
    calls.clear()
    logs = []
    monkeypatch.setattr(MC, "simulate", lambda x, ids, mcs, params, **kw: (calls.append(f"{mcs.target}/{mcs.profile}"), real(x, ids, mcs, params, **kw))[1])
    s5.run_simulate(CFG, "moon", paths, runs=4, workers=1, block_runs=2, log=logs.append)
    assert len(calls) == 8 - 3 and set(calls).isdisjoint(set(first_three))  # only the unfinished profiles ran
    assert sum("already done" in m for m in logs) == 3
    assert len(list(paths.results("moon").glob("mc_*.parquet"))) == 8


def test_a_changed_run_count_does_not_reuse_old_markers(tmp_path):
    root, paths, g = prepare(tmp_path, n=1500, seed=6)
    s5.run_simulate(CFG, "moon", paths, runs=4, workers=1, block_runs=2, log=lambda m: None)
    logs = []
    s5.run_simulate(CFG, "moon", paths, runs=6, workers=1, block_runs=2, log=logs.append)
    assert not any("already done" in m for m in logs)


def test_runner_skips_unchanged_and_runs_count_is_part_of_the_hash(tmp_path):
    root, paths, g = prepare(tmp_path, n=1500, seed=7)
    mk = lambda: Runner(paths.results("moon"), CFG, body="moon", runs=3, base_dir=root, log=lambda m: None)  # noqa: E731
    st = lambda r: s5.stage(CFG, "moon", paths, runs=r, workers=1, log=lambda m: None)  # noqa: E731
    assert [r.action for r in mk().run([st(3)])] == ["ran"] and [r.action for r in mk().run([st(3)])] == ["skipped"]
    assert [r.action for r in mk().run([st(4)])] == ["ran"]  # --runs changes the hash


def test_cli_has_workers_and_runs_options():
    from typer.testing import CliRunner

    from gsop.cli import app

    out = CliRunner().invoke(app, ["s5", "simulate", "--help"]).output
    assert "--workers" in out and "--runs" in out and "--target-check" in out


def test_memory_chunk_rows_is_bounded_and_does_not_change_scores():
    r = s5.memory_chunk_rows(CFG, 14)
    assert 4096 <= r <= 262_144
