import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import stage_inputs as SI
import synth
from gsop import gates as G
from gsop.cli import app
from gsop.config import load_config
from gsop.paths import Paths
from gsop.runner import Runner
from gsop.similarity import SetupError
from gsop.stages import s3

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def run(tmp, body, **kw):
    paths = Paths(tmp)
    return s3.run_gates(CFG, body, paths.earth_parquet, paths.targets_parquet(body), paths.results(body), batch_rows=777, log=lambda m: None, **kw)


def test_moon_tables_summary_and_the_planted_traps(tmp_path):
    root, g, rows = SI.write_inputs(tmp_path, "moon")
    summary = run(root, "moon")
    out = Paths(root).results("moon")
    passed, rejected = pd.read_parquet(out / "gates_passed.parquet"), pd.read_parquet(out / "gates_rejected.parquet")
    assert set(summary["targets"]) == {"connecting_ridge_c1", "shackleton_rim"} and summary["unscored"] == []
    for t, c in summary["targets"].items():
        p, r = passed[passed.target == t], rejected[rejected.target == t]
        assert len(p) + len(r) == len(g.df) == c["passed"] + c["rejected"] == c["tiles"]  # passed + rejected = all land tiles
        assert set(p.tile_id).isdisjoint(set(r.tile_id)) and p.tile_id.is_unique and r.tile_id.is_unique
        # the three gate traps and the three missing-data tiles are rejected with the right bits; the analog passes
        # (the stored mask also carries the landing slope bit, informative for the landing profile; compare the base bits)
        by = dict(zip(r.tile_id, (r.reasons & np.uint8(G.BASE_MASK)).astype(np.uint8)))
        assert by[g.truth["gate_trap"]["water"]] == G.BITS["water_ice"] and by[g.truth["gate_trap"]["ndvi"]] == G.BITS["vegetation"]
        assert by[g.truth["gate_trap"]["builtup"]] == G.BITS["builtup_cropland"]
        assert all(by[tid] & G.BITS["missing_data"] for tid in g.truth["missing"].values())
        assert g.truth["analog"] in set(p.tile_id) and g.truth["analog"] not in by
        assert c["reasons"]["WATER_OR_ICE"] >= 1 and c["reasons"]["MISSING_DATA"] == 3
    assert (out / "gates_summary.json").is_file()
    on_disk = json.loads((out / "gates_summary.json").read_text(encoding="utf-8"))
    assert on_disk["bit_table"] == G.BITS and on_disk["reason_codes"]["missing_data"] == "MISSING_DATA"


def test_landing_ok_marks_tiles_that_pass_other_profiles_but_not_landing(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "moon")
    run(root, "moon")
    passed = pd.read_parquet(Paths(root).results("moon") / "gates_passed.parquet")
    p = passed[passed.target == "connecting_ridge_c1"].merge(g.df[["tile_id", "slope_med"]], on="tile_id")
    assert ((p.slope_med <= 8.0) == p.landing_ok).all() and (~p.landing_ok).any() and p.landing_ok.any()


def test_mars_labels_every_row_when_the_target_fails_its_own_gate_but_still_scores_it(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "mars", mars_elevation=2400.0)
    summary = run(root, "mars")
    assert summary["unscored"] == ["arcadia"] and set(summary["targets"]) == {"jezero"}
    c = summary["targets"]["jezero"]
    assert c["fails_own_gate"] is True and c["own_gate_reasons"] == ["MARS_TARGET_CHECK:ELEVATION"]
    out = Paths(root).results("mars")
    for name in ("gates_passed.parquet", "gates_rejected.parquet"):
        df = pd.read_parquet(out / name)
        assert df.fails_own_gate.all() and (df.own_gate_reasons == "MARS_TARGET_CHECK:ELEVATION").all()
    assert summary["validation"][0]["fails_own_gate"] is True and summary["validation"][0]["scored"] is True


def test_mars_strict_mode_skips_a_failing_target(tmp_path):
    root, _, _ = SI.write_inputs(tmp_path, "mars", mars_elevation=2400.0)
    summary = run(root, "mars", target_check="strict")
    assert summary["not_scored_by_target_check"] == ["jezero"] and summary["targets"] == {} and summary["target_check_mode"] == "reject"


def test_a_passing_mars_target_has_clean_flags(tmp_path):
    root, _, _ = SI.write_inputs(tmp_path, "mars")
    run(root, "mars")
    df = pd.read_parquet(Paths(root).results("mars") / "gates_passed.parquet")
    assert not df.fails_own_gate.any() and (df.own_gate_reasons == "").all()


def test_a_family_with_no_target_value_stops_the_stage_with_a_clear_message(tmp_path):
    root, _, _ = SI.write_inputs(tmp_path, "mars", drop_columns=("silica_index",))
    with pytest.raises(SetupError, match="composition"):
        run(root, "mars")
    assert not (Paths(root).results("mars") / "gates_passed.parquet").exists()  # nothing half-written is left behind


def test_the_runner_skips_a_rerun_and_reruns_when_config_changes(tmp_path):
    root, _, _ = SI.write_inputs(tmp_path, "moon")
    paths = Paths(root)
    st = lambda: s3.stage(CFG, "moon", paths, log=lambda m: None)  # noqa: E731
    runner = lambda: Runner(paths.results("moon"), CFG, body="moon", base_dir=root, log=lambda m: None)  # noqa: E731
    assert [r.action for r in runner().run([st()])] == ["ran"]
    assert [r.action for r in runner().run([st()])] == ["skipped"]
    import shutil

    cfg2_dir = tmp_path / "cfg"
    shutil.copytree(Path(__file__).resolve().parents[1] / "config", cfg2_dir)
    g = cfg2_dir / "gates.yaml"
    g.write_text(g.read_text(encoding="utf-8").replace("ndvi_median: 0.2 ", "ndvi_median: 0.25"), encoding="utf-8")
    cfg2 = load_config(cfg2_dir)
    r2 = Runner(paths.results("moon"), cfg2, body="moon", base_dir=root, log=lambda m: None).run([s3.stage(cfg2, "moon", paths, log=lambda m: None)])
    assert [r.action for r in r2] == ["ran"]


def test_strict_flag_changes_the_stage_hash(tmp_path):
    root, _, _ = SI.write_inputs(tmp_path, "mars")
    paths = Paths(root)
    a, b = s3.stage(CFG, "mars", paths), s3.stage(CFG, "mars", paths, "strict")
    assert a.code_version != b.code_version


def test_cli_reports_clear_errors(tmp_path):
    result = CliRunner().invoke(app, ["s3", "gates", "--body", "moon"])
    if not Paths().earth_parquet.exists():
        assert result.exit_code == 1 and "missing input" in result.output
