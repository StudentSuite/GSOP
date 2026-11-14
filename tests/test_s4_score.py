import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stage_inputs as SI
from gsop import similarity as S
from gsop.config import load_config
from gsop.paths import Paths
from gsop.runner import Runner
from gsop.stages import s3, s4

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def run_both(tmp, body, **kw):
    paths = Paths(tmp)
    s3.run_gates(CFG, body, paths.earth_parquet, paths.targets_parquet(body), paths.results(body), log=lambda m: None, **{k: v for k, v in kw.items() if k == "target_check"})
    return s4.run_scores(CFG, body, paths.earth_parquet, paths.targets_parquet(body), paths.results(body) / "gates_passed.parquet", paths.results(body),
                         log=lambda m: None, **kw)


@pytest.fixture(scope="module")
def moon(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("moon")
    root, g, _ = SI.write_inputs(tmp, "moon", n=5000, seed=1)
    summary = run_both(root, "moon", chunk_rows=1000)
    return root, g, summary, pd.read_parquet(Paths(root).results("moon") / "scores_base.parquet")


def test_planted_analog_ranks_first_on_the_base_score_for_every_moon_target(moon):
    root, g, summary, df = moon
    for target in ("connecting_ridge_c1", "shackleton_rim"):
        d = df[df.target == target].sort_values("score_base", ascending=False)
        assert d.iloc[0].tile_id == g.truth["analog"] and d.iloc[0].score_base == pytest.approx(100.0, abs=1e-2)
        # next come the near misses, in order of closeness
        assert list(d.tile_id.iloc[1:4]) == g.truth["near_miss"], list(d.tile_id.iloc[1:4])
        # gate traps and missing-data tiles are absent from the scored set
        scored = set(d.tile_id)
        assert scored.isdisjoint(set(g.truth["gate_trap"].values())) and scored.isdisjoint(set(g.truth["missing"].values()))
        # the one-family matcher scores well below the analog
        for fam, tid in g.truth["single_family"].items():
            assert d.set_index("tile_id").loc[tid, "score_base"] < 60


def test_schema_sorted_ids_and_profiles(moon):
    root, g, summary, df = moon
    fams = list(CFG.weights("moon"))
    expected = ["target", "tile_id", *[f"sim_{f}" for f in fams], "illumination_match", "score_base", "score_landing_mobility", "score_power_night",
                "score_ice_resources", "fails_own_gate", "own_gate_reasons"]
    assert list(df.columns) == expected and "score_geology_life" not in df.columns
    for t, sub in df.groupby("target"):
        assert sub.tile_id.is_monotonic_increasing and sub.tile_id.is_unique
    assert df.dtypes["score_base"] == np.float32 and df.dtypes["sim_slope"] == np.float32
    assert (df.illumination_match == df.sim_illumination).all()


def test_landing_profile_is_nan_where_the_extra_gate_fails(moon):
    root, g, summary, df = moon
    merged = df[df.target == "connecting_ridge_c1"].merge(g.df[["tile_id", "slope_med"]], on="tile_id")
    assert merged.score_landing_mobility.isna().equals(merged.slope_med > 8.0)
    assert merged.score_base.notna().all() and merged.score_power_night.notna().all()


def test_profiles_reweight_the_same_family_similarities(moon):
    root, g, summary, df = moon
    d = df[df.target == "connecting_ridge_c1"]
    w = json.loads((Paths(root).results("moon") / "weights_used.json").read_text(encoding="utf-8"))["targets"]["connecting_ridge_c1"]["profiles"]
    fams = list(CFG.weights("moon"))
    for profile in ("base", "power_night", "ice_resources"):
        expected = 100.0 * sum(w[profile]["weights"][f] * d[f"sim_{f}"].to_numpy("float64") for f in fams)
        assert np.allclose(d[f"score_{profile}"], expected, atol=2e-3), profile
        assert sum(w[profile]["weights"].values()) == pytest.approx(1.0)
    assert w["landing_mobility"]["extra_gate"] == "slope_med <= 8" and w["base"]["extra_gate"] is None


def test_sigma_table_weights_and_summary_files(moon):
    root, g, summary, df = moon
    out = Paths(root).results("moon")
    sigma = json.loads((out / "sigma_table.json").read_text(encoding="utf-8"))
    assert set(sigma["targets"]) == {"connecting_ridge_c1", "shackleton_rim"} and sigma["iqr_multiplier"] == 0.25
    slope = sigma["targets"]["connecting_ridge_c1"]["slope_med"]
    assert slope["sigma"] == max(slope["target_spread"], slope["measurement_uncertainty"], slope["iqr_term"]) and slope["measurement_uncertainty"] == 2.5
    used = json.loads((out / "weights_used.json").read_text(encoding="utf-8"))
    assert sum(used["derived_family_weights"].values()) == pytest.approx(1.0) and used["targets"]["shackleton_rim"]["dropped"] == {"columns": {}, "families": []}
    s = json.loads((out / "scores_summary.json").read_text(encoding="utf-8"))
    assert set(s["timing"]) == {"connecting_ridge_c1", "shackleton_rim"} and s["timing"]["connecting_ridge_c1"]["peak_rss_mb"] > 0 and s["timing"]["connecting_ridge_c1"]["seconds"] >= 0


def test_mars_has_four_profiles_plus_base_confidence_flags_and_the_target_label(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "mars", n=4000, seed=2, mars_elevation=2400.0)
    run_both(root, "mars")
    out = Paths(root).results("mars")
    df = pd.read_parquet(out / "scores_base.parquet")
    assert {c for c in df.columns if c.startswith("score_")} == {"score_base", "score_landing_mobility", "score_power_night", "score_ice_resources", "score_geology_life"}
    assert (df.target == "jezero").all() and df.fails_own_gate.all() and (df.own_gate_reasons == "MARS_TARGET_CHECK:ELEVATION").all()
    used = json.loads((out / "weights_used.json").read_text(encoding="utf-8"))["targets"]["jezero"]["profiles"]
    assert used["landing_mobility"]["confidence"] == "low" and used["power_night"]["confidence"] == "low" and used["ice_resources"]["confidence"] is None
    assert used["landing_mobility"]["extra_gate"] == "slope_p90 <= 30"
    top = df.sort_values("score_base", ascending=False).iloc[0]
    assert top.tile_id == g.truth["analog"] and top.illumination_match == pytest.approx(1.0, abs=1e-3)  # sunlight family of the analog


def test_output_does_not_depend_on_the_row_order_of_the_input(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "moon", n=3000, seed=3)
    a = run_both(root, "moon")
    first = pd.read_parquet(Paths(root).results("moon") / "scores_base.parquet")
    e = pd.read_parquet(Paths(root).earth_parquet).sample(frac=1.0, random_state=9)
    e.to_parquet(Paths(root).earth_parquet)
    run_both(root, "moon")
    second = pd.read_parquet(Paths(root).results("moon") / "scores_base.parquet")
    pd.testing.assert_frame_equal(first, second)


def test_runner_skips_a_second_run_and_chunking_does_not_change_scores(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "moon", n=2500, seed=4)
    paths = Paths(root)
    s3.run_gates(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon"), log=lambda m: None)
    st = s4.stage(CFG, "moon", paths, log=lambda m: None)
    mk = lambda: Runner(paths.results("moon"), CFG, body="moon", base_dir=root, log=lambda m: None)  # noqa: E731
    assert [r.action for r in mk().run([st])] == ["ran"] and [r.action for r in mk().run([st])] == ["skipped"]
    a = pd.read_parquet(paths.results("moon") / "scores_base.parquet")
    s4.run_scores(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), paths.results("moon") / "gates_passed.parquet", paths.results("moon"), chunk_rows=97, log=lambda m: None)
    pd.testing.assert_frame_equal(a, pd.read_parquet(paths.results("moon") / "scores_base.parquet"))


def test_nothing_to_score_is_an_error_not_an_empty_file(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "mars", n=500, seed=5, mars_elevation=2400.0)
    paths = Paths(root)
    s3.run_gates(CFG, "mars", paths.earth_parquet, paths.targets_parquet("mars"), paths.results("mars"), target_check="strict", log=lambda m: None)
    with pytest.raises(s4.ScoreError, match="no target had any tile to score"):
        s4.run_scores(CFG, "mars", paths.earth_parquet, paths.targets_parquet("mars"), paths.results("mars") / "gates_passed.parquet", paths.results("mars"),
                      target_check="strict", log=lambda m: None)
    assert not (paths.results("mars") / "scores_base.parquet").exists()
