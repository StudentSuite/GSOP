"""The committed golden slice (issue C29): size, input hashes, structure, and that `compare` notices what it should. The end-to-end run is tests/test_golden.py."""

import copy
import json

import pytest

from gsop import golden as G
from gsop.config import load_config

CFG = load_config(G.REPO_ROOT / "config")
TOL, JAC = CFG.run.golden_score_tolerance, CFG.run.golden_top1_jaccard_min


@pytest.fixture(scope="module")
def expected():
    return G.read_golden()


def test_the_golden_folder_is_under_5_mb_and_has_the_files():
    size = sum(p.stat().st_size for p in G.GOLDEN_DIR.rglob("*") if p.is_file())
    assert size < 5_000_000, size
    for body in G.BODIES:
        assert all(p.is_file() for p in G.input_files(G.GOLDEN_DIR, body))
    assert G.EXPECTED.is_file()


def test_committed_inputs_match_the_recorded_hashes_and_config(expected):
    assert G.input_hashes(G.GOLDEN_DIR) == expected["inputs"]
    assert expected["config_hash"] == CFG.config_hash(G.CONFIG_PARTS), "config changed: rerun `gsop golden --update` and commit golden.json"
    assert expected["runs"] == CFG.run.golden_runs == 50 and expected["tiles"] == CFG.run.golden_tiles == 5000 and expected["master_seed"] == CFG.run.seed


def test_the_slice_has_5000_tiles_per_body_and_both_bodies_have_their_profiles(expected):
    import pandas as pd

    for body in G.BODIES:
        assert len(pd.read_parquet(G.input_files(G.GOLDEN_DIR, body)[0])) == 5000
    moon, mars = expected["bodies"]["moon"], expected["bodies"]["mars"]
    assert set(moon["top1"]) == {"connecting_ridge_c1", "shackleton_rim"} and set(mars["top1"]) == {"jezero"}
    assert set(moon["top1"]["shackleton_rim"]) == {"base", "landing_mobility", "power_night", "ice_resources"}
    assert set(mars["top1"]["jezero"]) == {"base", "landing_mobility", "power_night", "ice_resources", "geology_life"}
    for body in (moon, mars):
        for t, n in body["gated"].items():
            assert 0 < n <= 5000 and len(body["scores"][t]["tile_id"]) == n
            assert all(len(v) > 0 for v in body["top1"][t].values())
    assert json.loads(G.EXPECTED.read_text(encoding="utf-8")) == expected  # plain JSON, no NaN


def test_compare_accepts_the_file_against_itself_and_small_noise(expected):
    assert G.compare(expected, expected, TOL, JAC) == []
    noisy = copy.deepcopy(expected)
    col = noisy["bodies"]["moon"]["scores"]["shackleton_rim"]["score_base"]
    noisy["bodies"]["moon"]["scores"]["shackleton_rim"]["score_base"] = [None if v is None else v + TOL / 2 for v in col]
    assert G.compare(expected, noisy, TOL, JAC) == []


def test_compare_reports_each_kind_of_difference_with_its_size(expected):
    bad = copy.deepcopy(expected)
    s = bad["bodies"]["mars"]["scores"]["jezero"]
    s["score_ice_resources"][3] = (s["score_ice_resources"][3] or 0.0) + 0.5
    bad["bodies"]["moon"]["top1"]["shackleton_rim"]["base"] = list(range(10_000, 10_000 + len(bad["bodies"]["moon"]["top1"]["shackleton_rim"]["base"])))
    first_profile = next(iter(bad["bodies"]["moon"]["mc"]["shackleton_rim"]))
    bad["bodies"]["moon"]["mc"]["shackleton_rim"][first_profile]["mc_mean"][0] += 1.0
    bad["inputs"]["moon/earth_features_6km.parquet"] = "0" * 64
    bad["config_hash"] = "x"
    fails = G.compare(expected, bad, TOL, JAC)
    by_check = {f.check: f for f in fails}
    assert set(by_check) == {"score", "top1_jaccard", "mc_mean", "input_hash", "config_hash"}
    assert by_check["score"].where == "mars/jezero/score_ice_resources" and "0.5" in by_check["score"].detail and "tolerance" in by_check["score"].detail
    assert "Jaccard 0.0000" in by_check["top1_jaccard"].detail
    assert str(by_check["mc_mean"]).startswith("mc_mean [moon/shackleton_rim/")


def test_compare_flags_a_different_gated_set_and_missing_bodies(expected):
    bad = copy.deepcopy(expected)
    t = bad["bodies"]["mars"]["scores"]["jezero"]
    for k in t:
        t[k] = t[k][1:]
    only_moon = copy.deepcopy(expected)
    del only_moon["bodies"]["mars"]
    checks = {f.check for f in G.compare(expected, bad, TOL, JAC)} | {f.check for f in G.compare(expected, only_moon, TOL, JAC)}
    assert {"gated_tiles", "bodies"} <= checks


def test_top_fraction_ids_rule():
    import numpy as np

    ids = np.arange(10)
    s = np.array([5, 9, 9, np.nan, 1, 9, 2, 3, 4, 0], dtype="float64")
    assert G.top_fraction_ids(ids, s, 0.1) == [1]  # ceil(0.1 * 9) = 1, tie among 1, 2 and 5 goes to the lowest tile id
    assert G.top_fraction_ids(ids, s, 0.3) == [1, 2, 5]
    assert G.top_fraction_ids(ids, np.full(10, np.nan), 0.1) == []
