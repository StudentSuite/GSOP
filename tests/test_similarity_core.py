from pathlib import Path

import numpy as np
import pytest

import synth
from gsop import similarity as S
from gsop import weights as W
from gsop.config import ConfigError, load_config
from gsop.targets import TargetProfile

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def profile_for(body, drop=()):
    t = synth.target_profile(body, CFG)
    return TargetProfile("synthetic", dict(zip(t.column, t.value)), dict(zip(t.column, t.spread)), frozenset(drop))


def setup_for(body, **kw):
    prof = profile_for(body, **kw)
    s = S.make_setup(CFG, body, prof, policy=kw.get("policy"))
    w = S.renormalise(W.weight_vector(W.derive_weights(CFG, body)), s.families)
    return s.with_sigma(np.maximum(s.target_spread, 1.0)), w


def test_exact_target_gives_one_per_column_and_100_overall():
    s, w = setup_for("moon")
    x = s.target_values.astype(np.float32)[None, :]
    cols = S.column_similarity(x, s.target_values, s.sigma)
    assert np.allclose(cols, 1.0)
    res = S.score_tiles(x, s.target_values, s.sigma, s.family_cols, w, return_family=True)
    assert res.score[0] == pytest.approx(100.0, abs=1e-3) and np.allclose(res.family, 1.0) and res.family.shape == (1, len(s.families))


def test_far_values_are_tiny_and_never_nan_or_inf():
    s, w = setup_for("moon")
    far = np.stack([s.target_values + 1e6 * s.sigma, s.target_values + 1e30 * s.sigma, s.target_values + 14 * s.sigma]).astype(np.float32)
    cols = S.column_similarity(far, s.target_values, s.sigma)
    assert np.isfinite(cols).all() and (cols >= 0).all() and cols.max() < 1e-30
    score = S.score_tiles(far, s.target_values, s.sigma, s.family_cols, w).score
    assert np.isfinite(score).all() and (score < 1e-20).all()


def test_missing_values_propagate_as_missing_not_zero():
    s, w = setup_for("moon")
    x = np.repeat(s.target_values.astype(np.float32)[None, :], 3, axis=0)
    x[1, 0] = np.nan  # a slope column of tile 1
    res = S.score_tiles(x, s.target_values, s.sigma, s.family_cols, w, return_family=True)
    assert res.score[0] == pytest.approx(100, abs=1e-3) and np.isnan(res.score[1]) and res.score[2] == pytest.approx(100, abs=1e-3)
    assert np.isnan(res.family[1, 0]) and np.isfinite(res.family[1, 1:]).all()


def test_a_zero_weight_family_cannot_poison_the_score():
    s, w = setup_for("moon")
    w2 = w.copy()
    w2[0] = 0.0
    w2 = w2 / w2.sum()
    x = np.repeat(s.target_values.astype(np.float32)[None, :], 1, axis=0)
    x[0, s.family_cols[0][0]] = np.nan
    assert S.score_tiles(x, s.target_values, s.sigma, s.family_cols, w2).score[0] == pytest.approx(100, abs=1e-3)


def test_chunking_does_not_change_the_result():
    s, w = setup_for("mars")
    x = np.random.default_rng(1).normal(size=(1000, len(s.columns))).astype(np.float32) * 3 + s.target_values.astype(np.float32)
    a = S.score_tiles(x, s.target_values, s.sigma, s.family_cols, w, chunk_rows=1000).score
    b = S.score_tiles(x, s.target_values, s.sigma, s.family_cols, w, chunk_rows=37).score
    assert np.array_equal(a, b)


def test_output_is_float32_and_the_same_function_serves_the_inner_loop():
    s, w = setup_for("moon")
    x = np.random.default_rng(2).normal(size=(50, len(s.columns))).astype(np.float32)
    r = S.score_tiles(x, s.target_values + 0.5, s.sigma, s.family_cols, w * 0 + 1.0 / len(w))  # other target values and weights per call
    assert r.score.dtype == np.float32 and r.family is None


def test_setup_collects_the_columns_the_weights_use():
    s, _ = setup_for("moon")
    assert set(s.columns) == set(synth.used_columns("moon", CFG)) and s.families == tuple(CFG.weights("moon"))
    assert s.dropped == {"columns": {}, "families": []}
    fam = dict(zip(s.families, s.family_cols))
    assert [s.columns[i] for i in fam["slope"]] == ["slope_med", "slope_p90"]


def test_a_missing_target_column_is_dropped_inside_its_family():
    s, w = setup_for("mars", drop=("dust_index",))
    assert "dust_index" not in s.columns and s.dropped["columns"] == {"thermal_inertia": ["dust_index"]}
    fam = dict(zip(s.families, s.family_cols))
    assert [s.columns[i] for i in fam["thermal_inertia"]] == ["ti_proxy"]  # the family keeps its weight and uses ti_proxy alone


def test_a_family_without_any_target_value_follows_the_policy():
    with pytest.raises(S.SetupError, match="no usable target value.*composition"):
        S.make_setup(CFG, "mars", profile_for("mars", drop=("silica_index",)))
    s = S.make_setup(CFG, "mars", profile_for("mars", drop=("silica_index",)), policy="drop_and_renormalise")
    assert "composition" not in s.families and s.dropped["families"] == ["composition"]
    w = S.renormalise(W.weight_vector(W.derive_weights(CFG, "mars")), s.families)
    assert w.sum() == pytest.approx(1.0) and len(w) == len(s.families) == 8
    with pytest.raises(S.SetupError, match="unknown policy"):
        S.make_setup(CFG, "mars", profile_for("mars"), policy="maybe")


def test_default_policy_is_error_from_config():
    assert CFG.run.unavailable_target_families == "error" and CFG.run.sigma_iqr_multiplier == 0.25
