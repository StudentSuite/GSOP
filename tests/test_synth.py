import numpy as np
import pandas as pd
import pytest
import synth

from gsop.config import load_config
from pathlib import Path

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def row(grid, tile_id):
    return grid.df[grid.df.tile_id == tile_id].iloc[0]


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_deterministic_for_a_seed_and_different_for_another(body):
    a = synth.make_grid(5000, seed=3, body=body)
    b = synth.make_grid(5000, seed=3, body=body)
    c = synth.make_grid(5000, seed=4, body=body)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert a.truth == b.truth
    assert not a.df.equals(c.df)


def test_default_size_columns_and_dtypes():
    g = synth.make_grid()
    assert len(g.df) == 20_000 == synth.DEFAULT_N
    expected = set(CFG.features.columns) | set(CFG.features.gate_layers)
    assert expected <= set(g.df.columns)
    assert {"tile_id", "lat", "lon"} <= set(g.df.columns)
    assert g.df.tile_id.is_unique and sorted(g.df.tile_id)[-1] == 19_999
    assert g.df["slope_med"].dtype == np.float32
    assert g.df.lat.between(-90, 90).all() and g.df.lon.between(-180, 180).all()


def test_scales_to_a_large_grid():
    g = synth.make_grid(300_000, seed=1, body="mars")
    assert len(g.df) == 300_000


def test_rows_are_shuffled_but_ids_are_fixed():
    g = synth.make_grid(5000, seed=0)
    assert not (g.df.tile_id.to_numpy()[:20] == np.arange(20)).all()


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_layout_matches_the_docstring(body):
    t = synth.planted_ids(body)
    nfam = len(CFG.weights(body))
    assert t["analog"] == 0
    assert list(t["single_family"].values()) == list(range(1, nfam + 1))
    assert list(t["gate_trap"].values()) == [nfam + 1, nfam + 2, nfam + 3]
    assert list(t["missing"].values()) == [nfam + 4, nfam + 5, nfam + 6]
    assert t["near_miss"] == [nfam + 7, nfam + 8, nfam + 9]
    assert t["n_planted"] == nfam + 10
    assert f"K = {nfam + 10}" in synth.__doc__
    assert "0     ANALOG" in synth.__doc__


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_analog_equals_target_on_every_column_and_passes_gates(body):
    g = synth.make_grid(5000, seed=2, body=body)
    r = row(g, 0)
    for col, val in zip(g.target.column, g.target.value):
        assert r[col] == pytest.approx(val, rel=1e-6)
    thr = CFG.gates.thresholds
    assert r.water_ice_fraction <= thr["water_ice_fraction"]
    assert r.ndvi_median <= thr["ndvi_median"]
    assert r.builtup_cropland_fraction <= thr["builtup_cropland_fraction"]


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_single_family_tiles_match_only_their_family(body):
    g = synth.make_grid(5000, seed=2, body=body)
    tvalue = dict(zip(g.target.column, g.target.value))
    tspread = dict(zip(g.target.column, g.target.spread))
    for fam, tid in g.truth["single_family"].items():
        r = row(g, tid)
        fam_cols = set(CFG.weights(body)[fam].columns)
        for col in g.used_columns:
            close = abs(r[col] - tvalue[col]) <= tspread[col]
            if col in fam_cols:
                assert close, (fam, col)
            elif col not in fam_cols:
                assert not close, (fam, col)


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_gate_traps_equal_target_but_break_exactly_one_gate(body):
    g = synth.make_grid(5000, seed=2, body=body)
    thr = CFG.gates.thresholds
    for kind, tid in g.truth["gate_trap"].items():
        r = row(g, tid)
        for col, val in zip(g.target.column, g.target.value):
            assert r[col] == pytest.approx(val, rel=1e-6)
        broken = {
            "water": r.water_ice_fraction > thr["water_ice_fraction"],
            "ndvi": r.ndvi_median > thr["ndvi_median"],
            "builtup": r.builtup_cropland_fraction > thr["builtup_cropland_fraction"],
        }
        assert broken[kind] and sum(broken.values()) == 1


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_missing_tiles_have_one_nan_in_a_used_column(body):
    g = synth.make_grid(5000, seed=2, body=body)
    used = synth.used_columns(body)
    assert list(g.truth["missing"]) == [used[0], used[len(used) // 2], used[-1]]
    for col, tid in g.truth["missing"].items():
        r = row(g, tid)
        assert r[g.used_columns].isna().sum() == 1 and np.isnan(r[col])


def test_near_misses_are_ordered_by_distance_from_the_target():
    g = synth.make_grid(5000, seed=2, body="moon")
    tv = g.target.set_index("column")
    dist = []
    for tid in g.truth["near_miss"]:
        r = row(g, tid)
        dist.append(float(sum(abs(r[c] - tv.value[c]) / tv.spread[c] for c in g.used_columns)))
    assert dist == sorted(dist) and dist[0] > 0


def test_background_is_far_from_the_target_and_passes_gates():
    g = synth.make_grid(20_000, seed=5, body="moon")
    bg = g.df[g.df.tile_id >= g.truth["n_planted"]]
    tv = g.target.set_index("column")
    z = np.zeros(len(bg))
    for c in g.used_columns:
        sd = synth.BACKGROUND[c][1]
        z += ((bg[c] - tv.value[c]) / sd) ** 2
    assert np.sqrt(z / len(g.used_columns)).min() > 1.0
    thr = CFG.gates.thresholds
    assert (bg.water_ice_fraction <= thr["water_ice_fraction"]).all()
    assert (bg.ndvi_median <= thr["ndvi_median"]).all()


def test_target_profiles_exist_for_both_bodies():
    moon, mars = synth.target_profile("moon"), synth.target_profile("mars")
    assert set(moon.column) == set(synth.used_columns("moon"))
    assert set(mars.column) == set(synth.used_columns("mars"))
    assert "air_density" in set(mars.column) and "air_density" not in set(moon.column)
    assert (moon.spread > 0).all() and (mars.spread > 0).all()
    assert set(moon.source) == {"synthetic"}
    with pytest.raises(ValueError):
        synth.target_profile("venus")


def test_too_small_grid_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        synth.make_grid(10)
