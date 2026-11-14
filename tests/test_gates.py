"""Gate tests on synthetic tiles (issue C11).

Boundary behaviour, documented here and in SPEC.md: every gate rejects only a value STRICTLY beyond its threshold. A value equal to the
threshold passes: water/ice 0.5, NDVI 0.2, built-up/farmed 0.2, Moon slope_med 8, Mars slope_p90 30 all pass at exactly the limit.
"""

from pathlib import Path

import numpy as np
import pytest

import synth
from gsop import gates as G
from gsop.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
T = CFG.gates.thresholds


def tiles(n=1, **over):
    cols = {"water_ice_fraction": np.zeros(n), "ndvi_median": np.zeros(n), "builtup_cropland_fraction": np.zeros(n), "slope_med": np.full(n, 1.0),
            "slope_p90": np.full(n, 2.0), "relief": np.full(n, 50.0)}
    cols.update({k: np.broadcast_to(np.asarray(v, dtype="float64"), (n,)).copy() for k, v in over.items()})
    return cols


def codes(cols, body="moon", required=("slope_med", "relief"), landing=False):
    r = G.evaluate_gates(CFG, body, cols, list(required))
    return [G.decode(int(m), CFG, applies_landing_gate=landing) for m in r.reasons]


@pytest.mark.parametrize("override,expected", [
    ({"water_ice_fraction": 0.9}, ["WATER_OR_ICE"]),
    ({"ndvi_median": 0.7}, ["NDVI_HIGH"]),
    ({"builtup_cropland_fraction": 0.6}, ["BUILTUP_OR_CROPLAND"]),
])
def test_a_tile_violating_exactly_one_gate_gets_exactly_that_reason(override, expected):
    assert codes(tiles(**override)) == [expected]


def test_the_landing_slope_gate_rejects_only_for_the_landing_profile():
    cols = tiles(slope_med=12.0)
    assert codes(cols, landing=True) == [["LANDING_SLOPE"]]
    r = G.evaluate_gates(CFG, "moon", cols, ["slope_med"])
    assert r.passed(applies_landing_gate=False)[0] and not r.passed(applies_landing_gate=True)[0]  # base profile keeps it, landing drops it
    assert codes(cols, landing=False) == [[]]


def test_missing_data_is_rejected_with_missing_data():
    assert codes(tiles(slope_med=np.nan)) == [["MISSING_DATA"]]
    assert codes(tiles(relief=np.nan)) == [["MISSING_DATA"]]
    assert codes(tiles(relief=np.inf)) == [["MISSING_DATA"]]  # not a usable number either
    # a column the weights do not use may be missing
    assert codes(tiles(relief=np.nan), required=["slope_med"]) == [[]]


def test_a_missing_gate_layer_is_missing_data_not_a_pass():
    for layer in ("water_ice_fraction", "ndvi_median", "builtup_cropland_fraction"):
        assert codes(tiles(**{layer: np.nan})) == [["MISSING_DATA"]], layer


def test_a_tile_violating_two_gates_carries_both_reasons():
    got = codes(tiles(water_ice_fraction=0.9, ndvi_median=0.8))
    assert got == [["WATER_OR_ICE", "NDVI_HIGH"]]
    got = codes(tiles(water_ice_fraction=0.9, ndvi_median=0.8, builtup_cropland_fraction=0.9, slope_med=np.nan), landing=True)
    assert set(got[0]) == {"WATER_OR_ICE", "NDVI_HIGH", "BUILTUP_OR_CROPLAND", "MISSING_DATA"}
    cols = tiles(water_ice_fraction=0.9, slope_med=30.0)
    assert set(codes(cols, landing=True)[0]) == {"WATER_OR_ICE", "LANDING_SLOPE"}


@pytest.mark.parametrize("body,column,limit", [("moon", "slope_med", 8.0), ("mars", "slope_p90", 30.0)])
def test_purpose_extra_gate_boundary_equal_passes(body, column, limit):
    at = tiles(**{column: limit})
    above = tiles(**{column: np.nextafter(limit, np.inf)})
    below = tiles(**{column: limit - 0.001})
    req = [column]
    assert G.evaluate_gates(CFG, body, at, req).passed(True)[0], "equal passes"
    assert not G.evaluate_gates(CFG, body, above, req).passed(True)[0]
    assert G.evaluate_gates(CFG, body, below, req).passed(True)[0]


def test_base_gate_boundaries_equal_passes():
    cols = tiles(water_ice_fraction=T["water_ice_fraction"], ndvi_median=T["ndvi_median"], builtup_cropland_fraction=T["builtup_cropland_fraction"])
    assert codes(cols) == [[]]
    over = tiles(water_ice_fraction=np.nextafter(T["water_ice_fraction"], 1.0))
    assert codes(over) == [["WATER_OR_ICE"]]


def test_near_misses_are_tiles_that_pass_within_ten_percent():
    # NDVI 0.19 passes but is within 10% of 0.2; 0.15 is not near; 0.21 fails (and is not a near miss, it is a rejection)
    r = G.evaluate_gates(CFG, "moon", tiles(3, ndvi_median=[0.19, 0.15, 0.21]), ["slope_med"])
    assert [bool(m & G.BITS["vegetation"]) for m in r.near] == [True, False, False]
    assert [bool(m & G.BITS["vegetation"]) for m in r.reasons] == [False, False, True]
    # the landing slope gate on the Moon: 7.5 passes within 10% of 8
    r = G.evaluate_gates(CFG, "moon", tiles(3, slope_med=[7.5, 6.0, 8.5]), ["slope_med"])
    assert [bool(m & G.LANDING_BIT) for m in r.near] == [True, False, False]


def test_synthetic_grid_traps_are_rejected_and_the_planted_analog_passes():
    for body in ("moon", "mars"):
        g = synth.make_grid(5000, seed=2, body=body, cfg=CFG)
        cols = {c: g.df[c].to_numpy("float64") for c in g.df.columns}
        r = G.evaluate_gates(CFG, body, cols, g.used_columns)
        # the landing slope bit is evaluated for every tile but applied only by the landing profile; the synthetic target's slope is
        # above the Moon limit of 8, so the analog legitimately carries that bit and only that bit would matter for landing_mobility
        by_id = dict(zip(g.df.tile_id, (r.reasons & ~np.uint8(G.LANDING_BIT)).astype(np.uint8)))
        assert by_id[g.truth["analog"]] == 0
        assert by_id[g.truth["gate_trap"]["water"]] == G.BITS["water_ice"]
        assert by_id[g.truth["gate_trap"]["ndvi"]] == G.BITS["vegetation"]
        assert by_id[g.truth["gate_trap"]["builtup"]] == G.BITS["builtup_cropland"]
        for col, tid in g.truth["missing"].items():
            assert by_id[tid] == G.BITS["missing_data"], col
        for tid in g.truth["near_miss"] + list(g.truth["single_family"].values()):
            assert by_id[tid] == 0  # near misses and single-family tiles pass the gates
        # background tiles pass the base gates (their gate layers are drawn well inside the thresholds)
        bg = g.df.tile_id.to_numpy() >= g.truth["n_planted"]
        assert (r.reasons[bg] & (G.BASE_MASK & ~G.BITS["missing_data"]) == 0).all()


def test_counts_add_up_to_all_tiles():
    g = synth.make_grid(5000, seed=5, body="moon", cfg=CFG)
    cols = {c: g.df[c].to_numpy("float64") for c in g.df.columns}
    r = G.evaluate_gates(CFG, "moon", cols, g.used_columns)
    c = G.counts_by_reason(r, CFG)
    assert c["PASSED"] + c["REJECTED"] == len(g.df) == 5000
