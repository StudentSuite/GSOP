"""Checks on the real files in config/ (one section per config issue)."""

import warnings
from pathlib import Path

import yaml

from gsop import config as C

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def read(name):
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


# --- A07: weights.yaml ------------------------------------------------------
# Expected values typed in from setup prompt section A3, independent of the file.
EXPECTED_MOON = {
    "slope": (16, 1.0, (0.21, 0.39), 0.30, ["slope_med", "slope_p90"]),
    "illumination": (19, 0.5, (0.10, 0.24), 0.18, ["sun_max_elev", "sun_low_share"]),
    "cold_dry": (26, 0.3, (0.10, 0.20), 0.15, ["water_index"]),
    "roughness": (7, 0.9, (0.07, 0.25), 0.12, ["rough_a", "rough_b"]),
    "relief": (6, 1.0, (0.07, 0.18), 0.11, ["relief"]),
    "temperature": (16, 0.2, (0.04, 0.09), 0.06, ["t_mean", "t_coldest"]),
    "composition": (7, 0.4, (0.03, 0.09), 0.05, ["silica_index"]),
    "thermal_inertia": (2, 0.5, (0.02, 0.04), 0.03, ["ti_proxy"]),
}
EXPECTED_MARS = {
    "composition": (26, 0.7, (0.16, 0.30), 0.23, ["silica_index"]),
    "water": (22, 0.7, (0.12, 0.25), 0.19, ["water_index"]),
    "sunlight": (17, 0.7, (0.10, 0.20), 0.15, ["insol_annual", "sun_max_elev"]),
    "slope": (9, 1.0, (0.08, 0.16), 0.11, ["slope_med", "slope_p90"]),
    "thermal_inertia": (12, 0.6, (0.06, 0.12), 0.09, ["ti_proxy", "dust_index"]),
    "roughness": (7, 0.9, (0.05, 0.10), 0.08, ["rough_a", "rough_b"]),
    "relief": (6, 1.0, (0.05, 0.11), 0.08, ["relief"]),
    "elevation_density": (20, 0.15, (0.02, 0.05), 0.04, ["air_density"]),
    "temperature": (5, 0.5, (0.02, 0.05), 0.03, ["t_mean", "t_coldest"]),
}


def _parse_weights():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fam, alts = C._parse_weights(read("weights"), C._Warn(sink))
    assert sink == []
    return fam, alts


def test_weights_match_setup_prompt_a3():
    fam, _ = _parse_weights()
    for body, expected in (("moon", EXPECTED_MOON), ("mars", EXPECTED_MARS)):
        assert list(fam[body]) == list(expected), "family order must follow A3"
        for name, (ps, match, rng, pub, cols) in expected.items():
            f = fam[body][name]
            assert (f.paper_score, f.matchability, f.range, f.published) == (ps, match, rng, pub)
            assert list(f.columns) == cols


def test_moon_thermal_inertia_published_is_kept_for_d3():
    fam, _ = _parse_weights()
    assert fam["moon"]["thermal_inertia"].published == 0.03


def _derived(families):
    total = sum(f.paper_score for f in families.values())
    raw = {n: f.paper_score / total * f.matchability for n, f in families.items()}
    s = sum(raw.values())
    return {n: v / s for n, v in raw.items()}


def test_derived_weights_reproduce_the_setup_prompt_numbers():
    fam, _ = _parse_weights()
    moon = _derived(fam["moon"])
    mars = _derived(fam["mars"])
    for got, want in zip(moon.values(), [0.3042, 0.1806, 0.1483, 0.1198, 0.1141, 0.0608, 0.0532, 0.0190]):
        assert abs(got - want) < 6e-5
    for got, want in zip(mars.values(), [0.2289, 0.1937, 0.1497, 0.1132, 0.0906, 0.0792, 0.0755, 0.0377, 0.0314]):
        assert abs(got - want) < 6e-5
    for body, derived in (("moon", moon), ("mars", mars)):
        assert abs(sum(derived.values()) - 1) < 1e-12
        for name, value in derived.items():
            assert abs(value - fam[body][name].published) <= 0.012


def test_alt_weight_sets():
    _, alts = _parse_weights()
    assert alts["moon"]["liu2025"].weights == {
        "roughness": 0.39, "slope": 0.26, "cold_dry": 0.16, "illumination": 0.11, "composition": 0.08,
    }
    assert alts["moon"]["no_illumination"].rule
    assert alts["moon"]["equal"].rule
    assert alts["mars"]["mashal2025"].weights == {"composition": 0.75, "water": 0.125, "sunlight": 0.125}
    assert alts["mars"]["equal"].rule


def test_weights_file_has_header_comment():
    text = (CONFIG_DIR / "weights.yaml").read_text(encoding="utf-8")
    assert "derived = paper_score/sum(paper_score) * matchability, renormalised. Code computes it; do not hard-code." in text


# --- A08: purposes.yaml -----------------------------------------------------
def _parse_purposes():
    fam, _ = _parse_weights()
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = C._parse_purposes(read("purposes"), fam, C._Warn(sink))
    assert sink == []
    return out


def test_purpose_profiles_per_body():
    p = _parse_purposes()
    assert sorted(p["moon"]) == sorted(["base", "landing_mobility", "power_night", "ice_resources"])
    assert sorted(p["mars"]) == sorted(["base", "landing_mobility", "power_night", "ice_resources", "geology_life"])
    assert not any("life" in name for name in p["moon"]), "no Moon life-detection profile (D5)"


def test_purpose_multipliers_and_gates():
    p = _parse_purposes()
    assert p["moon"]["base"].multipliers == {}
    assert p["moon"]["landing_mobility"].multipliers == {"slope": 2, "roughness": 2, "relief": 2}
    assert p["moon"]["landing_mobility"].extra_gate.text == "slope_med <= 8"
    assert p["mars"]["landing_mobility"].extra_gate.text == "slope_p90 <= 30"
    assert p["moon"]["power_night"].multipliers == {"illumination": 2.5, "temperature": 2}
    assert p["mars"]["power_night"].multipliers == {"sunlight": 2.5, "temperature": 2}
    assert p["moon"]["ice_resources"].multipliers == {"cold_dry": 2, "thermal_inertia": 2, "temperature": 2}
    assert p["mars"]["ice_resources"].multipliers == {"water": 2, "thermal_inertia": 2, "temperature": 2}
    assert p["mars"]["geology_life"].multipliers == {"composition": 2, "water": 1.5}
    assert p["mars"]["base"].extra_gate is None


def test_confidence_low_only_on_mars_landing_and_power():
    p = _parse_purposes()
    low = {(b, n) for b in p for n, pur in p[b].items() if pur.confidence == "low"}
    assert low == {("mars", "landing_mobility"), ("mars", "power_night")}


def test_extra_gate_passes():
    gate = _parse_purposes()["moon"]["landing_mobility"].extra_gate
    assert gate.passes(8.0) and not gate.passes(8.01)


# --- A09: gates.yaml --------------------------------------------------------
def test_gates_values_and_reason_codes():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = C._parse_gates(read("gates"), C._Warn(sink))
    assert sink == []
    t = g.thresholds
    assert t["water_ice_fraction"] == 0.5
    assert t["ndvi_median"] == 0.2
    assert t["builtup_cropland_fraction"] == 0.2
    assert t["mars_target_check"] == {
        "mode": "warn", "max_abs_lat_deg": 50, "max_elevation_m": 2000, "min_thermal_inertia": 100,
    }
    assert set(g.reason_codes) >= {
        "water_ice", "vegetation", "builtup_cropland", "landing_mobility_slope", "missing_data", "mars_target_check",
    }
    assert len(set(g.reason_codes.values())) == len(g.reason_codes)


def test_gate_file_flags_unsourced_thresholds():
    text = (CONFIG_DIR / "gates.yaml").read_text(encoding="utf-8")
    assert text.count("TUNABLE, NOT SOURCED") == 3


# --- A10: features.yaml -----------------------------------------------------
A10_COLUMNS = [
    "slope_med", "slope_p90", "relief", "rough_a", "rough_b", "sun_max_elev", "sun_low_share",
    "insol_annual", "t_mean", "t_coldest", "ti_proxy", "dust_index", "water_index", "silica_index",
    "air_density",
]


def _parse_features():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = C._parse_features(read("features"), C._Warn(sink))
    assert sink == []
    return feats


def test_features_list_every_column_and_gate_layer():
    feats = _parse_features()
    assert list(feats.columns) == A10_COLUMNS
    assert set(feats.gate_layers) == {"water_ice_fraction", "ndvi_median", "builtup_cropland_fraction"}


def test_every_weights_column_exists_in_features():
    fam, _ = _parse_weights()
    feats = _parse_features()
    used = {c for body in fam.values() for f in body.values() for c in f.columns}
    assert used <= set(feats.columns)
    assert set(feats.columns) <= used, "a feature no family uses is probably a typo"


def test_every_uncertainty_floor_has_a_source_or_an_explicit_todo():
    feats = _parse_features()
    for name, f in feats.columns.items():
        for body, src in f.sources.items():
            if src is None:
                continue
            assert body in f.uncertainty, f"{name}: no uncertainty entry for {body}"
            entry = f.uncertainty[body]
            assert entry == "TODO" or entry["source"], f"{name}.{body}"


def test_barker_floors_and_proxy_flags():
    feats = _parse_features().columns
    assert feats["slope_med"].uncertainty["moon"]["min"] == 1.5
    assert feats["slope_med"].uncertainty["moon"]["max"] == 2.5
    assert feats["relief"].uncertainty["moon"]["min"] == 0.3
    assert feats["relief"].uncertainty["moon"]["max"] == 0.5
    assert {n for n, f in feats.items() if f.proxy} == {"water_index", "silica_index"}


def test_rough_baselines_are_flagged_todo():
    text = (CONFIG_DIR / "features.yaml").read_text(encoding="utf-8")
    feats = _parse_features().columns
    assert feats["rough_a"].extra["baseline_m"] == 100
    assert feats["rough_b"].extra["baseline_m"] == 1000
    assert text.count("TODO confirm with Vihaan") >= 2


def test_earth_sources_are_leads_until_verified():
    for name, f in _parse_features().columns.items():
        earth = f.sources.get("earth")
        if earth and any(k in earth["name"] for k in ("NASADEM", "MODIS", "ASTER", "MERRA")):
            assert earth.get("lead") is True, name


# --- A11: targets.yaml ------------------------------------------------------
def _parse_targets():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = C._parse_targets(read("targets"), C._Warn(sink))
    assert sink == []
    return out


def test_moon_targets_match_setup_prompt():
    targets, _ = _parse_targets()
    c1 = targets["moon"]["connecting_ridge_c1"]
    assert (c1.lat, c1.lon_e, c1.source) == (-89.45, 222.84, "Glaser 2018")
    assert c1.attributes == {
        "lit_surface_pct": 76.0, "lit_2m_pct": 88.0, "max_shadow_h_surface": 268,
        "max_shadow_h_2m": 112, "nearest_psr_m": 100, "psr_slope_deg": 8.4,
    }
    rim = targets["moon"]["shackleton_rim"]
    assert (rim.lat, rim.lon_e, rim.source) == (-89.78, 203.97, "Glaser 2018")
    assert rim.attributes == {
        "lit_surface_pct": 81.0, "lit_2m_pct": 85.5, "max_shadow_h_surface": 221,
        "max_shadow_h_2m": 66, "nearest_psr_m": 370,
    }
    assert set(targets["moon"]) == {"connecting_ridge_c1", "shackleton_rim"}, "no C2, no Malapert"


def test_mars_targets_and_arcadia_is_not_guessed():
    targets, _ = _parse_targets()
    j = targets["mars"]["jezero"]
    assert (j.lat, j.lon_e, j.source) == (18.4386, 77.5031, "Grant 2018")
    a = targets["mars"]["arcadia"]
    assert a.lat is None and a.lon_e is None
    assert a.status == "unscored_no_target_data"


def test_target_window_settings():
    _, s = _parse_targets()
    assert s.target_window_km == 15
    assert s.tile_stride_km == 3
    assert s.spread_iqr_divisor == 1.349
    assert s.receiver_height_m == 2


def test_no_c2_anywhere_in_targets_file():
    lines = (CONFIG_DIR / "targets.yaml").read_text(encoding="utf-8").splitlines()
    data = "\n".join(line.split("#")[0] for line in lines).lower()
    assert "c2" not in data.replace("connecting_ridge_c1", "")
