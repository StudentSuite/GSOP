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


# --- A12: ledger.yaml -------------------------------------------------------
LEDGER_IDS = [
    "gravity", "air_pressure", "cold_traps", "polar_lighting", "radiation_uv", "regolith",
    "rock_chemistry", "oxidants", "sterility", "line_of_sight_gps", "electrostatic_dust", "regolith_mechanics",
]
LEDGER_STATUS = {
    "gravity": "Broken", "air_pressure": "Broken", "cold_traps": "Broken", "polar_lighting": "Partly",
    "radiation_uv": "Partly", "regolith": "Partly", "rock_chemistry": "Partly", "oxidants": "Partly",
    "sterility": "Broken", "line_of_sight_gps": "Broken", "electrostatic_dust": "Broken",
    "regolith_mechanics": "Partly",
}


def _parse_ledger():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = C._parse_ledger(read("ledger"), _parse_purposes(), C._Warn(sink))
    assert sink == []
    return out


def test_ledger_has_the_twelve_rows_with_the_specified_status():
    ledger = _parse_ledger()
    assert list(ledger.rows) == LEDGER_IDS
    assert {rid: row.status for rid, row in ledger.rows.items()} == LEDGER_STATUS


def test_provisional_rows_cite_shivens_report():
    ledger = _parse_ledger()
    prov = {rid for rid, row in ledger.rows.items() if row.provisional}
    assert prov == {"electrostatic_dust", "regolith_mechanics"}
    for rid in prov:
        assert ledger.rows[rid].source == "Shiven's report (unverified)"


def test_fidelity_status_values_and_relevance_matrix():
    ledger = _parse_ledger()
    assert ledger.status_values == {"Broken": 0, "Partly": 0.5, "Matchable": 1}
    assert ledger.relevance_provisional is True
    assert set(ledger.relevance) == {"base", "landing_mobility", "power_night", "ice_resources", "geology_life"}
    assert set(ledger.relevance["base"]) == set(LEDGER_IDS)
    for purpose, rows in ledger.relevance.items():
        assert rows, purpose
        assert all(reason.strip() for reason in rows.values())


def test_ledger_header_states_the_fidelity_formula():
    text = (CONFIG_DIR / "ledger.yaml").read_text(encoding="utf-8")
    assert "fidelity = 100 * mean(status_value over the ledger rows relevant to that purpose)" in text
    assert "Broken 0, Partly 0.5, Matchable 1" in text


# --- A13: known_sites.yaml --------------------------------------------------
def _parse_sites():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = C._parse_known_sites(read("known_sites"), C._Warn(sink))
    assert sink == []
    return out


def test_site_lists_and_nothing_is_confirmed():
    ks = _parse_sites()
    assert list(ks.primary) == [
        "lonar", "haughton", "mcmurdo_dry_valleys", "atacama_salars", "ladakh_puga", "mdrs_utah",
    ]
    assert list(ks.secondary) == ["sf_volcanic_field", "gosses_bluff", "mauna_kea", "askja"]
    assert list(ks.gate_tests) == ["manicouagan", "tatacoa", "samoylov"]
    assert list(ks.negative_controls) == ["amazon_basin", "congo_basin"]
    assert ks.unconfirmed_count == 15
    assert not any(s.confirmed for s in ks.all_sites.values()), "confirming is a later human step (D01)"


def test_site_coordinates_match_setup_prompt():
    ks = _parse_sites()
    expected = {
        "lonar": (19.98, 76.51), "haughton": (75.37, -89.68), "mcmurdo_dry_valleys": (-77.5, 162.0),
        "atacama_salars": (-23.5, -68.25), "ladakh_puga": (33.22, 78.32), "mdrs_utah": (38.41, -110.79),
        "sf_volcanic_field": (35.35, -111.5), "gosses_bluff": (-23.8, 132.3), "mauna_kea": (19.82, -155.47),
        "askja": (65.03, -16.75), "manicouagan": (51.1, -68.7), "tatacoa": (3.23, -75.17),
        "samoylov": (72.37, 126.48), "amazon_basin": (-3.0, -60.0), "congo_basin": (-0.5, 23.0),
    }
    got = {n: (s.lat, s.lon) for n, s in ks.all_sites.items()}
    assert got == expected


def test_bodies_radius_and_expectations():
    ks = _parse_sites()
    assert ks.primary["lonar"].bodies == ("mars", "moon")
    assert ks.primary["mcmurdo_dry_valleys"].bodies == ("mars",)
    assert ks.secondary["sf_volcanic_field"].radius_km == 30
    assert ks.secondary["mauna_kea"].bodies == ("moon",)
    assert all(s.expect == "rejected" for s in ks.gate_tests.values())
    assert ks.site_match_radius_km == 20


def test_pre_registered_pass_criteria():
    pc = _parse_sites().pass_criteria
    assert pc == {
        "primary_median_rank_percentile_min": 90,
        "negative_control_rank_percentile_below": 25,
        "gate_tests": "all_rejected",
    }


# --- A14: run.yaml ----------------------------------------------------------
def test_run_values_match_setup_prompt():
    sink = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = C._parse_run(read("run"), C._Warn(sink))
    assert sink == []
    assert run.runs == 2000
    assert run.seed == 20261114
    assert run.top_fractions == (0.01, 0.05, 0.10)
    assert (run.robust_top_fraction, run.robust_min_run_share) == (0.05, 0.9)
    assert run.candidate_fraction == 0.03
    assert run.chunk_ram_fraction == 0.4
    assert run.site_match_radius_km == 20
    assert run.novelty_radius_km == 100
    assert run.convergence_checkpoints == (500, 1000, 1500, 2000)
    assert run.golden_score_tolerance == 1e-4
    assert run.golden_top1_jaccard_min == 0.99
    assert run.operator_check_tolerance == 0.10


def test_site_radius_is_not_duplicated_between_files():
    assert read("run")["site_match_radius_km"] == read("known_sites")["site_match_radius_km"]


def test_changing_runs_changes_the_default_of_simulate(tmp_path, monkeypatch):
    import shutil

    from gsop import cli, config

    shutil.copytree(CONFIG_DIR, tmp_path / "config")
    monkeypatch.setattr(config, "DEFAULT_CONFIG_DIR", tmp_path / "config")
    assert cli._default_runs() == 2000
    path = tmp_path / "config" / "run.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("runs: 2000\n", "runs: 123\n", 1), encoding="utf-8")
    assert cli._default_runs() == 123


# --- the whole real config loads together -------------------------------------
def test_real_config_loads_without_warnings_and_hash_is_stable():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = C.load_config(CONFIG_DIR)
    assert cfg.warnings == []
    assert cfg.config_hash() == C.load_config(CONFIG_DIR).config_hash()
    assert len(cfg.weights("moon")) == 8 and len(cfg.weights("mars")) == 9
    assert cfg.known_sites.unconfirmed_count == 15


# --- A19: docs ---------------------------------------------------------------
DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_decisions_has_d1_to_d10_and_is_linked_from_readme():
    text = (DOCS / "DECISIONS.md").read_text(encoding="utf-8")
    for i in range(1, 11):
        assert f"**D{i}.**" in text
    assert "Anay" in text and "2026-11-14" in text
    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
    assert "docs/SPEC.md" in readme and "docs/DECISIONS.md" in readme
    assert "updates `docs/SPEC.md` in the same pull request" in readme


def test_spec_headings_cover_pipeline_stages_s1_to_s9():
    text = (DOCS / "SPEC.md").read_text(encoding="utf-8")
    for stage in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"):
        assert f"({stage})" in text
    for heading in ("Grid", "Features", "Operators", "Similarity", "Sigma rule", "Weights", "Purposes", "Gates",
                    "Monte Carlo", "ML", "Validation", "Discovery", "Ledger and fidelity", "Known weaknesses"):
        assert f"## {heading}" in text, heading


# --- A20: UNVERIFIED.md and PROVENANCE.md ------------------------------------
def test_unverified_has_the_seeded_rows():
    text = (DOCS / "UNVERIFIED.md").read_text(encoding="utf-8")
    assert "| Item | Where used | Why unverified | Who must confirm | Status |" in text
    for needle in ("Known-site coordinates", "Arcadia Planitia centre", "relevance matrix", "Sigma floors",
                   "Tolerances", "rough_a", "Earth dataset leads"):
        assert needle in text, needle


def test_provenance_table_columns_and_no_invented_ids():
    text = (DOCS / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "| Dataset | Id / version | Date range | Resolution | Licence | How accessed | Checked on | Checked by |" in text
    assert "## Software" in text
    dataset_rows = [ln for ln in text.split("## Software")[0].splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    assert dataset_rows
    assert all("NOT VERIFIED" in ln or "NOT CHOSEN" in ln for ln in dataset_rows)


def test_readme_states_the_no_invention_rule():
    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
    assert "an unverified item is allowed, an undocumented one is not" in readme
