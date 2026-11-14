import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsop.config import load_config
from gsop.earth import qa
from gsop.earth.build_parquet import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def healthy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(0.1, 0.9, n).astype("float32") for c in FEATURE_COLUMNS})
    df["slope_med"] = rng.uniform(0, 10, n)
    df["slope_p90"] = df.slope_med + rng.uniform(0, 8, n)
    df["rough_b"] = rng.uniform(1, 10, n)
    df["rough_a"] = df.rough_b + rng.uniform(0, 30, n)  # the finer baseline has the larger RMS over the same tile
    df["t_mean"] = rng.uniform(260, 300, n)
    df["t_coldest"] = df.t_mean - rng.uniform(1, 20, n)
    df["ti_proxy"] = rng.uniform(5, 30, n)
    df["relief"] = rng.uniform(0, 900, n)
    df["air_density"] = rng.uniform(0.7, 1.2, n)
    df["elevation_mean"] = rng.uniform(0, 3000, n)
    df["water_ice_fraction"] = rng.uniform(0, 0.2, n)
    df["ndvi_median"] = rng.uniform(0, 0.5, n)
    df["builtup_cropland_fraction"] = rng.uniform(0, 0.2, n)
    df["tile_id"] = np.arange(n)
    df["lat"] = rng.uniform(-60, 60, n)
    df["lon"] = rng.uniform(-170, 170, n)
    return df


def failed(checks):
    return {c.name for c in checks if not c.passed}


def test_healthy_data_passes_every_check():
    checks = qa.run_qa(healthy(), CFG)
    assert qa.overall_pass(checks), failed(checks)
    assert len(checks) > 20


def test_unit_error_and_flipped_sign_are_caught():
    df = healthy()
    df["slope_med"] = df["slope_med"] * 10  # a degrees-vs-percent style unit error: > 90
    df["t_coldest"] = df["t_mean"] + 5  # flipped sign
    df["rough_b"] = df["rough_a"] * 2.0  # roughness baselines swapped
    names = failed(qa.run_qa(df, CFG))
    assert any("slope_med within" in n for n in names)
    assert "t_coldest <= t_mean" in names
    assert any("rough_a >= rough_b" in n for n in names)
    assert not qa.overall_pass(qa.run_qa(df, CFG))


def test_unavailable_feature_and_nodata_sentinel_and_duplicate_ids():
    df = healthy()
    df["water_index"] = np.nan
    df.loc[df.index[:30], "silica_index"] = -9999.0
    df.loc[1, "tile_id"] = df.loc[0, "tile_id"]
    names = failed(qa.run_qa(df, CFG))
    assert "water_index built" in names
    assert "silica_index has no nodata sentinel values" in names
    assert "tile_id is unique" in names


def test_nan_share_limit_uses_the_config_value():
    df = healthy()
    df.loc[df.index[:200], "relief"] = np.nan  # 40% NaN, limit 20%
    assert "relief NaN share below 20%" in failed(qa.run_qa(df, CFG))


def test_bad_areas_point_at_the_failing_region():
    df = healthy()
    bad = (df.lat > 40) & (df.lon > 100)
    df.loc[bad, "slope_p90"] = 200.0
    areas = qa.bad_areas(df, qa.run_qa(df, CFG))
    cells = [c for name, v in areas.items() if "slope_p90" in name for c in v]
    assert cells and all(a >= 40 and b >= 100 for a, b, _ in cells)


def test_script_reports_when_there_is_no_earth_file(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("earth_qa", ROOT / "scripts" / "earth_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys

    old = sys.argv
    sys.argv = ["earth_qa", "--parquet", str(tmp_path / "missing.parquet")]
    try:
        assert mod.main() == 2
    finally:
        sys.argv = old
    assert "has not been built yet" in capsys.readouterr().err


def test_markdown_report_has_pass_fail_per_check_and_states_the_blocking_rule(tmp_path):
    spec = importlib.util.spec_from_file_location("earth_qa", ROOT / "scripts" / "earth_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = healthy()
    df["t_coldest"] = df["t_mean"] + 5
    checks = qa.run_qa(df, CFG)
    md = mod.render_markdown(qa.feature_stats(df), checks, qa.bad_areas(df, checks), "earth_qa", set(), len(df), "x.parquet")
    assert "## Result: FAIL" in md and "**FAIL**" in md and "blocks the Moon and Mars runs" in md and "Known bad areas" in md
    assert "NOT RUN" in (ROOT / "docs" / "EARTH_QA.md").read_text(encoding="utf-8")
