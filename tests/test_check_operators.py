from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gsop import operators as O
from gsop.cli import app
from gsop.config import load_config
from gsop.earth import check_operators as CO

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def synthetic_window(perturb=None):
    """A window whose 'Earth Engine' numbers come from the numpy operator itself, optionally perturbed."""
    n = int(18000 / CO.NATIVE_M)
    x = (np.arange(n) + 0.5) * CO.NATIVE_M
    X, Y = np.meshgrid(x, x)
    dem = 300 + 0.04 * X + 25 * np.sin(X / 700.0) * np.cos(Y / 900.0)
    ops = CFG.features.operators
    tiles = O.tile_features(dem, CO.NATIVE_M, tile_m=ops.tile_m, baseline_m=ops.baseline_m, rough_baselines_m=(100.0, 1000.0))
    dem500 = O.block_mean(dem, 20)
    slope500 = O.horn_slope_deg(dem500, 500.0)
    ee_tiles = {k: tiles[k].copy() for k in CO.COMPARED}
    if perturb:
        for k, f in perturb.items():
            ee_tiles[k] = ee_tiles[k] * f
    return CO.WindowData(0.0, 18000.0, dem, dem500, slope500, ee_tiles)


def test_identical_paths_pass_with_zero_difference():
    res = CO.compare(synthetic_window(), CFG)
    assert res.passed and res.tolerance == 0.10
    assert all(r.rel_diff == pytest.approx(0.0, abs=1e-12) for r in res.features)
    assert res.intermediate["dem500 max abs difference (m)"] == pytest.approx(0.0)


def test_a_difference_just_inside_and_just_outside_the_tolerance():
    ok = CO.compare(synthetic_window({"slope_med": 1.09}), CFG)
    bad = CO.compare(synthetic_window({"slope_med": 1.12}), CFG)
    assert ok.passed and not bad.passed
    assert [r.name for r in bad.features if not r.passed] == ["slope_med"]
    md = CO.render_markdown(bad, when="2026-01-01T00:00:00+00:00")
    assert "**FAIL**" in md and "Result: **FAIL**" in md and "before any global export" in md and "Intermediate steps" in md


def test_flat_terrain_uses_an_absolute_floor_not_a_huge_ratio():
    assert CO.relative_difference(0.001, 0.0, floor=0.1) == pytest.approx(0.01)
    assert CO.relative_difference(10.0, 8.0, floor=0.1) == pytest.approx(0.25)


def test_window_is_three_by_three_tiles_with_lonar_in_the_centre():
    lonar = CFG.known_sites.primary["lonar"]
    x0, y_top, x1, y_bottom = CO.window_bounds(CFG, lonar.lat, lonar.lon)
    assert (x1 - x0, y_top - y_bottom) == (18000.0, 18000.0) and x0 % 6000 == 0 and y_bottom % 6000 == 0
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform(lonar.lon, lonar.lat)
    assert x0 + 6000 <= x < x0 + 12000 and y_bottom + 6000 <= y < y_bottom + 12000


def test_no_valid_tiles_is_a_failure_not_a_crash():
    w = synthetic_window()
    w.ee_tiles["relief"] = np.full((3, 3), np.nan)
    res = CO.compare(w, CFG)
    assert not res.passed and [r for r in res.features if r.name == "relief"][0].passed is False


def test_cli_exits_with_the_login_steps_when_earth_engine_is_missing(monkeypatch):
    from gsop.earth import gee_auth

    monkeypatch.setattr(gee_auth, "check_ee", lambda project=None: (False, "no login"))
    result = CliRunner().invoke(app, ["check-operators"])
    assert result.exit_code == 1


def test_doc_says_not_run():
    assert "NOT RUN" in (ROOT / "docs" / "OPERATOR_CHECK.md").read_text(encoding="utf-8")
