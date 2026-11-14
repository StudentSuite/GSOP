from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsop.config import load_config
from gsop.targets import diviner as D
from gsop.targets import moon as M

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
POLAR_WKT = ('PROJCS["x",GEOGCS["s",DATUM["u",SPHEROID["s",1737400,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
             'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],'
             'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')


def test_polar_formula_matches_the_pyproj_route_used_for_lola_and_the_real_pcp_header_line():
    for name, t in CFG.targets("moon").items():
        assert D.polar_xy_m(t.lat, t.lon_e) == pytest.approx(M.target_xy(t.lat, t.lon_e, POLAR_WKT), abs=1e-6)
    # first record of the real file pcp_avg_tbol_pols_ls030_slon01_240.tab (read 2026-11-14): x, y normalised, clon, clat
    x, y = D.polar_xy_m(-79.999451, -91.719263)
    assert x / D.MOON_RADIUS_M == pytest.approx(-0.174908, abs=2e-6) and y / D.MOON_RADIUS_M == pytest.approx(-0.005250, abs=2e-6)


def write_pcp(path, ls, slon, base, cx, cy, half=9000.0, step=240.0):
    """A synthetic PCP table in the real layout around (cx, cy): local time depends on x, temperature = base + 10*sin."""
    xs = np.arange(cx - half, cx + half, step)
    ys = np.arange(cy - half, cy + half, step)
    X, Y = np.meshgrid(xs, ys)
    ltim = (X - X.min()) / (X.max() - X.min()) * 23.99
    tbol = base + 40.0 * np.sin(2 * np.pi * ltim / 24.0) + slon * 0.01
    df = pd.DataFrame({"x": X.ravel() / D.MOON_RADIUS_M, "y": Y.ravel() / D.MOON_RADIUS_M, "clon": 0.0, "clat": -89.0, "ltim": ltim.ravel(), "tbol": tbol.ravel()})
    with open(path, "w", newline="") as fh:
        fh.write("        x,         y,        clon,       clat,    ltim,       tbol\r\n")
        df.to_csv(fh, index=False, header=False, float_format="%.6f", lineterminator="\r\n")


@pytest.fixture
def pcp(tmp_path):
    t = CFG.targets("moon")["connecting_ridge_c1"]
    cx, cy = D.polar_xy_m(t.lat, t.lon_e)
    season_base = {30: 100.0, 90: 90.0, 150: 80.0, 210: 70.0, 270: 85.0, 330: 95.0}  # Ls 210 is the coldest
    for ls, base in season_base.items():
        for slon in range(1, 5):  # 4 files per Ls bin is enough for a unit test
            write_pcp(tmp_path / f"pcp_avg_tbol_pols_ls{ls:03d}_slon{slon:02d}_240.tab", ls, slon, base, cx, cy)
    return tmp_path, cx, cy, season_base


def test_parse_name_and_window_read(pcp):
    folder, cx, cy, _ = pcp
    f = sorted(folder.glob("*.tab"))[0]
    assert D.parse_name(f) == (30, 1)
    with pytest.raises(ValueError):
        D.parse_name(Path("something_else.tab"))
    w = D.read_window(f, cx, cy, 3000.0, chunksize=200)  # tiny chunks exercise the streaming path
    assert len(w) > 100 and (w.x_m - cx).abs().max() <= 3000.0 and set(w.columns) == {"x_m", "y_m", "ltim", "tbol"}


def test_features_t_mean_coldest_and_swing(pcp):
    folder, cx, cy, base = pcp
    out = D.temperature_features(sorted(folder.glob("*.tab")), cx, cy, 15000.0, 6000.0, 3000.0, 1.349)
    assert out["n_ls_bins"] == 6 and out["t_mean"]["n_tiles"] == 16
    assert out["t_mean"]["value"] == pytest.approx(np.mean(list(base.values())), abs=3.0)
    assert out["t_coldest"]["value"] == pytest.approx(70.0, abs=3.0)  # the Ls 210 bin
    assert out["t_coldest"]["value"] < out["t_mean"]["value"]
    assert 20.0 < out["ti_proxy"]["value"] < 90.0  # the synthetic local-time curve swings by about 2 x 40 K
    assert out["t_mean"]["spread"] >= 0


def test_tile_geometry():
    corners = D.tile_edges(0.0, 0.0, 15000.0, 6000.0, 3000.0)
    assert len(corners) == 16 and corners[0] == (-7500.0, -7500.0)


def test_without_files_every_row_is_unavailable_with_the_reason():
    rows = D.moon_temperature_rows(CFG)
    assert len(rows) == 2 * 4  # t_mean, t_coldest, ti_proxy, silica_index for two targets
    assert all(r["unavailable"] and np.isnan(r["value"]) for r in rows)
    assert all("30.7 GB" in r["note"] or "Christiansen" in r["note"] for r in rows)
    assert {r["column"] for r in rows} == {"t_mean", "t_coldest", "ti_proxy", "silica_index"}


def test_an_incomplete_file_set_is_refused_not_averaged(pcp):
    folder, *_ = pcp
    rows = D.moon_temperature_rows(CFG, folder)  # 24 of 144 files
    assert all(r["unavailable"] for r in rows)
    assert any("only 24 of 144" in r["note"] for r in rows)


def test_complete_set_gives_values_with_grades_and_notes(tmp_path):
    # one synthetic table per (Ls, subsolar-longitude) bin covering a 40 km square around the pole, so both targets are inside
    for ls in (30, 90, 150, 210, 270, 330):
        for slon in range(1, 25):
            write_pcp(tmp_path / f"pcp_avg_tbol_pols_ls{ls:03d}_slon{slon:02d}_240.tab", ls, slon, 80.0 + ls / 10.0, 0.0, 0.0, half=20000.0, step=1000.0)
    rows = D.moon_temperature_rows(CFG, tmp_path)
    got = {(r["target"], r["column"]): r for r in rows}
    for target in ("connecting_ridge_c1", "shackleton_rim"):
        for col in ("t_mean", "t_coldest", "ti_proxy"):
            r = got[(target, col)]
            assert not r["unavailable"] and np.isfinite(r["value"]) and r["spread"] >= 0, (target, col)
        assert got[(target, "t_coldest")]["value"] < got[(target, "t_mean")]["value"]
        assert got[(target, "silica_index")]["unavailable"]
    grades = {r["column"]: r["evidence_grade"] for r in rows}
    assert grades["t_mean"] == "B" and grades["t_coldest"] == "B" and grades["ti_proxy"] == "C"
    assert all("NOT the coldest calendar month" in r["note"] for r in rows if r["column"] == "t_coldest")
