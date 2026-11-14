from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gsop import illumination as I
from gsop.config import load_config
from gsop.targets import diviner as D
from gsop.targets import moon as M

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
POLAR_WKT = ('PROJCS["x",GEOGCS["s",DATUM["u",SPHEROID["s",1737400,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
             'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],'
             'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')


def test_illumination_rows_follow_the_documented_rule():
    rows = M.moon_illumination_rows(CFG)
    assert len(rows) == 4 and {r["column"] for r in rows} == {"sun_max_elev", "sun_low_share"}
    by = {(r["target"], r["column"]): r for r in rows}
    moon = CFG.features.illumination.bodies["moon"]
    for name, t in CFG.targets("moon").items():
        expected_max = float(I.sun_max_elev(t.lat, moon.obliquity_deg)[0])
        r = by[(name, "sun_max_elev")]
        assert r["value"] == pytest.approx(expected_max, abs=0.1) and r["spread"] < 0.2  # geometry at the tile centres
        assert 1.0 < r["value"] < 3.0  # "Sun within about 1.5 deg of the horizon"
        assert by[(name, "sun_low_share")]["value"] == pytest.approx(1.0)
        assert r["evidence_grade"] == "B" and "Glaser 2018 (not converted" in r["note"] and "lit_surface_pct" in r["note"]


def test_glaser_values_are_carried_but_not_converted():
    rows = M.moon_illumination_rows(CFG)
    t = CFG.targets("moon")["connecting_ridge_c1"]
    note = [r for r in rows if r["target"] == "connecting_ridge_c1"][0]["note"]
    assert f"lit_surface_pct={t.attributes['lit_surface_pct']}" in note and f"nearest_psr_m={t.attributes['nearest_psr_m']}" in note
    # the lit percentage (76) must not leak into either column
    assert all(r["value"] < 10 for r in rows)


def test_polar_xy_to_lat_round_trip():
    for lat, lon in ((-89.45, 222.84), (-85.0, 10.0), (-89.99, 300.0)):
        x, y = D.polar_xy_m(lat, lon)
        assert float(M.polar_xy_to_lat(np.array([x]), np.array([y]))[0]) == pytest.approx(lat, abs=1e-9)


def write_psr(path, centre_xy, half_m=40000.0, cell=60.0, shadow_box=None):
    """A synthetic LPSR map: scaled int16 (scale 2.5e-05, offset 0.5, DN -20000 sunlit / 20000 shadow), like the real file."""
    cx, cy = centre_xy
    n = int(2 * half_m / cell)
    dn = np.full((n, n), -20000, dtype="int16")
    if shadow_box:  # (xmin, xmax, ymin, ymax) metres, absolute
        xs = cx - half_m + (np.arange(n) + 0.5) * cell
        ys = cy + half_m - (np.arange(n) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)
        dn[(X >= shadow_box[0]) & (X < shadow_box[1]) & (Y >= shadow_box[2]) & (Y < shadow_box[3])] = 20000
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="int16", crs=POLAR_WKT,
                       transform=from_origin(cx - half_m, cy + half_m, cell, cell), nodata=-32768) as ds:
        ds.write(dn, 1)
        ds.scales, ds.offsets = (2.5e-05,), (0.5,)


def test_psr_share_is_the_fraction_of_each_tile_in_shadow(tmp_path):
    shack = CFG.targets("moon")["shackleton_rim"]
    ridge = CFG.targets("moon")["connecting_ridge_c1"]
    sx, sy = D.polar_xy_m(shack.lat, shack.lon_e)
    rx, ry = D.polar_xy_m(ridge.lat, ridge.lon_e)
    f = tmp_path / "lpsr.tif"
    # one synthetic map for both targets: everything west of the Shackleton target (x < sx) is permanently shadowed
    write_psr(f, ((sx + rx) / 2, (sy + ry) / 2), half_m=40000.0, shadow_box=(sx - 100000, sx, -1e9, 1e9))
    rows = M.psr_share_rows(CFG, str(f), cache_dir=tmp_path / "c")
    by = {r["target"]: r for r in rows}
    assert set(by) == {"connecting_ridge_c1", "shackleton_rim"} and all(r["column"] == "water_index" for r in rows)
    # the ridge window (x from about -18.8 to -3.8 km) lies wholly west of the shadow edge (x = -2.7 km): every tile is 100% shadow
    assert by["connecting_ridge_c1"]["value"] == pytest.approx(1.0) and by["connecting_ridge_c1"]["spread"] == pytest.approx(0.0, abs=1e-9)
    # the Shackleton window (x from -10.2 to 4.8 km) is cut by the edge: tile shares span 1.0 down to about 0.0
    s = by["shackleton_rim"]
    assert 0.0 < s["value"] < 1.0 and s["spread"] > 0.1 and not s["unavailable"] and s["evidence_grade"] == "B"
    assert "permanent shadow" in s["note"] and "LPSR_85S_060M_201608" in s["source"]


def test_scale_and_offset_are_applied_from_the_file_not_hard_coded(tmp_path):
    t = CFG.targets("moon")["shackleton_rim"]
    f = tmp_path / "all_shadow.tif"
    cx, cy = D.polar_xy_m(t.lat, t.lon_e)
    write_psr(f, (cx, cy), shadow_box=(-1e9, 1e9, -1e9, 1e9))
    info = M.read_window(str(f), t.lat, t.lon_e, 15.0)
    assert np.nanmin(info["dem"]) == pytest.approx(1.0) and np.nanmax(info["dem"]) == pytest.approx(1.0)


def test_spec_documents_the_mapping_rule_and_the_verified_encoding():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "not converted" in text.lower() and "1.0 = permanent shadow" in text
