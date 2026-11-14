from pathlib import Path

import numpy as np
import pytest

from gsop import illumination as I
from gsop.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
ILL = CFG.features.illumination


def test_body_constants_come_from_config_with_sources():
    assert ILL.bodies["earth"].obliquity_deg == 23.44
    assert ILL.bodies["mars"].obliquity_deg == 25.19 and ILL.bodies["mars"].solar_irradiance_w_m2 == 586.2
    assert ILL.bodies["moon"].verified_on is None  # honest: the Moon obliquity is not on the fact sheet
    assert all(b.source for b in ILL.bodies.values())


def test_equator_hand_calculation():
    eps = ILL.bodies["earth"].obliquity_deg
    assert I.sun_max_elev(0.0, eps)[0] == pytest.approx(90.0)  # Sun passes overhead at the equinoxes
    # equator: half the time the Sun is up; share of daylight below 5 deg is about 5/90 * 2/pi-ish; check bounds
    share = I.sun_low_share(0.0, eps)[0]
    assert 0.03 < share < 0.07
    # annual mean of S0/pi * cos(d): between S0/pi * cos(eps) and S0/pi
    s0 = ILL.bodies["earth"].solar_irradiance_w_m2
    val = I.insol_annual(0.0, eps, s0)[0]
    assert s0 / np.pi * np.cos(np.radians(eps)) < val < s0 / np.pi
    # independent dense quadrature of the same integral: mean over L of cos(asin(sin eps sin L))
    longitude = np.linspace(0, 2 * np.pi, 200001)[:-1]
    reference = s0 / np.pi * np.mean(np.cos(np.arcsin(np.sin(np.radians(eps)) * np.sin(longitude))))
    assert val == pytest.approx(reference, rel=1e-4)


def test_mid_latitude_hand_calculation():
    eps = 23.44
    assert I.sun_max_elev(45.0, eps)[0] == pytest.approx(90 - (45 - 23.44))  # 68.44
    assert I.sun_max_elev(-45.0, eps)[0] == pytest.approx(68.44)
    assert I.sun_max_elev(10.0, eps)[0] == pytest.approx(90.0)  # inside the tropics


def test_polar_tile_matches_the_lunar_pole_story():
    moon = ILL.bodies["moon"]
    lat = -89.45  # Connecting Ridge C1
    assert I.sun_max_elev(lat, moon.obliquity_deg)[0] == pytest.approx(1.54 + 0.55)  # 2.09 deg
    assert I.sun_low_share(lat, moon.obliquity_deg)[0] == pytest.approx(1.0)  # Sun never above 5 deg
    # annual mean flux is tiny but positive
    flux = I.insol_annual(lat, moon.obliquity_deg, moon.solar_irradiance_w_m2)[0]
    assert 0 < flux < 20


def test_pole_exact_degenerate_case_does_not_blow_up():
    out = I.compute_illumination([-90.0, 90.0], ILL.bodies["moon"], ILL)
    assert np.isfinite(out["sun_max_elev"]).all() and np.isfinite(out["insol_annual"]).all()
    assert out["sun_max_elev"][0] == pytest.approx(1.54)
    assert out["sun_low_share"][0] == pytest.approx(1.0)


def test_tilted_numeric_quadrature_agrees_with_the_flat_closed_form():
    lat = np.array([-60.0, -20.0, 0.0, 33.0, 70.0])
    s0, eps = 1361.0, 23.44
    flat = I.insol_annual(lat, eps, s0)
    num = I.insol_annual(lat, eps, s0, slope_deg=np.zeros(5), aspect_deg=np.zeros(5), tilted_days=180, hour_samples=288)
    assert np.allclose(num, flat, rtol=2e-3, atol=0.5)


def test_slope_direction_matters_in_the_right_way():
    s0, eps, lat = 1361.0, 23.44, 45.0
    flat = I.insol_annual(lat, eps, s0)[0]
    toward_equator = I.insol_annual(lat, eps, s0, slope_deg=[20.0], aspect_deg=[180.0])[0]  # south-facing in north
    away = I.insol_annual(lat, eps, s0, slope_deg=[20.0], aspect_deg=[0.0])[0]
    assert away < flat < toward_equator or (away < flat and toward_equator > away)
    with pytest.raises(ValueError):
        I.insol_annual(lat, eps, s0, slope_deg=[5.0])


def test_north_south_symmetry_and_bounds_of_low_share():
    eps = 23.44
    lats = np.array([0.0, 30.0, 60.0, 80.0, 89.0])
    share = I.sun_low_share(lats, eps)
    assert np.allclose(I.sun_low_share(-lats, eps), share, atol=1e-6)
    assert np.all((share > 0) & (share <= 1))
    assert share[0] < share[2]  # low Sun is much commoner at 60 deg than at the equator


def test_mars_and_earth_share_the_same_code_path():
    for body in ("earth", "mars", "moon"):
        out = I.compute_illumination(np.linspace(-80, 80, 9), ILL.bodies[body], ILL)
        assert set(out) == {"sun_max_elev", "sun_low_share", "insol_annual"}
        assert all(v.shape == (9,) for v in out.values())


def test_spec_states_what_the_feature_cannot_capture():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "terrain shadowing" in text.lower()
