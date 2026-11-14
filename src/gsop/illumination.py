"""Illumination features: ``sun_max_elev``, ``sun_low_share``, ``insol_annual``.

One analytic implementation shared by Earth, Moon and Mars; the only per-body inputs are the obliquity and
the solar irradiance, both from ``config/features.yaml`` (block ``illumination``). No per-body branches.

Geometry (Sun on the celestial sphere). Over a year the Sun's declination ``d`` goes between ``-eps`` and
``+eps`` (eps = obliquity), ``sin d = sin(eps) * sin(L)`` with ``L`` the Sun's ecliptic longitude. At latitude
``phi`` and hour angle ``h`` the Sun's elevation ``e`` satisfies::

    sin e = sin(phi) sin(d) + cos(phi) cos(d) cos(h)

Definitions (all for the *sky*, independent of the ground's slope):

* ``sun_max_elev`` (deg): highest Sun elevation over the year, exactly ``90 - distance(phi, [-eps, +eps])``.
* ``sun_low_share`` (0-1): share of all daylight (``e > 0``) over the year during which ``e`` is below
  ``low_sun_deg`` (5 degrees). Integrated analytically per day, summed over ``days_per_year`` samples.
* ``insol_annual`` (W/m2): annual mean of the solar flux on the ground, ``S0 * max(0, cos(theta))`` where
  ``theta`` is the angle between the Sun and the ground normal. Flat ground uses the closed-form daily mean
  ``S0/pi * (H0 sin phi sin d + cos phi cos d sin H0)``. With ``slope_deg`` and ``aspect_deg`` (aspect = compass
  direction the slope faces, clockwise from north) it integrates numerically over ``tilted_days`` x
  ``hour_samples``.

Approximations, stated plainly:

* Orbit is treated as circular: ``L`` is sampled uniformly (equal to uniform in time only for a circular
  orbit). Mars (eccentricity about 0.09) is the worst case. The mean irradiance is the fact-sheet value.
* **No terrain shadowing.** A flat, unobstructed horizon is assumed. At the lunar poles neighbouring
  mountains and crater rims, not the Sun's geometry, decide which places are lit, so these features say
  nothing about that. Earth cannot match polar lighting anyway; the limits ledger and the fidelity output
  carry that, and the illumination family uses the Glaser 2018 values for the two lunar targets.
* No atmosphere (refraction, extinction): the flux is top-of-atmosphere geometry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gsop.config import BodyConstants, IlluminationSettings

_TINY = 1e-12


def _as_array(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype="float64"))


def declination_samples(obliquity_deg: float, n: int) -> np.ndarray:
    """Sun declination (radians) at ``n`` ecliptic longitudes spread uniformly over a year."""
    longitude = (np.arange(n) + 0.5) * (2.0 * np.pi / n)
    return np.arcsin(np.sin(np.radians(obliquity_deg)) * np.sin(longitude))


def sun_max_elev(lat_deg, obliquity_deg: float) -> np.ndarray:
    """Highest Sun elevation (degrees) over the year at ``lat_deg``. Exact, no sampling."""
    lat = _as_array(lat_deg)
    distance = np.maximum(np.abs(lat) - obliquity_deg, 0.0)
    return 90.0 - distance


def _half_day_angle(a: np.ndarray, b: np.ndarray, sin_level: float) -> np.ndarray:
    """Half-length (radians, 0..pi) of the time the Sun is above elevation ``asin(sin_level)``.

    ``a = sin(phi) sin(d)`` and ``b = cos(phi) cos(d)``; above the level when ``cos h > (sin_level - a) / b``.
    """
    flat = np.abs(b) < _TINY  # Sun's elevation does not change during the day (exactly at a pole)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(flat, 0.0, (sin_level - a) / np.where(flat, 1.0, b))
    angle = np.arccos(np.clip(ratio, -1.0, 1.0))
    always = flat & (a > sin_level)
    return np.where(flat, np.where(always, np.pi, 0.0), angle)


def sun_low_share(lat_deg, obliquity_deg: float, low_sun_deg: float = 5.0, days: int = 360, chunk: int = 20000) -> np.ndarray:
    """Share of daylight over the year with the Sun below ``low_sun_deg``. NaN where there is no daylight."""
    lat = _as_array(lat_deg)
    d = declination_samples(obliquity_deg, days)[None, :]
    out = np.empty(lat.shape, dtype="float64")
    s_low = np.sin(np.radians(low_sun_deg))
    for i in range(0, lat.size, chunk):
        phi = np.radians(lat[i : i + chunk])[:, None]
        a, b = np.sin(phi) * np.sin(d), np.cos(phi) * np.cos(d)
        day = _half_day_angle(a, b, 0.0).sum(axis=1)
        above_low = _half_day_angle(a, b, s_low).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i : i + chunk] = np.where(day > 0, (day - above_low) / day, np.nan)
    return out


def _insol_flat(lat: np.ndarray, obliquity_deg: float, irradiance: float, days: int, chunk: int) -> np.ndarray:
    d = declination_samples(obliquity_deg, days)[None, :]
    out = np.empty(lat.shape, dtype="float64")
    for i in range(0, lat.size, chunk):
        phi = np.radians(lat[i : i + chunk])[:, None]
        a, b = np.sin(phi) * np.sin(d), np.cos(phi) * np.cos(d)
        h0 = _half_day_angle(a, b, 0.0)
        daily = irradiance / np.pi * (h0 * a + b * np.sin(h0))
        out[i : i + chunk] = daily.mean(axis=1)
    return out


def _insol_tilted(lat, slope_deg, aspect_deg, obliquity_deg, irradiance, days, hours, chunk=500) -> np.ndarray:
    d = declination_samples(obliquity_deg, days)[None, :, None]
    h = ((np.arange(hours) + 0.5) * (2.0 * np.pi / hours) - np.pi)[None, None, :]
    out = np.empty(lat.shape, dtype="float64")
    for i in range(0, lat.size, chunk):
        phi = np.radians(lat[i : i + chunk])[:, None, None]
        beta = np.radians(slope_deg[i : i + chunk])[:, None, None]
        gamma = np.radians(aspect_deg[i : i + chunk])[:, None, None]
        east = -np.cos(d) * np.sin(h)
        north = np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(h)
        up = np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(phi) * np.cos(h)
        cos_theta = np.sin(beta) * np.sin(gamma) * east + np.sin(beta) * np.cos(gamma) * north + np.cos(beta) * up
        flux = irradiance * np.where(up > 0, np.maximum(cos_theta, 0.0), 0.0)
        out[i : i + chunk] = flux.mean(axis=(1, 2))
    return out


def insol_annual(lat_deg, obliquity_deg: float, irradiance_w_m2: float, slope_deg=None, aspect_deg=None,
                 days: int = 360, tilted_days: int = 72, hour_samples: int = 96) -> np.ndarray:
    """Annual mean solar flux on the ground, W/m2. Flat ground unless both slope and aspect are given."""
    lat = _as_array(lat_deg)
    if (slope_deg is None) != (aspect_deg is None):
        raise ValueError("give both slope_deg and aspect_deg, or neither")
    if slope_deg is None:
        return _insol_flat(lat, obliquity_deg, irradiance_w_m2, days, chunk=20000)
    slope = np.broadcast_to(_as_array(slope_deg), lat.shape)
    aspect = np.broadcast_to(_as_array(aspect_deg), lat.shape)
    return _insol_tilted(lat, slope, aspect, obliquity_deg, irradiance_w_m2, tilted_days, hour_samples)


def compute_illumination(lat_deg, body: BodyConstants, settings: IlluminationSettings, slope_deg=None, aspect_deg=None) -> dict:
    """All three features for an array of latitudes. ``body`` is ``cfg.features.illumination.bodies[...]``."""
    lat = _as_array(lat_deg)
    return {
        "sun_max_elev": sun_max_elev(lat, body.obliquity_deg),
        "sun_low_share": sun_low_share(lat, body.obliquity_deg, settings.low_sun_deg, settings.days_per_year),
        "insol_annual": insol_annual(lat, body.obliquity_deg, body.solar_irradiance_w_m2, slope_deg, aspect_deg,
                                     settings.days_per_year, settings.tilted_days, settings.hour_samples),
    }
