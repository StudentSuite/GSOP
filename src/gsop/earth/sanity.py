"""Plausibility bounds for every exported Earth band. Shared by the small-region check (B17) and the Earth QA (B20).

These are loose physical or definitional bounds, not scientific targets: a value outside them means a unit error,
a flipped sign or a broken band, never "unusual terrain". They are tunable and unsourced beyond the reasoning given.
"""

from __future__ import annotations

# band: (lower, upper, why). None = unbounded on that side.
SANITY_RANGES: dict[str, tuple[float | None, float | None, str]] = {
    "slope_med": (0.0, 90.0, "slope in degrees"),
    "slope_p90": (0.0, 90.0, "slope in degrees"),
    "relief": (0.0, 15000.0, "max minus min of a tile; Earth's total relief is about 20 km, a 6 km tile cannot exceed 15 km"),
    "rough_a": (0.0, 5000.0, "RMS of heights; non-negative"),
    "rough_b": (0.0, 5000.0, "RMS of heights; non-negative"),
    "elevation_mean": (-500.0, 9000.0, "Earth land elevation in m (ASTER GED page range -500..9000)"),
    "t_mean": (150.0, 360.0, "Earth surface temperature in K"),
    "t_coldest": (150.0, 360.0, "Earth surface temperature in K"),
    "t_low_obs": (0.0, 1.0, "flag"),
    "ti_proxy": (0.0, 80.0, "day-night swing in K"),
    "ti_proxy_fallback": (0.0, 1.0, "flag"),
    "water_index": (0.0, 1.0, "index 0-1"),
    "dryness": (0.0, 1.0, "index 0-1"),
    "dust_index_raw": (0.0, 5.0, "aerosol optical thickness, non-negative"),
    "silica_index": (0.0, 5.0, "quartz index, positive ratio of emissivities near 1"),
    "land_frac": (0.0, 1.0, "fraction"),
    "gate_water_ice_frac": (0.0, 1.0, "fraction"),
    "gate_ndvi_med": (-1.0, 1.0, "NDVI"),
    "gate_built_crop_frac": (0.0, 1.0, "fraction"),
}
