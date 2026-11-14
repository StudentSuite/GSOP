"""Small feature functions that are not terrain, illumination or Earth Engine builds.

``air_density`` is one pure function for both bodies. It contains **no constants**: Earth uses the International
Standard Atmosphere troposphere, Mars an isothermal exponential law, with every number taken from
``config/features.yaml`` (block ``air_density``, each with a source).

    Earth:  T = T0 - L h;   p = p0 (T / T0)^(g / (R L));   rho = p / (R T)           (h in metres, valid to 11 km)
    Mars:   rho = rho0 exp(-(h - h_ref) / H),   rho0 = p0 / (R_specific T),   R_specific = R_universal / M

Heights are used as given (geometric elevation); the ISA is defined in geopotential altitude, which differs by
well under 0.2% below 11 km. The Mars pressure is stated "at mean radius" while MOLA elevations are relative to
the areoid; ``reference_elevation_m`` is a flagged TODO (docs/UNVERIFIED.md).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def air_density(elevation_m, params: Mapping[str, Any], universal_gas_constant: float) -> np.ndarray:
    """Air density in kg/m3 at ``elevation_m``. ``params`` is ``cfg.features.air_density["earth" | "mars"]``."""
    h = np.asarray(elevation_m, dtype="float64")
    model = params["model"]
    if model == "standard_atmosphere":
        t0, p0, lapse = params["sea_level_temperature_k"], params["sea_level_pressure_pa"], params["lapse_rate_k_per_m"]
        r, g = params["specific_gas_constant_j_kg_k"], params["gravity_m_s2"]
        if np.nanmax(h) > params["max_elevation_m"]:
            raise ValueError(f"elevation above {params['max_elevation_m']} m: the troposphere formula does not apply")
        t = t0 - lapse * h
        p = p0 * (t / t0) ** (g / (r * lapse))
        return p / (r * t)
    if model == "exponential":
        r_specific = universal_gas_constant / (params["molar_mass_g_mol"] / 1000.0)
        rho0 = params["surface_pressure_pa"] / (r_specific * params["temperature_k"])
        return rho0 * np.exp(-(h - params["reference_elevation_m"]) / params["scale_height_m"])
    raise ValueError(f"unknown air-density model {model!r}")
