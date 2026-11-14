"""Earth ``dust_index`` (Mars thermal-inertia family) and the "drop an unavailable column" rule.

Earth source (issue B04 verdict: usable with caveat): MERRA-2 dust extinction aerosol optical thickness at 550 nm
(``DUEXTTAU`` in ``NASA/GSFC/MERRA/aer/2``), annual mean. Earth Engine exports the **raw** annual mean as
``dust_index_raw``; this module scales it to 0..1 with data-driven limits so that **no cap is invented**:

    dust_index = clip((raw - lo) / (hi - lo), 0, 1),   lo, hi = 1st and 99th percentile over the land tiles

(the limits are computed once, at build time, in issue B18, and stored in ``grid.json``). Higher = more dust.
Caveat: MERRA-2 is about ten times coarser than a tile and measures *atmospheric* dust, whereas the Mars side
(TES) is surface albedo and dust cover.
"""

from __future__ import annotations

import numpy as np


def percentile_limits(raw, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float, float]:
    values = np.asarray(raw, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite dust values to take percentiles of")
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if not hi > lo:
        raise ValueError("dust percentile limits collapsed (hi <= lo); dust_index cannot be scaled")
    return float(lo), float(hi)


def normalise_dust(raw, lo: float, hi: float) -> np.ndarray:
    if not hi > lo:
        raise ValueError("hi must be above lo")
    return np.clip((np.asarray(raw, dtype="float64") - lo) / (hi - lo), 0.0, 1.0)
