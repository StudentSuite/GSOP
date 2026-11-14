"""Reference definition of ``ti_proxy`` in numpy (the Earth Engine build in ``earth/gee_export.py`` must match).

``ti_proxy`` = day-night surface-temperature swing (K, day LST minus night LST) in the **dry season**: the
``dry_months`` consecutive (cyclic) calendar months with the least precipitation. It is a proxy for thermal
inertia (high inertia gives a small swing), **not a measurement** of it.

Fallback: when no dry season can be defined, use the median swing over all valid observations and set
``ti_proxy_fallback``. A dry season is *not* defined when precipitation is missing, or when it is not
clearly drier than average: the driest-window precipitation must be below ``ratio_max`` times the average
window precipitation (annual / 12 * dry_months). A place with essentially no precipitation at all (annual
total about 0) has a trivially defined dry season (every window is equally dry).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_NO_RAIN_MM = 1e-6


def driest_window_start(pr_monthly: Sequence[float], months: int = 3) -> tuple[int, float]:
    """(0-based start month, precipitation total) of the driest cyclic window of ``months`` months."""
    pr = np.asarray(pr_monthly, dtype="float64")
    if pr.shape != (12,):
        raise ValueError("pr_monthly must have 12 values")
    totals = np.array([sum(pr[(s + k) % 12] for k in range(months)) for s in range(12)])
    start = int(np.argmin(totals))  # ties: the earliest start month
    return start, float(totals[start])


def dry_season_defined(pr_monthly: Sequence[float], months: int = 3, ratio_max: float = 0.5) -> bool:
    pr = np.asarray(pr_monthly, dtype="float64")
    if pr.shape != (12,) or not np.isfinite(pr).all():
        return False
    annual = float(pr.sum())
    if annual <= _NO_RAIN_MM:
        return True
    _, driest = driest_window_start(pr, months)
    return driest < ratio_max * annual / 12.0 * months


def ti_proxy(pr_monthly, swing_monthly, swing_all, months: int = 3, ratio_max: float = 0.5) -> tuple[float, bool]:
    """Return ``(ti_proxy, fallback)`` for one tile.

    ``pr_monthly``: 12 climatological monthly precipitation values (mm). ``swing_monthly``: 12 mean day-night
    swings (K), NaN where a month has no valid observation. ``swing_all``: every valid composite swing (K).
    """
    swing_m = np.asarray(swing_monthly, dtype="float64")
    all_swings = np.asarray(swing_all, dtype="float64")
    all_swings = all_swings[np.isfinite(all_swings)]
    fallback_value = float(np.median(all_swings)) if all_swings.size else float("nan")
    if dry_season_defined(pr_monthly, months, ratio_max):
        start, _ = driest_window_start(pr_monthly, months)
        window = swing_m[[(start + k) % 12 for k in range(months)]]
        if np.isfinite(window).all():
            return float(window.mean()), False
    return fallback_value, True
