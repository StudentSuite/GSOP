"""Seam test for the NASADEM / Copernicus switch (issue B07).

Take a narrow strip of tile values (rows = latitude, columns = longitude) that crosses a seam latitude
(60 N or 56 S). Compare the mean on the two sides of the seam with the same statistic at *placebo*
latitudes elsewhere on the strip. The step is judged against the median placebo step (so a natural
north-south gradient does not count) in units of the spread of the placebo steps. If it is a lot larger,
the DEM switch is making fake terrain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

Z_LIMIT = 3.0  # |seam step| above this many placebo standard deviations counts as a step. TUNABLE, NOT SOURCED.


@dataclass
class SeamResult:
    seam_lat: float
    step: float
    placebo_sd: float
    z: float
    n_placebo: int
    verdict: str  # "seam OK" | "seam step" | "insufficient data"

    def as_dict(self) -> dict:
        return asdict(self)


def row_means(values: np.ndarray) -> np.ndarray:
    """Mean over longitude of each latitude row, ignoring NaN."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.asarray(values, dtype="float64"), axis=1)


def window_step(means: np.ndarray, k: int, rows: int) -> Optional[float]:
    """Mean of the ``rows`` rows from index k upward minus mean of the ``rows`` rows just below k."""
    if k - rows < 0 or k + rows > len(means):
        return None
    above, below = means[k : k + rows], means[k - rows : k]
    if np.isnan(above).all() or np.isnan(below).all():
        return None
    return float(np.nanmean(above) - np.nanmean(below))


def seam_test(lat: np.ndarray, values: np.ndarray, seam_lat: float, window_deg: float = 0.5) -> SeamResult:
    """Run the seam test. ``lat`` is the row latitudes (any order), ``values`` is ``(len(lat), n_lon)``."""
    lat = np.asarray(lat, dtype="float64")
    order = np.argsort(lat)
    lat = lat[order]
    means = row_means(np.asarray(values)[order])
    spacing = float(np.median(np.diff(lat)))
    rows = max(1, int(round(window_deg / spacing)))
    k_seam = int(np.searchsorted(lat, seam_lat))
    step = window_step(means, k_seam, rows)
    placebo = []
    for k in range(rows, len(means) - rows + 1):
        if abs(lat[min(k, len(lat) - 1)] - seam_lat) <= 2 * window_deg:
            continue  # keep placebo windows clear of the seam
        s = window_step(means, k, rows)
        if s is not None:
            placebo.append(s)
    if step is None or len(placebo) < 10:
        return SeamResult(seam_lat, float("nan") if step is None else step, float("nan"), float("nan"),
                          len(placebo), "insufficient data")
    # A natural north-south gradient gives every window a similar step, so measure the seam step
    # relative to the typical (median) placebo step and in units of the spread of the placebo steps.
    excess = step - float(np.median(placebo))
    sd = float(np.std(placebo))
    z = abs(excess) / sd if sd > 0 else (0.0 if excess == 0 else float("inf"))
    return SeamResult(seam_lat, step, sd, z, len(placebo), "seam step" if z > Z_LIMIT else "seam OK")
