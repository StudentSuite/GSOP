"""Reference definitions of the three gate layers (numpy; the Earth Engine builds must match).

Bands exported (thresholds are NOT applied here, that is stage s3 and ``config/gates.yaml``):

* ``gate_water_ice_frac``   share of the tile that is open water or permanent snow and ice (MCD12Q1 classes 17, 15)
* ``gate_ndvi_med``         median NDVI over the tile of the 2015-2024 per-pixel median NDVI (MOD13A2, good/marginal QA)
* ``gate_built_crop_frac``  share of the tile that is cropland, urban or cropland/natural mosaic (classes 12, 13, 14)

Class numbers come from ``config/features.yaml`` (block ``land_cover``) and were read from the MODIS catalog page.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def class_fraction(classes: np.ndarray, wanted: Iterable[int]) -> float:
    """Share of valid (non-negative, non-NaN) cells whose class is in ``wanted``; NaN if there are none."""
    arr = np.asarray(classes, dtype="float64")
    valid = np.isfinite(arr) & (arr >= 0)
    if not valid.any():
        return float("nan")
    return float(np.isin(arr[valid], list(wanted)).sum() / valid.sum())


def median_ndvi(ndvi_cells: np.ndarray) -> float:
    """Median of the finite NDVI cells (already per-pixel climatological medians); NaN if none."""
    arr = np.asarray(ndvi_cells, dtype="float64")
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")
