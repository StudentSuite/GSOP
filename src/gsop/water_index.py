"""Reference definition of the Earth ``water_index`` (numpy; the Earth Engine build must match).

``water_index`` is the Moon's ``cold_dry`` family and the Mars ``water`` family, so on Earth it stands in for
"cold and dry ground with ice": **a cross-instrument proxy, one of the two weakest comparisons in GSOP**.

    water_index = mean( permafrost_probability , dryness )      both in 0..1, higher = more ice-like

* ``permafrost_probability``: 0 (no permafrost) to 1 (certain permafrost).
* ``dryness = 1 - clip(AI / ai_cap, 0, 1)`` with ``AI = P / PET`` (annual precipitation over annual reference
  evapotranspiration, the UNEP aridity index). **0 means fully humid** (AI at or above ``ai_cap``, default 1:
  precipitation at least matches evaporative demand), **1 means no precipitation** (hyperarid). Direction:
  higher = drier.
* If either input is missing the result is NaN (no data in either layer gives NaN, never 0). The optional policy
  ``permafrost_outside_domain: zero`` first turns a missing permafrost value into 0.

Hot deserts therefore score high on dryness; only the permafrost half separates cold from hot, so a hot desert
gets about 0.5 and a cold desert with permafrost gets close to 1.
"""

from __future__ import annotations

import numpy as np


def dryness(precip_annual_mm, pet_annual_mm, ai_cap: float = 1.0) -> np.ndarray:
    """Dryness 0..1 from annual precipitation and reference evapotranspiration (mm). NaN where PET <= 0 or missing."""
    p = np.asarray(precip_annual_mm, dtype="float64")
    pet = np.asarray(pet_annual_mm, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ai = np.where(pet > 0, p / pet, np.nan)
    return 1.0 - np.clip(ai / ai_cap, 0.0, 1.0)


def water_index(permafrost_probability, dryness_01, outside_domain: str = "nan") -> np.ndarray:
    """Mean of permafrost probability and dryness; NaN if either is missing (after the optional zero policy)."""
    if outside_domain not in ("nan", "zero"):
        raise ValueError("outside_domain must be 'nan' or 'zero'")
    perm = np.asarray(permafrost_probability, dtype="float64")
    dry = np.asarray(dryness_01, dtype="float64")
    if outside_domain == "zero":
        perm = np.where(np.isnan(perm), 0.0, perm)
    return (np.clip(perm, 0.0, 1.0) + np.clip(dry, 0.0, 1.0)) / 2.0
