"""Earth ``silica_index`` reference definition (numpy; the Earth Engine build must match).

**Quartz Index** of Ninomiya et al. (2005, Remote Sensing of Environment), as quoted in arXiv 1107.3499 for ASTER
thermal-infrared bands::

    QI = (e11 * e11) / (e10 * e12)

``eN`` is the emissivity in ASTER TIR band N (ASTER GED ``emissivity_band10`` .. ``12``, scale 0.001). Quartz-rich
(silica-rich) rock has an emissivity minimum around band 10 and 12 and a higher value at band 11, so
**higher QI = more silica-rich**. Honest limits:

* The published index was defined on radiance-at-sensor data; here it is applied to ASTER GED *emissivity*, so values
  are not comparable with the paper's maps (only relative differences between tiles matter to GSOP).
* QI is a *quartz* index, correlated with, but not equal to, bulk SiO2 content. The separate SiO2 content index
  (Ninomiya and Fu 2002) was not retrieved, so it is not used.
* Moon (Diviner Christiansen feature) and Mars (TES) must be mapped to the same direction, higher = more
  silica-rich: the Christiansen-feature wavelength moves to **shorter** wavelengths as silica increases, so a
  lunar index must be a decreasing function of that wavelength (details in issue B26).
"""

from __future__ import annotations

import numpy as np


def quartz_index(e10, e11, e12) -> np.ndarray:
    """QI from emissivities (0..1). NaN where any input is missing or not positive."""
    a, b, c = (np.asarray(x, dtype="float64") for x in (e10, e11, e12))
    ok = (a > 0) & (b > 0) & (c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (b * b) / (a * c), np.nan)
