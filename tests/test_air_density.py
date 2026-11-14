import ast
from pathlib import Path

import numpy as np
import pytest

from gsop.config import load_config
from gsop.features_misc import air_density

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")
AD = CFG.features.air_density
R = AD["universal_gas_constant_j_mol_k"]


def test_earth_sea_level_matches_the_standard_atmosphere():
    # 1.225 kg/m3, printed on the ISA page (en.wikipedia.org/wiki/International_Standard_Atmosphere)
    assert air_density(0.0, AD["earth"], R) == pytest.approx(1.225, abs=5e-4)


def test_earth_high_elevation_matches_the_printed_layer_boundary():
    # 0.3639 kg/m3 at 11,000 m, printed on the same page's layer table
    assert air_density(11000.0, AD["earth"], R) == pytest.approx(0.3639, abs=5e-4)
    assert air_density(5300.0, AD["earth"], R) < 0.8 * air_density(0.0, AD["earth"], R)  # thinner, monotone
    with pytest.raises(ValueError, match="does not apply"):
        air_density(12000.0, AD["earth"], R)


def test_mars_density_matches_the_fact_sheet_within_its_own_uncertainty():
    # fact sheet: surface density ~0.016 +/- 0.006 kg/m3
    rho = air_density(0.0, AD["mars"], R)
    assert 0.010 < rho < 0.022 and rho == pytest.approx(0.0155, abs=5e-4)
    assert air_density(11000.0, AD["mars"], R) == pytest.approx(rho / np.e)  # one scale height up
    assert air_density(-4000.0, AD["mars"], R) > rho


def test_vectorised_and_unknown_model():
    out = air_density(np.array([0.0, 1000.0, 2000.0]), AD["earth"], R)
    assert out.shape == (3,) and np.all(np.diff(out) < 0)
    with pytest.raises(ValueError):
        air_density(0.0, {"model": "nope"}, R)


def test_no_constant_appears_in_code():
    """Numeric literals in features_misc.py are only 0.0 / 1000.0 (unit conversion); the physics is in config."""
    tree = ast.parse((ROOT / "src" / "gsop" / "features_misc.py").read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    assert numbers <= {1000.0}, numbers


def test_every_constant_has_a_source_and_the_unsourced_one_is_flagged():
    assert AD["earth"]["source"] and AD["mars"]["source"]
    text = (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")
    assert "TODO UNSOURCED" in text
    assert "air_density" in (ROOT / "docs" / "UNVERIFIED.md").read_text(encoding="utf-8")
