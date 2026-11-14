"""Pins the derived weights (issue C04): nobody changes them by accident. Failure messages print the family and both values."""

from pathlib import Path

import pytest

from gsop import weights as W
from gsop.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
PUBLISHED_TOLERANCE = 0.012  # setup prompt, section A3 test

EXPECTED = {  # setup prompt A3 / issue C03, to 4 decimals
    "moon": {"slope": 0.3042, "illumination": 0.1806, "cold_dry": 0.1483, "roughness": 0.1198, "relief": 0.1141,
             "temperature": 0.0608, "composition": 0.0532, "thermal_inertia": 0.0190},
    "mars": {"composition": 0.2289, "water": 0.1937, "sunlight": 0.1497, "slope": 0.1132, "thermal_inertia": 0.0906,
             "roughness": 0.0792, "relief": 0.0755, "elevation_density": 0.0377, "temperature": 0.0314},
}


def fam_cases():
    return [(body, fam) for body, d in EXPECTED.items() for fam in d]


@pytest.mark.parametrize("body,family", fam_cases())
def test_derived_weight_equals_the_expected_value_to_4_decimals(body, family):
    got = W.derive_weights(CFG, body)[family].weight
    want = EXPECTED[body][family]
    assert round(got, 4) == want, f"{body}.{family}: derived {got:.6f} (rounds to {round(got, 4)}), expected {want}"


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_family_order_and_names_match_the_expected_lists(body):
    assert list(W.derive_weights(CFG, body)) == list(EXPECTED[body])


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_each_set_sums_to_one(body):
    total = sum(w.weight for w in W.derive_weights(CFG, body).values())
    assert total == pytest.approx(1.0, abs=1e-12), f"{body}: weights sum to {total!r}"


@pytest.mark.parametrize("body,family", fam_cases())
def test_each_published_weight_is_inside_its_monte_carlo_range(body, family):
    w = W.derive_weights(CFG, body)[family]
    lo, hi = w.range
    assert lo <= w.published <= hi, f"{body}.{family}: published {w.published} outside range {w.range}"


@pytest.mark.parametrize("body,family", [c for c in fam_cases() if c != ("moon", "thermal_inertia")])
def test_each_derived_weight_is_inside_its_range_and_within_tolerance_of_published(body, family):
    w = W.derive_weights(CFG, body)[family]
    lo, hi = w.range
    assert lo <= w.weight <= hi, f"{body}.{family}: derived {w.weight:.4f} outside range {w.range}"
    assert abs(w.weight - w.published) <= PUBLISHED_TOLERANCE, f"{body}.{family}: derived {w.weight:.4f} vs published {w.published}"


def test_moon_thermal_inertia_derived_019_vs_published_003_is_allowed_decision_D3():
    """D3: the derived 0.019 drives the run; the published 0.03 is a rounding-up artifact so the published column sums to 1.

    It differs from published by 0.011, inside the 0.012 tolerance, but it sits just BELOW its own Monte Carlo range [0.02, 0.04]: that is
    deliberate (the range is a sampling range, the derived value is the base score) and is the one case excluded from the range check above.
    """
    w = W.derive_weights(CFG, "moon")["thermal_inertia"]
    assert round(w.weight, 4) == 0.0190 and w.published == 0.03
    assert abs(w.weight - w.published) == pytest.approx(0.011, abs=2e-4) and abs(w.weight - w.published) <= PUBLISHED_TOLERANCE
    assert w.weight < w.range[0], "derived 0.019 is below the range lower bound 0.02, as documented"


def test_failure_message_carries_the_family_and_both_values():
    with pytest.raises(AssertionError) as err:
        assert round(0.3, 4) == 0.3042, f"moon.slope: derived {0.3:.6f} (rounds to {round(0.3, 4)}), expected 0.3042"
    assert "moon.slope" in str(err.value) and "0.3042" in str(err.value) and "0.300000" in str(err.value)
