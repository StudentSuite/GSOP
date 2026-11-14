from pathlib import Path

import pytest

from gsop import weights as W
from gsop.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def sets(body):
    return W.alt_weight_sets(CFG, body)


def test_the_configured_sets_exist_per_body():
    assert set(sets("moon")) == {"liu2025", "no_illumination", "equal"}
    assert set(sets("mars")) == {"mashal2025", "equal"}


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_every_set_sums_to_one_and_covers_every_family(body):
    fams = set(W.derive_weights(CFG, body))
    for name, s in sets(body).items():
        assert set(s.weights) == fams, (body, name)
        assert sum(s.weights.values()) == pytest.approx(1.0, abs=1e-12), (body, name)
        assert all(w >= 0 for w in s.weights.values())


def test_liu2025_matches_its_definition_and_zeroes_the_families_it_does_not_name():
    s = sets("moon")["liu2025"]
    assert s.weights["roughness"] == pytest.approx(0.39) and s.weights["slope"] == pytest.approx(0.26)
    assert s.weights["cold_dry"] == pytest.approx(0.16) and s.weights["illumination"] == pytest.approx(0.11) and s.weights["composition"] == pytest.approx(0.08)
    assert set(s.zeroed) == {"relief", "temperature", "thermal_inertia"}
    assert all(s.weights[f] == 0.0 for f in s.zeroed)
    assert not s.renormalised  # the listed numbers already sum to 1


def test_mashal2025_matches_its_definition_and_zeroes_the_rest():
    s = sets("mars")["mashal2025"]
    assert s.weights["composition"] == pytest.approx(0.75) and s.weights["water"] == pytest.approx(0.125) and s.weights["sunlight"] == pytest.approx(0.125)
    assert set(s.zeroed) == set(W.derive_weights(CFG, "mars")) - {"composition", "water", "sunlight"}
    assert all(s.weights[f] == 0.0 for f in s.zeroed) and not s.renormalised


def test_equal_sets_give_every_family_the_same_weight():
    for body, n in (("moon", 8), ("mars", 9)):
        s = sets(body)["equal"]
        assert all(w == pytest.approx(1.0 / n) for w in s.weights.values()) and s.zeroed == ()


def test_no_illumination_is_the_primary_with_illumination_zero_renormalised_decision_D4():
    base = W.weight_vector(W.derive_weights(CFG, "moon"))
    s = sets("moon")["no_illumination"]
    assert s.weights["illumination"] == 0.0 and s.zeroed == ("illumination",) and s.renormalised
    rest = 1.0 - base["illumination"]
    for f, w in base.items():
        if f != "illumination":
            assert s.weights[f] == pytest.approx(w / rest), f
    # ratios among the other families are untouched
    assert s.weights["slope"] / s.weights["relief"] == pytest.approx(base["slope"] / base["relief"])


def test_explicit_sets_that_do_not_sum_to_one_are_renormalised_and_say_so():
    w, zeroed, renorm = W._explicit({"a": 2.0, "b": 2.0}, ["a", "b", "c"])
    assert w == {"a": 0.5, "b": 0.5, "c": 0.0} and zeroed == ("c",) and renorm
    with pytest.raises(W.AltWeightError):
        W._explicit({}, ["a"])


def test_no_illumination_is_refused_on_mars_and_unknown_rules_fail():
    from dataclasses import replace

    from gsop.config import AltWeightSet

    class FakeCfg:
        def __init__(self, base, sets_):
            self._b, self._s = base, sets_

        def weights(self, body):
            return self._b.weights(body)

        def alt_weight_sets(self, body):
            return self._s

    with pytest.raises(W.AltWeightError, match="Moon sensitivity run"):
        W.alt_weight_sets(FakeCfg(CFG, {"x": AltWeightSet("x", None, W.RULE_NO_ILLUMINATION)}), "mars")
    with pytest.raises(W.AltWeightError, match="unknown rule"):
        W.alt_weight_sets(FakeCfg(CFG, {"x": AltWeightSet("x", None, "something else")}), "moon")


def test_spec_lists_how_missing_families_were_handled():
    text = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "families not named get weight 0" in text and "liu2025" in text and "mashal2025" in text
