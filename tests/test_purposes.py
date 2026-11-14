import ast
from pathlib import Path

import pytest

from gsop import weights as W
from gsop.config import load_config
from gsop.purposes import PurposeError, apply_purpose, parse_gate, purpose_names

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def base(body):
    return W.weight_vector(W.derive_weights(CFG, body))


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_every_profile_sums_to_one(body):
    for name in purpose_names(CFG, body):
        res = apply_purpose(CFG, base(body), name, body)
        assert sum(res.weights.values()) == pytest.approx(1.0, abs=1e-12), (body, name)
        assert set(res.weights) == set(base(body))


def test_profiles_per_body():
    assert sorted(purpose_names(CFG, "moon")) == sorted(["base", "landing_mobility", "power_night", "ice_resources"])
    assert sorted(purpose_names(CFG, "mars")) == sorted(["base", "landing_mobility", "power_night", "ice_resources", "geology_life"])


def test_base_is_unchanged():
    res = apply_purpose(CFG, base("moon"), "base", "moon")
    assert all(res.weights[f] == pytest.approx(w) for f, w in base("moon").items())
    assert res.extra_gate is None and res.confidence is None


def test_multipliers_multiply_then_renormalise_by_hand():
    """Moon landing_mobility: slope, roughness, relief x2. Hand calculation from the derived weights."""
    b = base("moon")
    raw = {f: w * (2.0 if f in ("slope", "roughness", "relief") else 1.0) for f, w in b.items()}
    total = sum(raw.values())
    res = apply_purpose(CFG, b, "landing_mobility", "moon")
    for f in b:
        assert res.weights[f] == pytest.approx(raw[f] / total)
    assert res.weights["slope"] / res.weights["cold_dry"] == pytest.approx(2 * b["slope"] / b["cold_dry"])


def test_power_night_uses_the_body_specific_family_name():
    moon = apply_purpose(CFG, base("moon"), "power_night", "moon")
    mars = apply_purpose(CFG, base("mars"), "power_night", "mars")
    assert moon.weights["illumination"] > base("moon")["illumination"] and "sunlight" not in moon.weights
    assert mars.weights["sunlight"] > base("mars")["sunlight"] and "illumination" not in mars.weights


def test_extra_gate_and_confidence_metadata():
    assert apply_purpose(CFG, base("moon"), "landing_mobility", "moon").extra_gate.text == "slope_med <= 8"
    assert apply_purpose(CFG, base("mars"), "landing_mobility", "mars").extra_gate.text == "slope_p90 <= 30"
    low = {(b, n) for b in ("moon", "mars") for n in purpose_names(CFG, b) if apply_purpose(CFG, base(b), n, b).confidence == "low"}
    assert low == {("mars", "landing_mobility"), ("mars", "power_night")}


def test_unknown_profile_and_moon_geology_life_raise_clear_errors():
    with pytest.raises(PurposeError, match="unknown purpose profile 'nope' for moon; known: base"):
        apply_purpose(CFG, base("moon"), "nope", "moon")
    with pytest.raises(PurposeError, match="no Moon life-detection profile"):
        apply_purpose(CFG, base("moon"), "geology_life", "moon")
    with pytest.raises(PurposeError, match="unknown body"):
        apply_purpose(CFG, base("moon"), "base", "venus")
    assert apply_purpose(CFG, base("mars"), "geology_life", "mars").weights["composition"] > base("mars")["composition"]


@pytest.mark.parametrize("text,column,op,value", [
    ("slope_med <= 8", "slope_med", "<=", 8.0), ("slope_p90<=30", "slope_p90", "<=", 30.0), ("  relief >= -1.5 ", "relief", ">=", -1.5),
    ("x < 0", "x", "<", 0.0), ("elevation_mean > 100.25", "elevation_mean", ">", 100.25),
])
def test_gate_parser_accepts_column_comparison_number(text, column, op, value):
    g = parse_gate(text)
    assert (g.column, g.op, g.value) == (column, op, value)


@pytest.mark.parametrize("text", [
    "slope_med == 8", "slope_med <= 8 and relief < 5", "8 >= slope_med", "slope_med <= eight", "__import__('os').system('x') <= 1",
    "slope_med <= 8; relief", "abs(slope_med) <= 8", "slope_med.real <= 8", "slope_med <= 1e3", "slope_med", "", "slope_med <= 8 or True",
    "slope_med != 8", "(slope_med) <= 8", "slope_med <= 8 # x",
])
def test_gate_parser_rejects_everything_else(text):
    with pytest.raises(PurposeError, match="not allowed"):
        parse_gate(text)
    with pytest.raises(PurposeError):
        parse_gate(None)  # type: ignore[arg-type]


def test_gate_boundary_equal_passes():
    g = parse_gate("slope_med <= 8")
    assert g.passes(8.0) and g.passes(7.99) and not g.passes(8.0001)
    assert parse_gate("x >= 3").passes(3.0) and not parse_gate("x < 3").passes(3.0)


def test_eval_is_never_used():
    tree = ast.parse((ROOT / "src" / "gsop" / "purposes.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__"}
