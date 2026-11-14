from pathlib import Path

import numpy as np
import pytest

from gsop import target_check as TC
from gsop.config import load_config
from gsop.targets import profiles_from_frame

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
OK = {"elevation_mean": -2500.0, "thermal_inertia_themis": 313.0}


def check(values, target="jezero", mode=None, body="mars"):
    return TC.check_target(CFG, body, target, values, mode)


def test_jezero_passes_with_its_real_numbers():
    c = check(OK)
    assert c.passed and not c.fails_own_gate and c.reasons == () and c.scored and c.mode == "warn"
    assert c.inputs == {"lat": 18.4386, "elevation_m": -2500.0, "thermal_inertia": 313.0}


def test_a_target_above_2_km_is_flagged_but_still_scored_in_warn_mode():
    c = check({"elevation_mean": 2400.0, "thermal_inertia_themis": 313.0})
    assert c.fails_own_gate and c.reasons == ("MARS_TARGET_CHECK:ELEVATION",) and c.mode == "warn"
    assert c.scored, "warn mode still runs the target"
    assert c.flags() == {"fails_own_gate": True, "own_gate_reasons": "MARS_TARGET_CHECK:ELEVATION"}
    v = c.validation_entry()
    assert v["fails_own_gate"] is True and v["scored"] is True and v["reasons"] == ["MARS_TARGET_CHECK:ELEVATION"]


def test_strict_mode_does_not_score_a_failing_target():
    c = check({"elevation_mean": 2400.0, "thermal_inertia_themis": 313.0}, mode="strict")
    assert c.mode == "reject" and c.fails_own_gate and not c.scored
    assert check(OK, mode="strict").scored  # a passing target is scored in strict mode too
    with pytest.raises(ValueError, match="--target-check"):
        TC.effective_mode(CFG, "sloppy")
    assert TC.effective_mode(CFG, "warn") == "warn" and TC.effective_mode(CFG) == "warn"


def test_each_threshold_and_its_boundary():
    assert check({"elevation_mean": 1999.99, "thermal_inertia_themis": 100.0}).passed  # ti equal passes
    assert check({"elevation_mean": 2000.0, "thermal_inertia_themis": 100.0}).reasons == ("MARS_TARGET_CHECK:ELEVATION",)  # elevation equal fails (below 2 km)
    assert check({"elevation_mean": 0.0, "thermal_inertia_themis": 99.99}).reasons == ("MARS_TARGET_CHECK:THERMAL_INERTIA",)


def test_latitude_check_uses_the_target_latitude_from_config(monkeypatch):
    from dataclasses import replace

    t = CFG.targets("mars")["jezero"]
    monkeypatch.setattr(CFG, "targets", lambda body: {"jezero": replace(t, lat=51.0)})
    assert check(OK).reasons == ("MARS_TARGET_CHECK:LATITUDE",)
    monkeypatch.setattr(CFG, "targets", lambda body: {"jezero": replace(t, lat=-50.0)})
    assert check(OK).passed  # exactly 50 degrees is inside


def test_several_failures_are_all_reported():
    c = check({"elevation_mean": 5000.0, "thermal_inertia_themis": 50.0})
    assert set(c.reasons) == {"MARS_TARGET_CHECK:ELEVATION", "MARS_TARGET_CHECK:THERMAL_INERTIA"}


def test_a_missing_input_cannot_pass():
    c = check({"thermal_inertia_themis": 313.0})
    assert c.fails_own_gate and c.reasons == ("MARS_TARGET_CHECK:INPUT_MISSING_ELEVATION",)
    c = check({"elevation_mean": np.nan, "thermal_inertia_themis": np.nan})
    assert set(c.reasons) == {"MARS_TARGET_CHECK:INPUT_MISSING_ELEVATION", "MARS_TARGET_CHECK:INPUT_MISSING_THERMAL_INERTIA"}


def test_moon_targets_have_no_such_check():
    c = check({}, target="connecting_ridge_c1", body="moon")
    assert not c.applicable and c.passed and not c.fails_own_gate and c.scored and c.flags()["fails_own_gate"] is False


def test_the_flag_is_in_the_documented_results_schema_and_gates_yaml_is_the_only_source_of_thresholds():
    import ast

    tree = ast.parse((Path(TC.__file__)).read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    assert numbers == set(), numbers  # no 50, 2000 or 100 in the code


def test_the_flag_columns_are_documented_in_the_results_schema():
    text = (Path(__file__).resolve().parents[1] / "docs" / "RESULTS_SCHEMA.md").read_text(encoding="utf-8")
    assert "`fails_own_gate`" in text and "`own_gate_reasons`" in text and "validation.json" in text


def test_cli_accepts_the_target_check_flag_on_the_stage_commands():
    from typer.testing import CliRunner

    from gsop.cli import app

    for cmd in (["s3", "gates"], ["s4", "score"], ["s5", "simulate"], ["run"]):
        out = CliRunner().invoke(app, [*cmd, "--help"]).output
        assert "--target-check" in out, cmd
