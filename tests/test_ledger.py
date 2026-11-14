"""Limits ledger and Fidelity (issue C27): hand-calculated values, provisional flags, and the guard that the ledger never reaches the score."""

import ast
import json
import shutil
from pathlib import Path

import pytest
import yaml

from gsop import ledger as L
from gsop.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")

# Worked by hand from config/ledger.yaml (Broken 0, Partly 0.5, Matchable 1), NOT through the code under test:
#   ice_resources: cold_traps B, air_pressure B, regolith P, regolith_mechanics P, oxidants P, rock_chemistry P -> (0+0+.5+.5+.5+.5)/6 = 2/6
#   landing_mobility: gravity B, air_pressure B, regolith P, regolith_mechanics P, electrostatic_dust B, line_of_sight_gps B -> 1/6
#   power_night: polar_lighting P, cold_traps B, radiation_uv P, electrostatic_dust B -> 1/4
#   geology_life: rock_chemistry P, regolith P, oxidants P, sterility B, radiation_uv P -> 2/5
#   base: all 12 rows, 6 Broken and 6 Partly -> 3/12
HAND = {"ice_resources": 100 * 2 / 6, "landing_mobility": 100 * 1 / 6, "power_night": 100 * 1 / 4, "geology_life": 100 * 2 / 5, "base": 100 * 3 / 12}


@pytest.mark.parametrize("purpose,expected", sorted(HAND.items()))
def test_fidelity_matches_the_hand_calculation(purpose, expected):
    assert L.fidelity(CFG, purpose)["fidelity"] == pytest.approx(expected, abs=1e-9)


def test_fidelity_is_a_mean_of_the_status_values_in_the_file(tmp_path):
    """Change one status in a copy of the config: the number moves by exactly 100/n * the difference."""
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    raw = yaml.safe_load((d / "ledger.yaml").read_text())
    for r in raw["rows"]:
        if r["id"] == "oxidants":
            r["status"] = "Matchable"
    (d / "ledger.yaml").write_text(yaml.safe_dump(raw))
    cfg2 = load_config(d)
    assert L.fidelity(cfg2, "ice_resources")["fidelity"] == pytest.approx(HAND["ice_resources"] + 100 * 0.5 / 6)
    assert L.fidelity(cfg2, "power_night")["fidelity"] == pytest.approx(HAND["power_night"])  # oxidants is not a power_night row


def test_rows_carry_their_reason_and_provisional_flags():
    f = L.fidelity(CFG, "landing_mobility")
    assert {r["id"] for r in f["rows"]} == {"gravity", "air_pressure", "regolith", "regolith_mechanics", "electrostatic_dust", "line_of_sight_gps"}
    assert all(r["reason"] for r in f["rows"])
    assert set(f["provisional_rows"]) == {"electrostatic_dust", "regolith_mechanics"}
    assert f["provisional"] is True and f["relevance_provisional"] is True
    # a purpose with no provisional row is still provisional while the matrix is a draft
    assert L.fidelity(CFG, "geology_life")["provisional_rows"] == [] and L.fidelity(CFG, "geology_life")["provisional"] is True


def test_report_has_one_entry_per_target_and_purpose_of_the_body():
    moon = L.fidelity_report(CFG, "moon")
    assert {(e["target"], e["purpose"]) for e in moon["entries"]} == {(t, p) for t in ("connecting_ridge_c1", "shackleton_rim")
                                                                       for p in ("base", "landing_mobility", "power_night", "ice_resources")}
    mars = L.fidelity_report(CFG, "mars")
    assert {e["purpose"] for e in mars["entries"]} == {"base", "landing_mobility", "power_night", "ice_resources", "geology_life"}
    assert {e["target"] for e in mars["entries"]} == {"jezero", "arcadia"}
    assert {e["target"]: e["scored"] for e in mars["entries"]} == {"jezero": True, "arcadia": False}
    assert len(moon["rows"]) == 12 and moon["never_used_in_scores"] is True
    assert all(e["relevance_provisional"] for e in moon["entries"])


def test_write_fidelity_json_round_trips(tmp_path):
    path = L.write_fidelity(CFG, "mars", tmp_path / "results" / "mars")
    data = json.loads(path.read_text())
    assert data["body"] == "mars" and data["ledger_hash"] == CFG.config_hash(["ledger"])
    jez = {e["purpose"]: e for e in data["entries"] if e["target"] == "jezero"}
    assert jez["ice_resources"]["fidelity"] == pytest.approx(HAND["ice_resources"])
    assert jez["ice_resources"]["provisional_rows"] == ["regolith_mechanics"]


def test_unknown_or_empty_purpose_is_an_error():
    with pytest.raises(L.LedgerError, match="no relevance entry"):
        L.fidelity(CFG, "nonsense")


def test_the_ledger_never_reaches_the_score():
    """No module on the scoring path imports ledger.py, and no scoring stage hashes the ledger file."""
    importers = []
    for p in (ROOT / "src" / "gsop").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(n == "gsop.ledger" or n.endswith(".ledger") for n in names):
                importers.append(p.relative_to(ROOT / "src" / "gsop").as_posix())
    assert set(importers) <= {"cli.py"}, importers  # only the command that writes fidelity.json may import it

    from gsop.stages import s3, s4, s5

    for mod in (s3, s4, s5):
        src = Path(mod.__file__).read_text()
        for line in src.splitlines():
            if "config_parts" in line:
                assert "ledger" not in line, (mod.__name__, line)


def test_changing_the_ledger_changes_no_score_input(tmp_path):
    """The parts of the config hash the scoring stages depend on do not move when the ledger is edited."""
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    raw = yaml.safe_load((d / "ledger.yaml").read_text())
    for r in raw["rows"]:
        r["status"] = "Matchable"
    (d / "ledger.yaml").write_text(yaml.safe_dump(raw))
    cfg2 = load_config(d)
    parts = ["gates", "purposes", "weights", "features", "run", "targets"]
    assert cfg2.config_hash(parts) == CFG.config_hash(parts)
    assert cfg2.config_hash(["ledger"]) != CFG.config_hash(["ledger"])
    assert L.fidelity(cfg2, "base")["fidelity"] == pytest.approx(100.0)
