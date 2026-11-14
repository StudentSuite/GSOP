import shutil
import warnings
from pathlib import Path

import pytest

from gsop.config import ConfigError, ConfigWarning, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture
def cfg_dir(tmp_path):
    dest = tmp_path / "config"
    shutil.copytree(FIXTURE, dest)
    return dest


def edit(cfg_dir, name, old, new):
    path = cfg_dir / f"{name}.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {name}.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_fixture_loads_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(FIXTURE)
    assert cfg.warnings == []


def test_typed_accessors():
    cfg = load_config(FIXTURE)
    assert set(cfg.weights("moon")) == {"slope", "illumination"}
    assert cfg.weights("mars")["sunlight"].range == (0.10, 0.20)
    assert cfg.run.seed == 20261114
    assert cfg.run.runs == 2000
    assert cfg.known_sites.unconfirmed_count == 3
    assert cfg.targets("mars")["m2"].lat is None
    assert cfg.purposes("moon")["landing_mobility"].extra_gate.column == "slope_med"
    assert cfg.purposes("mars")["landing_mobility"].confidence == "low"
    assert "geology_life" not in cfg.purposes("moon")
    assert "geology_life" in cfg.purposes("mars")
    assert "sunlight" not in cfg.purposes("moon")["power_night"].multipliers
    assert cfg.gates.reason_codes["water_ice"] == "WATER_OR_ICE"
    with pytest.raises(ConfigError, match="unknown body"):
        cfg.weights("venus")


def test_missing_key_names_file_and_key(cfg_dir):
    edit(cfg_dir, "weights", "matchability: 1.0, range: [0.21, 0.39]", "range: [0.21, 0.39]")
    with pytest.raises(ConfigError, match=r"weights\.yaml: moon\.slope: missing required key 'matchability'"):
        load_config(cfg_dir)


def test_wrong_type(cfg_dir):
    edit(cfg_dir, "run", "runs: 2000", "runs: lots")
    with pytest.raises(ConfigError, match=r"run\.yaml: runs: expected an integer"):
        load_config(cfg_dir)


def test_range_must_be_ordered(cfg_dir):
    edit(cfg_dir, "weights", "range: [0.21, 0.39]", "range: [0.5, 0.2]")
    with pytest.raises(ConfigError, match=r"moon\.slope\.range: lower bound"):
        load_config(cfg_dir)


def test_unknown_family_in_purposes(cfg_dir):
    edit(cfg_dir, "purposes", "multipliers: {slope: 2}\n  extra", "multipliers: {slpoe: 2}\n  extra")
    with pytest.raises(ConfigError, match=r"purposes\.yaml: landing_mobility\.multipliers\.slpoe: unknown family"):
        load_config(cfg_dir)


def test_column_missing_from_features(cfg_dir):
    edit(cfg_dir, "weights", "columns: [sun_max_elev]}\nmars", "columns: [sun_max_elevation]}\nmars")
    with pytest.raises(ConfigError, match="sun_max_elevation.*not defined in features.yaml"):
        load_config(cfg_dir)


def test_null_target_needs_status(cfg_dir):
    edit(cfg_dir, "targets", ", status: unscored_no_target_data", "")
    with pytest.raises(ConfigError, match=r"targets\.yaml: mars\.m2: lat/lon is null"):
        load_config(cfg_dir)


def test_ledger_relevance_row_must_exist(cfg_dir):
    edit(cfg_dir, "ledger", "{gravity: \"loads\"}", "{gravty: \"loads\"}")
    with pytest.raises(ConfigError, match=r"relevance\.purposes\.landing_mobility\.gravty: unknown ledger row id"):
        load_config(cfg_dir)


def test_bad_ledger_status(cfg_dir):
    edit(cfg_dir, "ledger", "status: Broken", "status: Fine")
    with pytest.raises(ConfigError, match=r"rows\[0\]\.status"):
        load_config(cfg_dir)


def test_missing_file(cfg_dir):
    (cfg_dir / "gates.yaml").unlink()
    with pytest.raises(ConfigError, match=r"gates\.yaml: file not found"):
        load_config(cfg_dir)


def test_invalid_yaml(cfg_dir):
    (cfg_dir / "run.yaml").write_text("runs: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"run\.yaml: not valid YAML"):
        load_config(cfg_dir)


def test_unknown_key_is_a_warning_not_an_error(cfg_dir):
    edit(cfg_dir, "run", "runs: 2000", "runs: 2000\nsurprise: 1")
    with pytest.warns(ConfigWarning, match="unknown key 'surprise'"):
        cfg = load_config(cfg_dir)
    assert any("surprise" in m for m in cfg.warnings)


def test_error_is_one_line(cfg_dir):
    edit(cfg_dir, "run", "runs: 2000", "runs: lots")
    with pytest.raises(ConfigError) as err:
        load_config(cfg_dir)
    assert "\n" not in str(err.value)


def test_hash_is_stable_and_ignores_line_endings(cfg_dir, tmp_path):
    h_lf = load_config(cfg_dir).config_hash()
    crlf = tmp_path / "crlf"
    crlf.mkdir()
    for path in cfg_dir.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        (crlf / path.name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert load_config(crlf).config_hash() == h_lf
    assert len(h_lf) == 64


def test_hash_ignores_comments_and_key_order_but_not_values(cfg_dir):
    base = load_config(cfg_dir).config_hash()
    edit(cfg_dir, "run", "runs: 2000", "# a comment\nruns: 2000")
    assert load_config(cfg_dir).config_hash() == base
    edit(cfg_dir, "run", "runs: 2000", "runs: 2001")
    assert load_config(cfg_dir).config_hash() != base


def test_partial_hash_depends_only_on_its_files(cfg_dir):
    before = load_config(cfg_dir)
    edit(cfg_dir, "run", "runs: 2000", "runs: 10")
    after = load_config(cfg_dir)
    assert before.config_hash(["weights"]) == after.config_hash(["weights"])
    assert before.config_hash(["run"]) != after.config_hash(["run"])
    with pytest.raises(ConfigError, match="unknown config part"):
        after.config_hash(["nope"])
