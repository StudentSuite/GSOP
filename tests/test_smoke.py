"""Every module imports, the CLI starts, the fixtures work."""

import importlib
import pkgutil
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import gsop
from gsop.cli import app


def all_modules():
    return [m.name for m in pkgutil.walk_packages(gsop.__path__, prefix="gsop.")]


@pytest.mark.parametrize("name", all_modules())
def test_every_module_imports(name):
    importlib.import_module(name)


def test_expected_modules_exist():
    names = set(all_modules())
    expected = {
        "cli", "config", "doctor", "grid", "operators", "similarity", "gates", "weights", "montecarlo", "ml",
        "validate", "discover", "export", "runner", "runmeta", "manifest", "parallel",
        "earth.gee_export", "earth.local_loaders", "targets.moon", "targets.mars",
    }
    assert {f"gsop.{n}" for n in expected} <= names


def test_gsop_help_via_runner_lists_every_command():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "run", "golden",
                    "verify-data", "compare", "merge", "check-operators"):
        assert command in result.output


def test_gsop_help_as_a_real_process():
    proc = subprocess.run([sys.executable, "-m", "gsop", "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "Geospatial Similarity Observation Platform" in proc.stdout


def test_fixtures_work(cfg, synth_grid_moon, synth_grid_mars, results_dir, small_seed, seed_sequences):
    assert cfg.run.seed == 20261114
    assert len(synth_grid_moon.df) == len(synth_grid_mars.df) == 5_000
    assert results_dir.is_dir()
    assert small_seed == 12345 and len(seed_sequences) == 5


def test_slow_marker_is_registered(request):
    assert any(line.startswith("slow:") for line in request.config.getini("markers"))


@pytest.mark.slow
def test_slow_tests_are_skipped_unless_asked():
    # Only runs with --runslow.
    assert True


def test_readme_documents_the_slow_marker():
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "@pytest.mark.slow" in readme and "--runslow" in readme
