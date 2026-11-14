"""Shared fixtures and the ``slow`` marker.

Slow tests (``@pytest.mark.slow``) are skipped by default so ``uv run pytest -q`` stays under
two minutes on both machines. Run them with ``uv run pytest -q --runslow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsop.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL_SEED = 12345


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test: pass --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The real, validated config in config/."""
    return load_config(REPO_ROOT / "config")


@pytest.fixture(scope="session")
def synth_grid_moon():
    """Small synthetic Earth grid (5,000 tiles) with the Moon target planted. Read-only: do not mutate."""
    import synth

    return synth.make_grid(5_000, seed=SMALL_SEED, body="moon")


@pytest.fixture(scope="session")
def synth_grid_mars():
    """Small synthetic Earth grid (5,000 tiles) with the Mars target planted. Read-only: do not mutate."""
    import synth

    return synth.make_grid(5_000, seed=SMALL_SEED, body="mars")


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """A fresh results/<body> folder under a temp directory."""
    path = tmp_path / "results" / "moon"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def small_seed() -> int:
    return SMALL_SEED


@pytest.fixture
def seed_sequences() -> list[np.random.SeedSequence]:
    """Five spawned child seeds from a small master seed (same policy as the Monte Carlo)."""
    return np.random.SeedSequence(SMALL_SEED).spawn(5)
