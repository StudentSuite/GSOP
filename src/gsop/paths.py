"""Where inputs and outputs live. One place, so stages, the CLI and the tests agree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def earth_parquet(self) -> Path:
        return self.data / "processed" / "earth_features_6km.parquet"

    def targets_parquet(self, body: str) -> Path:
        return self.data / "processed" / f"targets_{body}.parquet"

    @property
    def tiles_parquet(self) -> Path:
        return self.data / "interim" / "tiles.parquet"

    def results(self, body: str) -> Path:
        return self.root / "results" / body
