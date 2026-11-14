"""Family weights, derived by rule from ``config/weights.yaml`` (never typed in).

    weight_f = (paper_score_f / sum(paper_score)) * matchability_f,   then rescaled so the weights sum to 1

Decision D3: the Moon ``thermal_inertia`` weight is the derived 0.019, not the published 0.03 (a rounding-up artifact so the published column
sums to 1); both stay in config, the derived value drives the run.

A family's weight is split equally over its columns: the family score is the *mean* of its columns' similarities (``gsop.similarity``), so
``column_weights`` returns ``weight / n_columns`` for each column.
"""

from __future__ import annotations

from dataclasses import dataclass

from gsop.config import Config


@dataclass(frozen=True)
class DerivedWeight:
    name: str
    weight: float  # derived, sums to 1 over the body's families
    range: tuple[float, float]  # what the Monte Carlo draws this family's weight from
    published: float  # the rounded number in the plan file, for comparison only
    columns: tuple[str, ...]
    paper_score: float
    matchability: float


def derive_weights(cfg: Config, body: str) -> dict[str, DerivedWeight]:
    """Derived family weights for ``body``, in the order of ``config/weights.yaml``."""
    families = cfg.weights(body)
    total = sum(f.paper_score for f in families.values())
    raw = {name: (f.paper_score / total) * f.matchability for name, f in families.items()}
    norm = sum(raw.values())
    return {
        name: DerivedWeight(name, raw[name] / norm, f.range, f.published, f.columns, f.paper_score, f.matchability)
        for name, f in families.items()
    }


def weight_vector(weights: dict[str, DerivedWeight]) -> dict[str, float]:
    """Plain ``{family: weight}``, the form the purpose and scoring functions take."""
    return {name: w.weight for name, w in weights.items()}


def column_weights(weights: dict[str, float], columns: dict[str, tuple[str, ...]]) -> dict[str, float]:
    """Weight of each column: the family weight split equally across the family's columns."""
    out: dict[str, float] = {}
    for fam, w in weights.items():
        cols = columns[fam]
        for c in cols:
            out[c] = out.get(c, 0.0) + w / len(cols)
    return out
