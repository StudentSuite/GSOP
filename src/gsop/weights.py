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


# ===========================================================================================================
# Alternative weight sets (issue C06): the sensitivity check that our own weights are not driving the result
# ===========================================================================================================
SUM_TOLERANCE = 1e-9  # a published set whose weights sum to 1 within this is not "renormalised"
RULE_EQUAL = "all families equal"
RULE_NO_ILLUMINATION = "primary with illumination=0, renormalised"  # decision D4 (Moon)
ILLUMINATION_FAMILY = "illumination"


class AltWeightError(ValueError):
    pass


@dataclass(frozen=True)
class AltSet:
    name: str
    weights: dict[str, float]  # every family of the body appears; sums to 1
    zeroed: tuple[str, ...]  # families that got weight 0 (absent from a published set, or switched off by a rule)
    renormalised: bool  # True if the listed weights did not sum to 1 and were rescaled
    definition: str


def _explicit(listed: dict[str, float], families: list[str]) -> tuple[dict[str, float], tuple[str, ...], bool]:
    """Published sets (Liu 2025, Mashal 2025) do not name every family: missing ones get 0, then the rest are renormalised to sum 1."""
    out = {f: float(listed.get(f, 0.0)) for f in families}
    zeroed = tuple(f for f in families if f not in listed)
    total = sum(out.values())
    if not total > 0:
        raise AltWeightError("an alternative weight set has no positive weight")
    renorm = abs(total - 1.0) > SUM_TOLERANCE
    return {f: w / total for f, w in out.items()}, zeroed, renorm


def alt_weight_sets(cfg: Config, body: str) -> dict[str, AltSet]:
    """Every alternative set configured for ``body`` (``alt_weight_sets`` in ``config/weights.yaml``), each summing to 1."""
    derived = weight_vector(derive_weights(cfg, body))
    families = list(derived)
    out: dict[str, AltSet] = {}
    for name, spec in cfg.alt_weight_sets(body).items():
        if spec.weights is not None:
            w, zeroed, renorm = _explicit(spec.weights, families)
            out[name] = AltSet(name, w, zeroed, renorm, "published weights; families not named get 0, then renormalised if the sum is not 1")
        elif spec.rule == RULE_EQUAL:
            out[name] = AltSet(name, {f: 1.0 / len(families) for f in families}, (), False, RULE_EQUAL)
        elif spec.rule == RULE_NO_ILLUMINATION:
            if ILLUMINATION_FAMILY not in derived:
                raise AltWeightError(f"{body} has no '{ILLUMINATION_FAMILY}' family: the no_illumination variant is a Moon sensitivity run (decision D4)")
            rest = {f: w for f, w in derived.items() if f != ILLUMINATION_FAMILY}
            total = sum(rest.values())
            w = {f: (rest[f] / total if f in rest else 0.0) for f in families}
            out[name] = AltSet(name, w, (ILLUMINATION_FAMILY,), True, RULE_NO_ILLUMINATION)
        else:
            raise AltWeightError(f"{body}.{name}: unknown rule {spec.rule!r} (known: {RULE_EQUAL!r}, {RULE_NO_ILLUMINATION!r})")
    return out
