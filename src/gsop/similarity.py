"""The similarity index (stage s4).

    s_k   = exp(-0.5 * ((x_tile - x_target) / sigma_k)^2)            per column, in [0, 1]
    S_f   = mean of s_k over the columns of family f                  per family
    score = 100 * sum_f w_f * S_f                                     per tile, in [0, 100]

``sigma_k`` follows the sigma rule (:func:`compute_sigma`, issue C08).

* **Missing data is missing, never 0.** A NaN column makes its family NaN and a NaN family makes the tile score NaN (the gate stage
  decides what to do with such tiles). A family with weight 0 (alternative weight sets) does not poison the score.
* **No overflow or NaN far from the target.** The squared distance is clipped at ``EXP_CAP`` before the exponential, so a tile 13 or more
  sigma away gets about 1e-37 (float32 stays normal, no denormals), never NaN or inf. Such tiles tie with one another; that is irrelevant
  for ranking because they are nowhere near the top.
* One function, :func:`score_tiles`, serves a single scoring run and the Monte Carlo inner loop: weights and target values are arguments.
  float32 throughout, processed in row chunks so memory stays bounded. The score of a tile is bit-identical whatever the chunk size.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from gsop.config import Config, ConfigError
from gsop.targets import TargetProfile

log = logging.getLogger("gsop.similarity")
EXP_CAP = 170.0  # clip on z^2: exp(-85) = 1e-37 is a normal float32
DEFAULT_CHUNK_ROWS = 262_144
SIGMA_MIN = 1e-9  # absolute guard: a column whose three sigma terms are all 0 would divide by zero


class SetupError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------
# the core maths
# ---------------------------------------------------------------------------------------------------------
def column_similarity(x: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``s_k`` for an ``(n, K)`` block: float32, NaN where ``x`` is NaN."""
    z = (x - target.astype(np.float32)) / sigma.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        z2 = np.minimum(z * z, np.float32(EXP_CAP))  # np.minimum propagates NaN, so missing stays missing
        return np.exp(np.float32(-0.5) * z2)


def family_similarity(s: np.ndarray, family_cols: Sequence[np.ndarray]) -> np.ndarray:
    """``S_f``: mean of the family's column similarities, shape ``(n, F)``. NaN if any column of the family is NaN."""
    out = np.empty((s.shape[0], len(family_cols)), dtype=np.float32)
    for j, cols in enumerate(family_cols):
        # explicit left-to-right sum: the result of a row never depends on the chunk it sits in (a BLAS matmul or a strided mean
        # would differ in the last bit between chunk sizes, and chunk size depends on each machine's RAM)
        acc = s[:, cols[0]].copy()
        for c in cols[1:]:
            acc += s[:, c]
        out[:, j] = acc / np.float32(len(cols)) if len(cols) > 1 else acc
    return out


def weighted_score(fam: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``100 * sum_f w_f S_f``. Only families with positive weight count, so a NaN in a zero-weight family cannot make the score NaN."""
    w = np.asarray(weights, dtype=np.float32)
    acc = np.zeros(fam.shape[0], dtype=np.float32)
    for j in range(fam.shape[1]):  # fixed family order, element-wise: chunk-invariant (no matmul)
        if w[j] > 0:
            acc += w[j] * fam[:, j]
    return np.float32(100.0) * acc


@dataclass
class ScoreResult:
    score: np.ndarray  # (n,) float32, 0-100, NaN where a used column is missing
    family: Optional[np.ndarray] = None  # (n, F) float32 per-family similarity


def score_tiles(
    x: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    family_cols: Sequence[np.ndarray],
    weights: np.ndarray,
    return_family: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> ScoreResult:
    """Score the ``(n, K)`` float32 feature block ``x`` against ``target`` (K,), ``sigma`` (K,), with family column indices and ``weights`` (F,)."""
    n = x.shape[0]
    score = np.empty(n, dtype=np.float32)
    fam_out = np.empty((n, len(family_cols)), dtype=np.float32) if return_family else None
    for i in range(0, n, chunk_rows):
        block = x[i : i + chunk_rows]
        fam = family_similarity(column_similarity(block, target, sigma), family_cols)
        score[i : i + chunk_rows] = weighted_score(fam, weights)
        if fam_out is not None:
            fam_out[i : i + chunk_rows] = fam
    return ScoreResult(score, fam_out)


# ---------------------------------------------------------------------------------------------------------
# a ready-to-run scoring set-up for one target on one body
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreSetup:
    body: str
    target: str
    columns: tuple[str, ...]  # the K feature columns the families use (after dropping unavailable ones)
    target_values: np.ndarray  # (K,) float64
    target_spread: np.ndarray  # (K,) float64
    sigma: np.ndarray  # (K,) float64, filled in by the sigma rule
    families: tuple[str, ...]  # families kept, in weights.yaml order
    family_cols: tuple[np.ndarray, ...]  # indices into ``columns``
    dropped: dict  # {"columns": {family: [col, ...]}, "families": [family, ...]} left out because the target had no value

    def with_sigma(self, sigma: np.ndarray) -> "ScoreSetup":
        from dataclasses import replace

        return replace(self, sigma=np.asarray(sigma, dtype="float64"))


def make_setup(cfg: Config, body: str, profile: TargetProfile, policy: Optional[str] = None) -> ScoreSetup:
    """Collect the columns the body's weights use for ``profile``.

    A column the target has no value for is dropped from its family (the family mean then uses the remaining columns, which renormalises
    inside the family). A family left with no column is handled by ``policy`` (default ``cfg.run.unavailable_target_families``):
    ``error`` stops with a message naming it; ``drop_and_renormalise`` leaves the family out (the caller renormalises the weights).
    """
    policy = policy or cfg.run.unavailable_target_families
    if policy not in ("error", "drop_and_renormalise"):
        raise SetupError(f"unknown policy {policy!r}")
    families, cols, dropped_cols, dropped_fams = [], [], {}, []
    fam_cols_names: dict[str, tuple[str, ...]] = {}
    usable = {c for c in profile.values if profile.has_value(c)}
    for fam, w in cfg.weights(body).items():
        keep = tuple(c for c in w.columns if c in usable)
        gone = [c for c in w.columns if c not in usable]
        if gone and keep:
            dropped_cols[fam] = gone
        if not keep:
            dropped_fams.append(fam)
            continue
        families.append(fam)
        fam_cols_names[fam] = keep
        for c in keep:
            if c not in cols:
                cols.append(c)
    if dropped_fams and policy == "error":
        raise SetupError(
            f"{body}/{profile.name}: no usable target value for the columns of famil{'y' if len(dropped_fams) == 1 else 'ies'} "
            f"{', '.join(dropped_fams)} (weights {', '.join(f'{cfg.weights(body)[f].published}' for f in dropped_fams)}). "
            "Supply the missing target data, or set unavailable_target_families: drop_and_renormalise in config/run.yaml (decision for Anay; see docs/SPEC.md)"
        )
    idx = {c: i for i, c in enumerate(cols)}
    return ScoreSetup(
        body=body, target=profile.name, columns=tuple(cols),
        target_values=np.array([profile.values[c] for c in cols], dtype="float64"),
        target_spread=np.array([profile.spread.get(c, 0.0) for c in cols], dtype="float64"),
        sigma=np.full(len(cols), np.nan), families=tuple(families),
        family_cols=tuple(np.array([idx[c] for c in fam_cols_names[f]], dtype="int64") for f in families),
        dropped={"columns": dropped_cols, "families": dropped_fams},
    )


def renormalise(weights: dict[str, float], families: Sequence[str]) -> np.ndarray:
    """Weights of the kept families as a vector summing to 1."""
    w = np.array([weights[f] for f in families], dtype="float64")
    return w / w.sum()


# ===========================================================================================================
# The sigma rule (issue C08)
#   sigma_k = max(target_spread_k, measurement_uncertainty_k, multiplier * IQR_k over the gated Earth tiles)
# The IQR term stops the score collapsing to zero everywhere for variables Earth cannot match (polar lighting).
# ===========================================================================================================
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_WARNED: set = set()


@dataclass(frozen=True)
class SigmaTerms:
    column: str
    sigma: float
    target_spread: float
    measurement_uncertainty: float
    iqr_term: float
    winner: str  # which of the three terms set sigma: "target_spread" | "measurement_uncertainty" | "iqr"


def measurement_uncertainty(cfg: Config, body: str, column: str) -> float:
    """Measurement-uncertainty floor of ``column``: the larger of the sourced target-side and Earth-side values (``max`` of each range).

    A ``TODO`` entry contributes 0. If neither side has a sourced value the result is 0 and a warning is logged once per column and body
    (the IQR term then carries the sigma).
    """
    feat = cfg.features.columns.get(column)
    if feat is None:
        raise ConfigError(f"features.yaml: unknown column '{column}'")
    sourced = []
    for side in (body, "earth"):
        entry = feat.uncertainty.get(side)
        if isinstance(entry, dict):
            sourced.append(float(entry["max"]))
    if not sourced:
        key = (body, column)
        if key not in _WARNED:
            _WARNED.add(key)
            log.warning("no sourced measurement uncertainty for %s on %s or on Earth (TODO in features.yaml): using 0, the IQR term carries sigma", column, body)
        return 0.0
    return max(sourced)


def compute_sigma(column: str, target_spread: float, meas_unc: float, earth_values_gated, iqr_multiplier: float = 0.25) -> SigmaTerms:
    """``max(target_spread, meas_unc, iqr_multiplier * IQR)`` with the IQR taken over the finite values of the **gated** Earth tiles only."""
    v = np.asarray(earth_values_gated, dtype="float64")
    v = v[np.isfinite(v)]
    iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 0.0
    iqr_term = iqr_multiplier * iqr
    spread = 0.0 if not np.isfinite(target_spread) else float(target_spread)
    terms = {"target_spread": spread, "measurement_uncertainty": float(meas_unc), "iqr": iqr_term}
    winner = max(terms, key=lambda k: terms[k])  # first listed wins a tie
    sigma = max(terms[winner], SIGMA_MIN)
    if terms[winner] < SIGMA_MIN:
        log.warning("sigma for %s is 0 (all three terms are 0); using %g", column, SIGMA_MIN)
    return SigmaTerms(column, sigma, spread, float(meas_unc), iqr_term, winner)


def build_sigma_table(cfg: Config, setup: ScoreSetup, earth_gated: "dict | object") -> dict[str, SigmaTerms]:
    """Sigma for every column of ``setup``. ``earth_gated`` maps column name to the values of the gated Earth tiles (a DataFrame works)."""
    return {
        c: compute_sigma(c, float(setup.target_spread[i]), measurement_uncertainty(cfg, setup.body, c), earth_gated[c], cfg.run.sigma_iqr_multiplier)
        for i, c in enumerate(setup.columns)
    }


def sigma_vector(setup: ScoreSetup, table: dict[str, SigmaTerms]) -> np.ndarray:
    return np.array([table[c].sigma for c in setup.columns], dtype="float64")


def write_sigma_table(path: Path, tables: dict[str, dict[str, SigmaTerms]], multiplier: float) -> Path:
    """Write ``sigma_table.json``: for every target, each column's sigma and the three terms that competed for it."""
    from dataclasses import asdict

    from gsop.atomic import atomic_write_text

    payload = {"rule": "sigma_k = max(target_spread_k, measurement_uncertainty_k, iqr_multiplier * IQR_k over gated Earth tiles)",
               "iqr_multiplier": multiplier,
               "targets": {t: {c: asdict(terms) for c, terms in cols.items()} for t, cols in tables.items()}}
    return atomic_write_text(Path(path), json.dumps(payload, indent=2) + "\n")
