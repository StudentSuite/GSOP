"""s5: the Monte Carlo over weights and data (decisions D2 and D8).

Order of operations for run ``i`` (documented in ``docs/SPEC.md``):

1. Seeds. Run ``i`` uses ``SeedSequence(master_seed, spawn_key=(i,))``, which is exactly child ``i`` of ``SeedSequence(master_seed).spawn(n)`` for any
   ``n``; it never depends on the worker count. Its two children, ``spawn_key=(i, 0)`` and ``(i, 1)``, feed the **weight stream** and the **data stream**,
   so the two streams never overlap.
2. Weights (issue C15). Draw each family weight uniformly within its ``range`` (``config/weights.yaml``), renormalise to sum 1, multiply by the purpose
   multipliers, renormalise again.
3. Data jitter (issue C16, in the same file below). The target's value of each column is perturbed by N(0, measurement uncertainty); Earth values get a
   systematic bias per column, N(0, Earth bias). Both reduce to ONE shift of the target value per column, so no per-tile work is needed.
4. Score every gated tile (``gsop.similarity.score_tiles``), update online statistics, and count membership of the top 1%, 5% and 10%.

Every function that depends on the run index is pure: any worker can reproduce any run from the setup and the index alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gsop.config import Config
from gsop.similarity import ScoreSetup

log = logging.getLogger("gsop.montecarlo")

WEIGHT_STREAM = 0
DATA_STREAM = 1


@dataclass(frozen=True)
class MCSetup:
    """Everything one Monte Carlo run needs, for one target and one profile. Small and picklable."""

    body: str
    target: str
    profile: str
    columns: tuple[str, ...]
    families: tuple[str, ...]
    family_cols: tuple[np.ndarray, ...]
    target_values: np.ndarray  # (K,) float64, the unjittered target
    sigma: np.ndarray  # (K,) float64, fixed for all runs
    weight_lo: np.ndarray  # (F,) lower bound of each family's weight range
    weight_hi: np.ndarray  # (F,) upper bound
    multipliers: np.ndarray  # (F,) purpose multipliers (1 for families the profile does not touch)
    target_unc: np.ndarray  # (K,) sd of the target value jitter (0 where the uncertainty is TODO)
    earth_bias_sd: np.ndarray  # (K,) sd of the Earth systematic bias (0 where TODO)
    master_seed: int


def run_seed(master_seed: int, i: int, stream: Optional[int] = None) -> np.random.SeedSequence:
    """Seed of run ``i`` (``stream=None``) or of its weight (0) / data (1) stream. Pure: no global state."""
    key = (int(i),) if stream is None else (int(i), int(stream))
    return np.random.SeedSequence(master_seed, spawn_key=key)


def sample_weights(mcs: MCSetup, i: int, return_raw: bool = False):
    """Family weights of run ``i``: uniform in each range, renormalise, apply the purpose multipliers, renormalise. Sums to 1."""
    rng = np.random.default_rng(run_seed(mcs.master_seed, i, WEIGHT_STREAM))
    raw = rng.uniform(mcs.weight_lo, mcs.weight_hi)  # F draws in family order
    w = raw / raw.sum()
    w = w * mcs.multipliers
    w = w / w.sum()
    return (w, raw) if return_raw else w


# ===========================================================================================================
# Data jitter (issue C16)
#   target value_k  ~ N(value_k, target measurement uncertainty_k)         (target side, features.yaml uncertainty.<body>.max)
#   Earth value_k   + b_k,  b_k ~ N(0, Earth systematic bias_k)            (features.yaml earth_bias.max)
# (x + b - t') / sigma = (x - (t' - b)) / sigma, so both terms are ONE shift of the target value per column: shift_k = e_k - b_k.
# Columns whose uncertainty or bias is TODO are not jittered (sd 0); that is logged once per column.
# Not jittered: sigma (fixed per column, see the sigma rule), the weights' ranges, and per-tile noise (see below).
# ===========================================================================================================
_JITTER_WARNED: set = set()
#: Per-tile noise for candidate tiles only (the issue's optional term) is NOT implemented: it would make pass 2 scores differ from pass 1 scores for
#: the same run, which pass 2 must not do. The parameter is kept at 0 and documented in SPEC.md.
CANDIDATE_TILE_NOISE_SD = 0.0


def target_uncertainty_sd(cfg: Config, body: str, column: str) -> float:
    """sd of the target-value jitter: the sourced target-side measurement uncertainty (``max`` of the range); 0 and one warning if TODO."""
    entry = cfg.features.columns[column].uncertainty.get(body)
    if isinstance(entry, dict):
        return float(entry["max"])
    key = ("target", body, column)
    if key not in _JITTER_WARNED:
        _JITTER_WARNED.add(key)
        log.warning("target value of %s on %s has no sourced uncertainty (TODO): not jittered", column, body)
    return 0.0


def earth_bias_sd(cfg: Config, column: str) -> float:
    """sd of the Earth systematic bias of ``column``: the sourced ``earth_bias`` (``max``); 0 and one warning if TODO."""
    entry = cfg.features.columns[column].earth_bias
    if isinstance(entry, dict):
        return float(entry["max"])
    key = ("earth_bias", column)
    if key not in _JITTER_WARNED:
        _JITTER_WARNED.add(key)
        log.warning("Earth systematic bias of %s has no sourced estimate (TODO): not jittered", column)
    return 0.0


def make_mc_setup(cfg: Config, setup: ScoreSetup, profile: str, master_seed: Optional[int] = None) -> MCSetup:
    """Monte Carlo set-up for one target and one profile from a ``ScoreSetup`` whose sigma is already filled in."""
    if np.isnan(setup.sigma).any():
        raise ValueError("the ScoreSetup has no sigma yet: run the sigma rule first (ScoreSetup.with_sigma)")
    fam_cfg = cfg.weights(setup.body)
    mult = cfg.purposes(setup.body)[profile].multipliers
    return MCSetup(
        body=setup.body, target=setup.target, profile=profile, columns=setup.columns, families=setup.families, family_cols=setup.family_cols,
        target_values=setup.target_values, sigma=setup.sigma,
        weight_lo=np.array([fam_cfg[f].range[0] for f in setup.families], dtype="float64"),
        weight_hi=np.array([fam_cfg[f].range[1] for f in setup.families], dtype="float64"),
        multipliers=np.array([mult.get(f, 1.0) for f in setup.families], dtype="float64"),
        target_unc=np.array([target_uncertainty_sd(cfg, setup.body, c) for c in setup.columns], dtype="float64"),
        earth_bias_sd=np.array([earth_bias_sd(cfg, c) for c in setup.columns], dtype="float64"),
        master_seed=cfg.run.seed if master_seed is None else int(master_seed),
    )


def sample_data_shift(mcs: MCSetup, i: int) -> np.ndarray:
    """The per-column shift of the target value in run ``i``: ``target_jitter - earth_bias``. Always draws both normals so the stream is stable."""
    rng = np.random.default_rng(run_seed(mcs.master_seed, i, DATA_STREAM))
    z = rng.normal(size=(2, len(mcs.columns)))
    return z[0] * mcs.target_unc - z[1] * mcs.earth_bias_sd


def run_params(mcs: MCSetup, i: int) -> tuple[np.ndarray, np.ndarray]:
    """``(weights (F,), jittered target values (K,))`` of run ``i``."""
    return sample_weights(mcs, i), mcs.target_values + sample_data_shift(mcs, i)


# ===========================================================================================================
# Core scoring with online statistics (issue C17): 4 million tiles x 2,000 runs never exists in memory
# ===========================================================================================================
import os  # noqa: E402

from gsop import similarity as _S  # noqa: E402

MAX_BUFFER_COLUMNS = 64  # the largest legitimate per-run buffer is (tiles x columns); an (tiles x runs) array would be far bigger


def debug_enabled() -> bool:
    return os.environ.get("GSOP_MC_DEBUG", "") not in ("", "0")


class AllocGuard:
    """In debug mode (``GSOP_MC_DEBUG=1``) refuse any buffer whose size looks like (tiles x runs); the message carries the allocated size."""

    def __init__(self, n_tiles: int, runs: int, enabled: Optional[bool] = None) -> None:
        self.n_tiles, self.runs = n_tiles, runs
        self.enabled = debug_enabled() if enabled is None else enabled

    def check(self, name: str, shape) -> None:
        elements = int(np.prod(shape))
        if self.enabled and elements > self.n_tiles * MAX_BUFFER_COLUMNS:
            raise AssertionError(f"{name}: allocating {elements:,} elements ({elements * 4 / 1e6:,.0f} MB as float32) for {self.n_tiles:,} tiles and {self.runs} runs "
                                 f"looks like an N x runs matrix; the engine must stream")


class OnlineStats:
    """Per-tile running statistics over runs: count, mean, M2 (for the variance), min, max. Mergeable (Chan et al.) in a fixed order."""

    __slots__ = ("count", "mean", "m2", "minimum", "maximum")

    def __init__(self, n_tiles: int) -> None:
        self.count = 0
        self.mean = np.zeros(n_tiles, dtype=np.float64)
        self.m2 = np.zeros(n_tiles, dtype=np.float64)
        self.minimum = np.full(n_tiles, np.inf, dtype=np.float32)
        self.maximum = np.full(n_tiles, -np.inf, dtype=np.float32)

    def update(self, score: np.ndarray) -> None:
        self.count += 1
        x = score.astype(np.float64)
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        np.minimum(self.minimum, score, out=self.minimum)
        np.maximum(self.maximum, score, out=self.maximum)

    def merge(self, other: "OnlineStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2, self.minimum, self.maximum = other.count, other.mean.copy(), other.m2.copy(), other.minimum.copy(), other.maximum.copy()
            return
        n = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * (self.count * other.count / n)
        self.mean += delta * (other.count / n)
        self.count = n
        np.minimum(self.minimum, other.minimum, out=self.minimum)
        np.maximum(self.maximum, other.maximum, out=self.maximum)

    @property
    def variance(self) -> np.ndarray:
        """Population variance over the runs seen (divide by count)."""
        return self.m2 / max(self.count, 1)

    def save(self, path) -> None:
        np.savez(path, count=self.count, mean=self.mean, m2=self.m2, minimum=self.minimum, maximum=self.maximum)

    @classmethod
    def load(cls, path) -> "OnlineStats":
        z = np.load(path)
        s = cls(len(z["mean"]))
        s.count, s.mean, s.m2, s.minimum, s.maximum = int(z["count"]), z["mean"], z["m2"], z["minimum"], z["maximum"]
        return s


def score_run(x: np.ndarray, mcs: MCSetup, i: int, chunk_rows: int = _S.DEFAULT_CHUNK_ROWS, guard: Optional[AllocGuard] = None) -> np.ndarray:
    """Scores (float32, one per tile) of run ``i`` for the ``(n, K)`` float32 feature block ``x``. Peak extra memory is one chunk, not the block."""
    w, target = run_params(mcs, i)
    if guard is not None:
        guard.check("run score vector", (x.shape[0],))
        guard.check("scoring chunk", (min(chunk_rows, x.shape[0]), x.shape[1]))
    return _S.score_tiles(x, target, mcs.sigma, mcs.family_cols, w, chunk_rows=chunk_rows).score


def run_stats(x: np.ndarray, mcs: MCSetup, first_run: int, last_run: int, chunk_rows: int = _S.DEFAULT_CHUNK_ROWS, debug: Optional[bool] = None) -> OnlineStats:
    """Online statistics over runs ``first_run`` to ``last_run - 1``. Never stores more than one run's scores."""
    guard = AllocGuard(x.shape[0], last_run - first_run, debug)
    stats = OnlineStats(x.shape[0])
    guard.check("online statistics", (x.shape[0],))
    for i in range(first_run, last_run):
        stats.update(score_run(x, mcs, i, chunk_rows, guard))
    return stats
