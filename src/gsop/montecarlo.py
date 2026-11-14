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
