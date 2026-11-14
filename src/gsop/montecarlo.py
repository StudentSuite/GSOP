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


# ===========================================================================================================
# Per-run top 1%, 5%, 10% membership (issue C18)
# ===========================================================================================================
MAX_RUNS_UINT16 = np.iinfo(np.uint16).max


def top_k_sizes(n_tiles: int, fractions) -> list[int]:
    """Number of tiles in each top set: ``ceil(fraction * n)``, at least 1."""
    return [max(1, int(np.ceil(f * n_tiles))) for f in fractions]


def top_sets(score: np.ndarray, fractions) -> list[np.ndarray]:
    """Indices of the top ``fraction`` of tiles for each fraction (nested: the 1% set is inside the 5% set, inside the 10% set), in ranked order.

    Tie rule (deterministic, independent of the algorithm's internals): tiles are ordered by score descending, and among equal scores the **lower
    index wins**, and tile order is ``tile_id`` ascending. Uses ``np.partition`` to find the cut value in O(N), then sorts only the top 10%.
    """
    n = len(score)
    if np.isnan(score).any():
        raise ValueError("NaN scores cannot be ranked (np.partition would put them first); gate missing tiles out before the Monte Carlo")
    sizes = top_k_sizes(n, fractions)
    kmax = max(sizes)
    if kmax >= n:
        cand = np.arange(n)
    else:
        thr = np.partition(score, n - kmax)[n - kmax]  # the kmax-th largest value
        strict = np.flatnonzero(score > thr)
        ties = np.flatnonzero(score == thr)[: kmax - len(strict)]  # lowest indices first
        cand = np.sort(np.concatenate([strict, ties]))
    ranked = cand[np.lexsort((cand, -score[cand]))]  # primary: score descending, then index ascending
    return [ranked[:k] for k in sizes]


def add_membership(counters: np.ndarray, score: np.ndarray, fractions) -> None:
    """Increment the uint16 ``counters`` (n_tiles, n_fractions) for the tiles in each of this run's top sets."""
    for j, idx in enumerate(top_sets(score, fractions)):
        counters[idx, j] += 1


@dataclass
class Pass1Block:
    """Result of a block of consecutive runs: statistics and, per segment between checkpoints, top-set counters."""

    first_run: int
    last_run: int
    stats: OnlineStats
    counters: dict  # segment index -> uint16 (n_tiles, n_fractions)


def run_block(x: np.ndarray, mcs: MCSetup, first_run: int, last_run: int, fractions, segment_edges=(), chunk_rows: int = _S.DEFAULT_CHUNK_ROWS,
              debug: Optional[bool] = None) -> Pass1Block:
    """Runs ``first_run`` to ``last_run - 1``: online statistics plus membership counters. ``segment_edges`` are the ascending run indices that START a
    new segment (the convergence checkpoints 500, 1000, 1500: runs 0-499 are segment 0, 500-999 segment 1, ...); a run goes to the counters of the
    segment it falls in, so cumulative counts at any checkpoint are exact."""
    n = x.shape[0]
    if last_run - first_run > MAX_RUNS_UINT16:
        raise ValueError("more runs than a uint16 counter can hold")
    guard = AllocGuard(n, last_run - first_run, debug)
    stats = OnlineStats(n)
    edges = list(segment_edges)
    counters: dict[int, np.ndarray] = {}
    for i in range(first_run, last_run):
        seg = int(np.searchsorted(edges, i, side="right")) if edges else 0
        if seg not in counters:
            counters[seg] = np.zeros((n, len(fractions)), dtype=np.uint16)
        score = score_run(x, mcs, i, chunk_rows, guard)
        stats.update(score)
        add_membership(counters[seg], score, fractions)
    return Pass1Block(first_run, last_run, stats, counters)


# ===========================================================================================================
# Pass 2: exact percentiles for the candidate tiles and their rank-percentile range (issue C19)
# ===========================================================================================================
from pathlib import Path  # noqa: E402

RANK_SCALE = 100.0  # ranks are stored as uint16 in hundredths of a percentile point (0..10000)


def select_candidates(mean_score: np.ndarray, count_top5: np.ndarray, fraction: float) -> np.ndarray:
    """Candidate tiles: the union of the top ``fraction`` of tiles by mean score and by membership count of the top 5% set (ties by mean, then lower index).

    Returned as sorted tile indices (positions in the tile block, not tile ids)."""
    n = len(mean_score)
    k = max(1, int(np.ceil(fraction * n)))
    by_mean = np.lexsort((np.arange(n), -mean_score))[:k]
    by_freq = np.lexsort((np.arange(n), -mean_score, -count_top5.astype(np.int64)))[:k]
    return np.union1d(by_mean, by_freq)


def candidate_scores_and_ranks(x: np.ndarray, mcs: MCSetup, i: int, cand_idx: np.ndarray, chunk_rows: int = _S.DEFAULT_CHUNK_ROWS,
                               guard: Optional[AllocGuard] = None) -> tuple[np.ndarray, np.ndarray]:
    """Run ``i``'s score of each candidate (identical to pass 1: same seeds, same code) and its rank percentile among ALL tiles of that run.

    Rank percentile = 100 * (number of tiles with a strictly lower score) / N, found by sorting the run's scores once and ``np.searchsorted``."""
    score = score_run(x, mcs, i, chunk_rows, guard)
    ordered = np.sort(score)
    cand = score[cand_idx]
    below = np.searchsorted(ordered, cand, side="left")
    rank = np.rint(below * (100.0 * RANK_SCALE / len(score))).astype(np.uint16)
    return cand, rank


class Pass2Store:
    """Disk-backed (runs x candidates) arrays for candidate scores (float32) and rank percentiles (uint16 hundredths). ``.npy`` files opened as
    memmaps so workers write their own rows and nothing has to fit in RAM."""

    def __init__(self, directory: Path, runs: int, n_candidates: int, create: bool = True) -> None:
        self.dir = Path(directory)
        self.runs, self.n = runs, n_candidates
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scores_path, self.rank_path = self.dir / "cand_scores.npy", self.dir / "cand_rank.npy"
        if create and not (self.scores_path.exists() and self.rank_path.exists()):
            np.lib.format.open_memmap(self.scores_path, mode="w+", dtype=np.float32, shape=(runs, n_candidates)).flush()
            np.lib.format.open_memmap(self.rank_path, mode="w+", dtype=np.uint16, shape=(runs, n_candidates)).flush()

    def open(self) -> tuple[np.ndarray, np.ndarray]:
        return np.load(self.scores_path, mmap_mode="r+"), np.load(self.rank_path, mmap_mode="r+")

    def block_marker(self, first: int, last: int) -> Path:
        return self.dir / f"done_{first}_{last}.txt"


def pass2_block(x: np.ndarray, mcs: MCSetup, cand_idx: np.ndarray, first: int, last: int, store: Pass2Store, chunk_rows: int = _S.DEFAULT_CHUNK_ROWS,
                debug: Optional[bool] = None) -> None:
    """Fill rows ``first`` to ``last - 1`` of the store. Skips a block whose completion marker exists (resume)."""
    marker = store.block_marker(first, last)
    if marker.exists():
        return
    scores, ranks = store.open()
    guard = AllocGuard(x.shape[0], last - first, debug)
    for i in range(first, last):
        s, r = candidate_scores_and_ranks(x, mcs, i, cand_idx, chunk_rows, guard)
        scores[i], ranks[i] = s, r
    scores.flush(), ranks.flush()
    marker.write_text("done\n", encoding="utf-8")


def summarise_candidates(store: Pass2Store, chunk_cols: int = 20_000) -> dict:
    """Exact per-candidate ``mc_median``, ``mc_p05``, ``mc_p95`` of the score and ``rank_pct_p05/p50/p95`` over all runs (numpy linear percentiles).

    Reads the memmap in blocks of candidate columns so memory stays at (runs x chunk_cols)."""
    scores, ranks = store.open()
    out = {k: np.empty(store.n, dtype=np.float32) for k in ("mc_median", "mc_p05", "mc_p95", "rank_pct_p05", "rank_pct_p50", "rank_pct_p95")}
    for a in range(0, store.n, chunk_cols):
        b = min(a + chunk_cols, store.n)
        s = np.asarray(scores[:, a:b], dtype=np.float32)
        r = np.asarray(ranks[:, a:b], dtype=np.float64) / RANK_SCALE
        out["mc_p05"][a:b], out["mc_median"][a:b], out["mc_p95"][a:b] = np.percentile(s, [5, 50, 95], axis=0)
        out["rank_pct_p05"][a:b], out["rank_pct_p50"][a:b], out["rank_pct_p95"][a:b] = np.percentile(r, [5, 50, 95], axis=0)
    return out


# ===========================================================================================================
# Outputs: robust flag, results table, convergence (issue C20), and the driver
# ===========================================================================================================
import time  # noqa: E402
from collections.abc import Mapping  # noqa: E402

import pandas as pd  # noqa: E402

DEFAULT_BLOCK_RUNS = 125  # runs per block; fixed (never derived from the worker count) so merged float statistics do not depend on workers


@dataclass(frozen=True)
class MCParams:
    """Run-level settings, all from ``config/run.yaml`` except ``runs`` which ``--runs`` may override."""

    runs: int
    fractions: tuple[float, ...]
    robust_fraction: float
    robust_min_share: float
    candidate_fraction: float
    checkpoints: tuple[int, ...]
    block_runs: int = DEFAULT_BLOCK_RUNS

    @classmethod
    def from_config(cls, cfg: Config, runs: Optional[int] = None, block_runs: int = DEFAULT_BLOCK_RUNS) -> "MCParams":
        r = cfg.run
        return cls(r.runs if runs is None else int(runs), tuple(r.top_fractions), r.robust_top_fraction, r.robust_min_run_share, r.candidate_fraction,
                   tuple(r.convergence_checkpoints), block_runs)

    @property
    def robust_index(self) -> int:
        try:
            return list(self.fractions).index(self.robust_fraction)
        except ValueError:
            raise ValueError(f"robust top fraction {self.robust_fraction} is not one of top_fractions {list(self.fractions)}") from None


def scaled_checkpoints(runs: int, checkpoints) -> list[int]:
    """Convergence checkpoints for ``runs`` runs: the configured ones when ``runs`` equals the largest (2,000), else scaled in proportion and
    de-duplicated; always ends with ``runs``."""
    top = max(checkpoints)
    cps = sorted({max(1, int(round(c * runs / top))) for c in checkpoints} | {runs})
    return [c for c in cps if c <= runs]


def block_plan(runs: int, block_runs: int, checkpoints: list[int]) -> list[tuple[int, int]]:
    """Consecutive blocks ``(first, last)`` covering ``0..runs``: split at every multiple of ``block_runs`` and at every checkpoint, so a block never spans
    two segments. Depends only on ``runs``, ``block_runs`` and the checkpoints (never on the worker count)."""
    cuts = sorted({0, runs, *range(block_runs, runs, block_runs), *[c for c in checkpoints if 0 < c < runs]})
    return list(zip(cuts[:-1], cuts[1:]))


def segment_starts(checkpoints: list[int]) -> list[int]:
    """Run indices that start segments 1, 2, ...: the checkpoints except the last (which is the end of the run)."""
    return list(checkpoints[:-1])


def merge_pass1(blocks: "list[Pass1Block]", n_tiles: int, n_segments: int, n_fractions: int) -> tuple[OnlineStats, list[np.ndarray]]:
    """Merge blocks in run order: statistics (fixed merge order) and the per-segment counters."""
    stats = OnlineStats(n_tiles)
    seg = [np.zeros((n_tiles, n_fractions), dtype=np.uint16) for _ in range(n_segments)]
    for b in sorted(blocks, key=lambda b: b.first_run):
        stats.merge(b.stats)
        for k, c in b.counters.items():
            seg[k] += c
    return stats, seg


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def top1_set(counts: np.ndarray, n: int, fractions) -> np.ndarray:
    """The top-1% set by membership: tiles ranked by count of the top-1% set, then of the 5% and 10% sets, then lower index."""
    k = top_k_sizes(n, fractions)[0]
    order = np.lexsort((np.arange(n), -counts[:, 2].astype(np.int64), -counts[:, 1].astype(np.int64), -counts[:, 0].astype(np.int64)))
    return order[:k]


def convergence_report(seg_counters: list[np.ndarray], checkpoints: list[int], fractions) -> dict:
    """Top-1% set overlap (Jaccard) between the set after each checkpoint (cumulative counts of the first ``c`` runs) and the final set."""
    n = seg_counters[0].shape[0]
    cumulative = np.zeros_like(seg_counters[0], dtype=np.int64)
    sets = []
    for j in range(len(checkpoints)):
        cumulative += seg_counters[j] if j < len(seg_counters) else 0
        sets.append(top1_set(cumulative, n, fractions))
    final = sets[-1]
    return {"definition": "top 1% of tiles by membership count of the top-1% set (then 5%, 10%, then lower index)",
            "final_runs": checkpoints[-1], "checkpoints": [{"runs": c, "jaccard_top1_vs_final": jaccard(s, final)} for c, s in zip(checkpoints, sets)]}


def build_results_table(tile_ids: np.ndarray, stats: OnlineStats, counts: np.ndarray, cand_idx: np.ndarray, pass2: Optional[Mapping[str, np.ndarray]],
                        params: MCParams) -> pd.DataFrame:
    """One row per tile that ever reached the top 10% or is a candidate. ``freq_topK`` = share of runs in the top K%; ``robust`` = ``freq`` of the
    robust fraction (5%) at least ``robust_min_share`` (90%). Pass 2 columns are NaN for non-candidates."""
    n = len(tile_ids)
    freq = counts.astype(np.float32) / np.float32(params.runs)
    is_cand = np.zeros(n, dtype=bool)
    is_cand[cand_idx] = True
    keep = np.flatnonzero((counts[:, -1] > 0) | is_cand)
    cols = {"tile_id": tile_ids[keep], "mc_mean": stats.mean[keep].astype(np.float32), "mc_std": np.sqrt(stats.variance[keep]).astype(np.float32)}
    for j, f in enumerate(params.fractions):
        cols[f"freq_top{int(round(f * 100))}"] = freq[keep, j]
    cols["robust"] = freq[keep, params.robust_index] >= params.robust_min_share
    cols["is_candidate"] = is_cand[keep]
    pos = np.full(n, -1, dtype=np.int64)
    pos[cand_idx] = np.arange(len(cand_idx))
    for name in ("mc_median", "mc_p05", "mc_p95", "rank_pct_p05", "rank_pct_p50", "rank_pct_p95"):
        col = np.full(len(keep), np.nan, dtype=np.float32)
        if pass2 is not None:
            m = pos[keep] >= 0
            col[m] = pass2[name][pos[keep][m]]
        cols[name] = col
    return pd.DataFrame(cols)


@dataclass
class MCResult:
    table: pd.DataFrame
    convergence: dict
    timing: dict
    candidates: int


def compute_pass1_block(task, shared):
    """Worker entry point (module level so it works under spawn): ``task = (mcs, first, last, fractions, segment_edges, chunk_rows)``."""
    mcs, first, last, fractions, edges, chunk_rows = task
    return run_block(shared["x"], mcs, first, last, fractions, edges, chunk_rows)


def simulate(x: np.ndarray, tile_ids: np.ndarray, mcs: MCSetup, params: MCParams, work_dir: Optional[Path] = None, chunk_rows: int = _S.DEFAULT_CHUNK_ROWS,
             log=print) -> MCResult:
    """The whole Monte Carlo for one target and profile (serial driver: pass 1, candidate selection, pass 2, results, convergence)."""
    n = x.shape[0]
    cps = scaled_checkpoints(params.runs, params.checkpoints)
    edges = segment_starts(cps)
    plan = block_plan(params.runs, params.block_runs, cps)
    t0 = time.perf_counter()
    blocks = [compute_pass1_block((mcs, a, b, params.fractions, edges, chunk_rows), {"x": x}) for a, b in plan]
    stats, seg = merge_pass1(blocks, n, len(cps), len(params.fractions))
    counts = sum(seg[1:], seg[0]) if len(seg) > 1 else seg[0]
    t1 = time.perf_counter()
    cand = select_candidates(stats.mean, counts[:, params.robust_index], params.candidate_fraction)
    import tempfile

    with tempfile.TemporaryDirectory() if work_dir is None else _NullContext(work_dir) as d:
        store = Pass2Store(Path(d), params.runs, len(cand))
        for a, b in plan:
            pass2_block(x, mcs, cand, a, b, store, chunk_rows)
        summary = summarise_candidates(store)
    t2 = time.perf_counter()
    table = build_results_table(tile_ids, stats, counts.astype(np.uint16), cand, summary, params)
    conv = convergence_report(seg, cps, params.fractions)
    log(f"[mc] {mcs.target}/{mcs.profile}: {params.runs} runs on {n:,} tiles; pass 1 {t1 - t0:.1f} s, pass 2 {t2 - t1:.1f} s, {len(cand):,} candidates, "
        f"{int(table.robust.sum()):,} robust")
    return MCResult(table, conv, {"pass1_seconds": round(t1 - t0, 3), "pass2_seconds": round(t2 - t1, 3), "runs": params.runs, "tiles": n}, len(cand))


class _NullContext:
    """``with`` wrapper around an existing directory (not deleted on exit)."""

    def __init__(self, path) -> None:
        self.path = str(path)

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, *exc) -> None:
        return None
