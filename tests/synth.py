"""Synthetic Earth grid for tests: we know the right answer because we planted it.

Everything here is SYNTHETIC test data. The numbers are not measurements and not the real
targets; they only need to put the target far from the background so a correct engine finds
the planted tile and a broken one does not.

``make_grid(n, seed, body)`` returns a :class:`SynthGrid` with

* ``df``      one row per tile (rows are shuffled): ``tile_id``, ``lat``, ``lon``, the 15 feature
              columns of ``config/features.yaml`` (float32) and the 3 gate layers;
* ``target``  a synthetic target profile, one row per column the body uses:
              ``column, value, spread, source, evidence_grade`` (see :func:`target_profile`);
* ``truth``   the ground-truth tile ids of every planted tile (below).

The same ``(n, seed, body)`` always gives an identical grid. Default ``n`` is 20,000; it scales to
4,000,000 (about 290 MB as float32).

Planted tiles
-------------
Planted ids are always the lowest ids, ``0..K-1``, so they are the same for every ``n`` and ``seed``
(the *rows* are shuffled, the ids are not). ``family order`` is the order of the families in
``config/weights.yaml`` for that body. Every other id is a background tile.

Moon (8 families, K = 18):

====  ==========================================================================================
id    what it is, and what a correct engine must do
====  ==========================================================================================
0     ANALOG. Equal to the target on every column, passes all gates. Must rank 1st, be ``robust``.
1-8   SINGLE FAMILY. Equal to the target on the columns of one family only (slope, illumination,
      cold_dry, roughness, relief, temperature, composition, thermal_inertia), far on the rest.
      Must NOT be ``robust`` and must rank below the analog and the near misses.
9     GATE TRAP water: equal to the target everywhere, but water/ice share above the gate. Must be
      rejected with the water/ice reason.
10    GATE TRAP ndvi: same, median NDVI above the gate. Must be rejected (vegetation).
11    GATE TRAP built-up: same, built-up or farmed share above the gate. Must be rejected.
12-14 MISSING DATA. Equal to the target everywhere except NaN in one column: the first, the middle
      and the last column the body uses (in weights.yaml order). Must be rejected (missing data).
15-17 NEAR MISS. Equal to the target plus 0.5, 1.0, 1.5 times the target spread on every used
      column, passes all gates. Under the base score they must rank 2nd, 3rd, 4th, in that order.
====  ==========================================================================================

Mars (9 families, K = 19): same layout, with ids 1-9 single-family (composition, water, sunlight,
slope, thermal_inertia, roughness, relief, elevation_density, temperature), 10-12 gate traps
(water, ndvi, built-up), 13-15 missing data, 16-18 near misses.

Gate traps and missing-data tiles are equal to the target apart from the one defect, so without the
gate they would rank at or near the top: that is what makes them a test of the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gsop.config import BODIES, Config, load_config

DEFAULT_N = 20_000

# column: (background mean, background sd, lower bound, upper bound). SYNTHETIC.
BACKGROUND: dict[str, tuple[float, float, float, float]] = {
    "slope_med": (5.0, 3.0, 0.0, 40.0),
    "slope_p90": (12.0, 6.0, 0.0, 60.0),
    "relief": (200.0, 150.0, 0.0, 3000.0),
    "rough_a": (2.0, 1.5, 0.0, 30.0),
    "rough_b": (25.0, 20.0, 0.0, 400.0),
    "sun_max_elev": (60.0, 12.0, 0.0, 90.0),
    "sun_low_share": (0.25, 0.10, 0.0, 1.0),
    "insol_annual": (200.0, 50.0, 0.0, 500.0),
    "t_mean": (285.0, 10.0, 180.0, 330.0),
    "t_coldest": (270.0, 12.0, 150.0, 320.0),
    "ti_proxy": (25.0, 6.0, 0.0, 80.0),
    "dust_index": (0.3, 0.1, 0.0, 1.0),
    "water_index": (0.4, 0.12, 0.0, 1.0),
    "silica_index": (1.0, 0.05, 0.5, 1.5),
    "air_density": (1.0, 0.15, 0.0, 1.3),
}
GATE_COLUMNS = ("water_ice_fraction", "ndvi_median", "builtup_cropland_fraction")
TARGET_SD_AWAY = {"moon": 4.0, "mars": 3.5}  # how many background sd the synthetic target sits from the mean
SPREAD_FRACTION = 0.1  # synthetic target spread = 10% of the background sd
NEAR_MISS_OFFSETS = (0.5, 1.0, 1.5)  # in units of the target spread


@dataclass
class SynthGrid:
    body: str
    df: pd.DataFrame
    target: pd.DataFrame
    truth: dict
    feature_columns: list
    used_columns: list


def _cfg(cfg: Optional[Config]) -> Config:
    return cfg if cfg is not None else load_config(Path(__file__).resolve().parents[1] / "config")


def used_columns(body: str, cfg: Optional[Config] = None) -> list:
    """Feature columns used by the body's families, in weights.yaml order, without repeats."""
    seen: dict[str, None] = {}
    for fam in _cfg(cfg).weights(body).values():
        for col in fam.columns:
            seen.setdefault(col)
    return list(seen)


def target_profile(body: str, cfg: Optional[Config] = None) -> pd.DataFrame:
    """Synthetic target profile (value + spread per used column) for ``body``."""
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}")
    rows = []
    for col in used_columns(body, cfg):
        mu, sd, lo, hi = BACKGROUND[col]
        k = TARGET_SD_AWAY[body]
        direction = 1.0 if mu + 4.0 * sd <= hi else -1.0
        if body == "mars" and lo <= mu - direction * k * sd <= hi:
            direction = -direction
        value = mu + direction * k * sd
        assert lo <= value <= hi, (col, value)
        rows.append(
            {"column": col, "value": value, "spread": SPREAD_FRACTION * sd,
             "source": "synthetic", "evidence_grade": "C"}
        )
    return pd.DataFrame(rows)


def planted_ids(body: str, cfg: Optional[Config] = None) -> dict:
    """Ground-truth ids of the planted tiles for ``body`` (the layout in the module docstring)."""
    nfam = len(_cfg(cfg).weights(body))
    families = list(_cfg(cfg).weights(body))
    nxt = 1
    single = {fam: nxt + i for i, fam in enumerate(families)}
    nxt += nfam
    traps = {"water": nxt, "ndvi": nxt + 1, "builtup": nxt + 2}
    nxt += 3
    cols = used_columns(body, cfg)
    miss_cols = [cols[0], cols[len(cols) // 2], cols[-1]]
    missing = {col: nxt + i for i, col in enumerate(miss_cols)}
    nxt += 3
    near = [nxt + i for i in range(len(NEAR_MISS_OFFSETS))]
    nxt += len(NEAR_MISS_OFFSETS)
    return {
        "analog": 0, "single_family": single, "gate_trap": traps, "missing": missing,
        "near_miss": near, "n_planted": nxt,
    }


def make_grid(n: int = DEFAULT_N, seed: int = 0, body: str = "moon", cfg: Optional[Config] = None) -> SynthGrid:
    cfg = _cfg(cfg)
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}")
    truth = planted_ids(body, cfg)
    if n < truth["n_planted"] + 10:
        raise ValueError(f"n must be at least {truth['n_planted'] + 10} to hold the planted tiles")
    rng = np.random.default_rng(seed)
    columns = list(BACKGROUND)
    used = used_columns(body, cfg)
    target = target_profile(body, cfg)
    tvalue = dict(zip(target["column"], target["value"]))
    tspread = dict(zip(target["column"], target["spread"]))
    gates = cfg.gates.thresholds

    # --- background -----------------------------------------------------------------------------
    feats = np.empty((n, len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        mu, sd, lo, hi = BACKGROUND[col]
        feats[:, j] = np.clip(rng.normal(mu, sd, size=n), lo, hi)
    col_index = {c: j for j, c in enumerate(columns)}
    gate = np.empty((n, 3), dtype=np.float32)
    gate[:, 0] = rng.uniform(0.0, 0.5 * gates["water_ice_fraction"], n)
    gate[:, 1] = rng.uniform(-0.1, 0.5 * gates["ndvi_median"], n)
    gate[:, 2] = rng.uniform(0.0, 0.5 * gates["builtup_cropland_fraction"], n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n))).astype(np.float32)
    lon = rng.uniform(-180.0, 180.0, n).astype(np.float32)

    def far(col: str) -> float:  # clearly away from the target on the other side of the mean
        mu, sd, lo, hi = BACKGROUND[col]
        away = 1.0 if tvalue.get(col, mu) < mu else -1.0
        return float(np.clip(mu + away * 1.0 * sd, lo, hi))

    def set_equal_target(i: int) -> None:
        for col in used:
            feats[i, col_index[col]] = tvalue[col]
        gate[i] = (0.0, 0.0, 0.0)

    # --- planted tiles (ids 0..K-1) -------------------------------------------------------------
    set_equal_target(truth["analog"])
    for fam, i in truth["single_family"].items():
        for col in used:
            feats[i, col_index[col]] = far(col)
        for col in cfg.weights(body)[fam].columns:
            feats[i, col_index[col]] = tvalue[col]
        gate[i] = (0.0, 0.0, 0.0)
    for kind, i in truth["gate_trap"].items():
        set_equal_target(i)
        slot = {"water": 0, "ndvi": 1, "builtup": 2}[kind]
        thr = {"water": gates["water_ice_fraction"], "ndvi": gates["ndvi_median"],
               "builtup": gates["builtup_cropland_fraction"]}[kind]
        gate[i, slot] = min(1.0, thr + 0.4)
    for col, i in truth["missing"].items():
        set_equal_target(i)
        feats[i, col_index[col]] = np.nan
    for off, i in zip(NEAR_MISS_OFFSETS, truth["near_miss"]):
        set_equal_target(i)
        for col in used:
            feats[i, col_index[col]] = tvalue[col] + off * tspread[col]

    # --- assemble and shuffle the rows (ids stay attached to their tiles) -----------------------
    df = pd.DataFrame(feats, columns=columns)
    for k, name in enumerate(GATE_COLUMNS):
        df[name] = gate[:, k]
    df.insert(0, "lon", lon)
    df.insert(0, "lat", lat)
    df.insert(0, "tile_id", np.arange(n, dtype=np.int64))
    order = rng.permutation(n)
    df = df.iloc[order].reset_index(drop=True)
    return SynthGrid(body, df, target, truth, columns, used)
