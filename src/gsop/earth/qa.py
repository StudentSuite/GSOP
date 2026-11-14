"""Earth feature QA (issue B20): every check has a pass or fail, and any failure blocks the Moon and Mars runs.

``run_qa(df, cfg)`` is pure (a DataFrame in, a list of :class:`Check` out). ``scripts/earth_qa.py`` adds histograms and
maps and writes ``docs/EARTH_QA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from gsop.config import Config
from gsop.earth.build_parquet import FEATURE_COLUMNS
from gsop.earth.sanity import SANITY_RANGES

NODATA_SENTINELS = (-9999.0, -32768.0, 65535.0)  # raw nodata values an export could leak through
ROUGH_B_GE_A_MIN_SHARE = 0.8  # TUNABLE, NOT SOURCED: "most tiles" for rough_b >= rough_a
BAD_AREA_DEGREES = 10  # failing tiles are binned into 10 x 10 degree cells for the "known bad areas" list


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    bad_mask: Optional[pd.Series] = None  # tiles that failed (for the bad-areas list), if the check is per tile


@dataclass
class FeatureStats:
    name: str
    unavailable: bool
    pct_nan: float
    pct_sentinel: float
    minimum: float
    median: float
    maximum: float


def feature_stats(df: pd.DataFrame, columns=FEATURE_COLUMNS) -> list[FeatureStats]:
    out = []
    for c in columns:
        v = df[c].astype("float64")
        finite = v[np.isfinite(v)]
        out.append(FeatureStats(
            c, bool(v.isna().all()), float(v.isna().mean() * 100),
            float(np.isin(v, NODATA_SENTINELS).mean() * 100),
            float(finite.min()) if len(finite) else np.nan, float(finite.median()) if len(finite) else np.nan,
            float(finite.max()) if len(finite) else np.nan,
        ))
    return out


def run_qa(df: pd.DataFrame, cfg: Config) -> list[Check]:
    checks: list[Check] = []
    warn = cfg.features.earth_build["nan_warn_fraction"]

    # 1. plausibility bounds, one check per available feature (and gate / flag column that has a bound)
    bound_cols = {"slope_med": "slope_med", "slope_p90": "slope_p90", "relief": "relief", "rough_a": "rough_a",
                  "rough_b": "rough_b", "t_mean": "t_mean", "t_coldest": "t_coldest", "ti_proxy": "ti_proxy",
                  "water_index": "water_index", "silica_index": "silica_index", "elevation_mean": "elevation_mean",
                  "water_ice_fraction": "gate_water_ice_frac", "ndvi_median": "gate_ndvi_med",
                  "builtup_cropland_fraction": "gate_built_crop_frac", "dust_index": None, "air_density": None}
    for col, sanity_key in bound_cols.items():
        if col not in df.columns or df[col].isna().all():
            continue
        lo, hi, why = SANITY_RANGES.get(sanity_key, (0.0, 1.0, "scaled 0-1")) if sanity_key else (
            (0.0, 1.0, "scaled 0-1") if col == "dust_index" else (0.0, 1.3, "kg/m3, Earth surface to 9 km"))
        v = df[col]
        bad = (v < lo) | (v > hi)
        checks.append(Check(f"{col} within [{lo}, {hi}] ({why})", not bool(bad.any()),
                            f"{int(bad.sum()):,} tile(s) outside", bad))

    # 2. specific cross-feature checks
    both = df[["t_coldest", "t_mean"]].dropna()
    bad = df["t_coldest"] > df["t_mean"] + 1e-3
    checks.append(Check("t_coldest <= t_mean", not bool(bad.fillna(False).any()), f"{int(bad.fillna(False).sum()):,} tile(s) with the coldest month warmer than the mean", bad.fillna(False)))
    rough = df[["rough_a", "rough_b"]].dropna()
    share = float((rough.rough_b >= rough.rough_a).mean()) if len(rough) else float("nan")
    checks.append(Check(f"rough_b >= rough_a on at least {ROUGH_B_GE_A_MIN_SHARE:.0%} of tiles", bool(share >= ROUGH_B_GE_A_MIN_SHARE),
                        f"{share:.1%} of tiles", (df["rough_b"] < df["rough_a"]).fillna(False)))
    checks.append(Check("slope_p90 >= slope_med", not bool((df["slope_p90"] < df["slope_med"] - 1e-3).fillna(False).any()),
                        f"{int((df['slope_p90'] < df['slope_med'] - 1e-3).fillna(False).sum()):,} tile(s) where the 90th percentile is below the median",
                        (df["slope_p90"] < df["slope_med"] - 1e-3).fillna(False)))

    # 3. missing data and nodata sentinels
    for s in feature_stats(df):
        if s.unavailable:
            checks.append(Check(f"{s.name} built", False, "UNAVAILABLE: every tile is NaN"))
        else:
            checks.append(Check(f"{s.name} NaN share below {warn:.0%}", s.pct_nan / 100 <= warn, f"{s.pct_nan:.1f}% NaN"))
        if s.pct_sentinel > 0:
            checks.append(Check(f"{s.name} has no nodata sentinel values", False, f"{s.pct_sentinel:.2f}% of tiles equal a raw nodata value"))
    checks.append(Check("tile_id is unique", bool(df["tile_id"].is_unique), f"{int(df['tile_id'].duplicated().sum()):,} duplicate id(s)"))
    return checks


def bad_areas(df: pd.DataFrame, checks: list[Check], top: int = 5) -> dict[str, list[tuple[int, int, int]]]:
    """For each failing per-tile check, the ``top`` 10-degree cells with the most failing tiles: (lat, lon, count)."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for c in checks:
        if c.passed or c.bad_mask is None or not bool(c.bad_mask.any()):
            continue
        sub = df.loc[c.bad_mask.index[c.bad_mask.to_numpy()]]
        cells = (np.floor(sub.lat / BAD_AREA_DEGREES) * BAD_AREA_DEGREES).astype(int).astype(str) + "," + \
                (np.floor(sub.lon / BAD_AREA_DEGREES) * BAD_AREA_DEGREES).astype(int).astype(str)
        counts = cells.value_counts().head(top)
        out[c.name] = [(int(k.split(",")[0]), int(k.split(",")[1]), int(n)) for k, n in counts.items()]
    return out


def overall_pass(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
