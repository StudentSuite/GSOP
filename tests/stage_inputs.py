"""Synthetic inputs in the real file formats for the stage tests: an Earth feature parquet and a targets parquet."""

from pathlib import Path

import numpy as np
import pandas as pd

import synth
from gsop import targets as TG
from gsop.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
MARS_CHECK_ROWS = {"elevation_mean": -2500.0, "thermal_inertia_themis": 313.0}


def target_rows(body, cfg=CFG, mars_elevation=-2500.0, only=None, drop_columns=(), away=None):
    """Rows in the targets_{body}.parquet format for every scored target, all with the synthetic profile of ``body``."""
    prof = synth.target_profile(body, cfg, away)
    rows = []
    for name, t in cfg.targets(body).items():
        if t.lat is None or (only and name not in only):
            continue
        for _, r in prof.iterrows():
            if r["column"] in drop_columns:
                rows.append({"target": name, "column": r["column"], "value": np.nan, "spread": np.nan, "source": "synthetic", "evidence_grade": "C",
                             "window": "w", "unavailable": True, "note": "unavailable: dropped by the test"})
            else:
                rows.append({"target": name, "column": r["column"], "value": float(r["value"]), "spread": float(r["spread"]), "source": "synthetic",
                             "evidence_grade": "C", "window": "w", "unavailable": False, "note": ""})
        if body == "mars":
            for col, val in {**MARS_CHECK_ROWS, "elevation_mean": mars_elevation}.items():
                rows.append({"target": name, "column": col, "value": val, "spread": 1.0, "source": "synthetic", "evidence_grade": "A", "window": "w",
                             "unavailable": False, "note": ""})
    return rows


def write_inputs(tmp: Path, body: str, n: int = 5000, seed: int = 0, cfg=CFG, away=None, **kw):
    """Write ``earth_features_6km.parquet`` and ``targets_{body}.parquet`` under ``tmp/data/processed``; return (root, grid, rows)."""
    g = synth.make_grid(n, seed=seed, body=body, cfg=cfg, away=away)
    out = tmp / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df = g.df.copy()
    df["x_m"], df["y_m"], df["land"], df["land_frac"] = 0.0, 0.0, True, np.float32(1.0)
    df.to_parquet(out / "earth_features_6km.parquet")
    rows = target_rows(body, cfg, away=away, **kw)
    frame = TG.to_frame(rows, cfg, body)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pandas(frame, schema=TG.SCHEMA, preserve_index=False), out / f"targets_{body}.parquet")
    return tmp, g, rows
