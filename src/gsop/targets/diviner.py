"""Moon temperature features at the targets from the LRO Diviner Polar Cumulative Product (issue B26).

Product (verified on the PDS, see ``docs/PROVENANCE.md``): ``urn:nasa:pds:lro_diviner_derived1``, collection
``data_derived_pcp``, south polar seasonal set ``seasonal/slon/pols/pcp_avg_tbol_pols_ls{LS}_slon{NN}_240.tab``: 6 Ls bins
x 24 subsolar-longitude bins = **144 ASCII files of about 214 MB each (30.7 GB)**, 240 m polar stereographic bins covering
80 to 90 S. Each file is a header line ``x, y, clon, clat, ltim, tbol`` followed by one line per bin; ``x`` and ``y`` are polar
stereographic coordinates normalised by the mean lunar radius 1737.4 km, ``ltim`` is local time (h), ``tbol`` the average
bolometric brightness temperature (K). The files are text, so they cannot be read by window: **this module works on files that
a person has already downloaded to a folder; nothing is downloaded here**, and with no folder every Diviner row is
``unavailable`` with the reason.

Definitions (the Moon has no dry season, no calendar month, and Diviner gives bolometric, not kinetic, temperature):

* ``t_mean``: for each of the 6 Ls bins average the tile's bolometric temperature over its 24 subsolar-longitude files, then
  average the 6 Ls bins.
* ``t_coldest``: the lowest of the 6 Ls-bin means. **This is the coldest Ls bin (about two months), not the coldest calendar
  month used on Earth**: a mismatch, flagged.
* ``ti_proxy``: swing of the all-year, all-file mean temperature by local time: bin every record of the tile by local time
  (1 h bins), average within a bin, take max minus min. This replaces the Earth "dry-season swing" because the Moon has no
  seasons of rain.
* ``silica_index``: **unavailable**. Bolometric temperature carries no Christiansen-feature information, and the Christiansen
  feature needs the narrow-band channels at suitable illumination, which the polar low Sun does not give.

All values are medians over the 6 km tiles (3 km stride) of the 15 km target window; spread is IQR / 1.349.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gsop.config import Config

MOON_RADIUS_M = 1_737_400.0
FILE_RE = re.compile(r"ls(?P<ls>\d{3})_slon(?P<slon>\d{2})", re.IGNORECASE)
LTIM_BIN_H = 1.0
UNAVAILABLE_REASON = (
    "Diviner south polar seasonal product not downloaded: 144 ASCII files (6 Ls bins x 24 subsolar-longitude bins) of about 214 MB "
    "each, 30.7 GB in total, not readable by window. Download them to a folder and pass it (see docs/SPEC.md)"
)


def polar_xy_m(lat: float, lon_east: float) -> tuple[float, float]:
    """South polar stereographic position in metres (sphere radius 1,737,400 m): x = rho sin(lon), y = rho cos(lon)."""
    rho = 2.0 * MOON_RADIUS_M * math.tan(math.radians(90.0 + lat) / 2.0)
    lon = math.radians(lon_east)
    return rho * math.sin(lon), rho * math.cos(lon)


def parse_name(path: Path) -> tuple[int, int]:
    m = FILE_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot read Ls and subsolar-longitude bin from file name {Path(path).name}")
    return int(m["ls"]), int(m["slon"])


def read_window(path: Path, cx_m: float, cy_m: float, half_m: float, chunksize: int = 500_000) -> pd.DataFrame:
    """Records of one file inside the square window (x_m, y_m, ltim, tbol). Streams the file; never loads it whole."""
    parts = []
    for chunk in pd.read_csv(path, skipinitialspace=True, chunksize=chunksize, usecols=["x", "y", "ltim", "tbol"]):
        x, y = chunk["x"] * MOON_RADIUS_M, chunk["y"] * MOON_RADIUS_M
        keep = (x - cx_m).abs().le(half_m) & (y - cy_m).abs().le(half_m)
        if keep.any():
            parts.append(pd.DataFrame({"x_m": x[keep], "y_m": y[keep], "ltim": chunk.loc[keep, "ltim"], "tbol": chunk.loc[keep, "tbol"]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["x_m", "y_m", "ltim", "tbol"])


def tile_edges(cx_m: float, cy_m: float, window_m: float, tile_m: float, stride_m: float) -> list[tuple[float, float]]:
    """Lower-left corners of the tiles of a ``window_m`` square centred on (cx, cy), ``stride_m`` apart."""
    n = int(round((window_m - tile_m) / stride_m)) + 1
    x0, y0 = cx_m - window_m / 2, cy_m - window_m / 2
    return [(x0 + i * stride_m, y0 + j * stride_m) for j in range(n) for i in range(n)]


def temperature_features(files: list[Path], cx_m: float, cy_m: float, window_m: float, tile_m: float, stride_m: float, divisor: float) -> dict:
    """t_mean, t_coldest, ti_proxy (value, spread, n_tiles) from the supplied files."""
    corners = tile_edges(cx_m, cy_m, window_m, tile_m, stride_m)
    by_ls: dict[int, list[np.ndarray]] = {}  # Ls bin -> per-file tile mean arrays
    pooled = [[] for _ in corners]  # per tile: (ltim, tbol) over all files
    for path in files:
        ls, _ = parse_name(path)
        df = read_window(path, cx_m, cy_m, window_m / 2)
        means = np.full(len(corners), np.nan)
        for k, (x0, y0) in enumerate(corners):
            inside = df[(df.x_m >= x0) & (df.x_m < x0 + tile_m) & (df.y_m >= y0) & (df.y_m < y0 + tile_m)]
            if len(inside):
                means[k] = inside.tbol.mean()
                pooled[k].append(inside[["ltim", "tbol"]])
        by_ls.setdefault(ls, []).append(means)
    ls_means = np.array([np.nanmean(np.vstack(v), axis=0) for _, v in sorted(by_ls.items())])  # (n_ls, n_tiles)
    t_mean = np.nanmean(ls_means, axis=0)
    t_cold = np.nanmin(ls_means, axis=0)
    swing = np.full(len(corners), np.nan)
    for k, frames in enumerate(pooled):
        if not frames:
            continue
        all_rec = pd.concat(frames)
        curve = all_rec.groupby(np.floor(all_rec.ltim / LTIM_BIN_H)).tbol.mean()
        if len(curve) >= 2:
            swing[k] = curve.max() - curve.min()

    def summarise(v: np.ndarray) -> dict:
        v = v[np.isfinite(v)]
        if not v.size:
            return {"value": float("nan"), "spread": float("nan"), "n_tiles": 0}
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return {"value": float(med), "spread": float((q3 - q1) / divisor), "n_tiles": int(v.size)}

    return {"t_mean": summarise(t_mean), "t_coldest": summarise(t_cold), "ti_proxy": summarise(swing),
            "n_files": len(files), "n_ls_bins": len(by_ls)}


COLUMN_NOTES = {
    "t_mean": ("B", "Diviner PCP bolometric brightness temperature, mean of the 6 Ls-bin means (not kinetic temperature)"),
    "t_coldest": ("B", "coldest of the 6 Ls-bin means (about two months), NOT the coldest calendar month used on Earth"),
    "ti_proxy": ("C", "all-year local-time swing (max minus min of 1 h bins); replaces the Earth dry-season swing; a proxy, not thermal inertia"),
}


def moon_temperature_rows(cfg: Config, pcp_dir: Optional[Path] = None) -> list[dict]:
    """Rows for t_mean, t_coldest, ti_proxy and silica_index at the Moon targets (``unavailable`` where not built)."""
    from gsop.targets.moon import SITE_FOR_TARGET

    ts = cfg.target_settings
    rows: list[dict] = []
    files = sorted(Path(pcp_dir).glob("pcp_avg_tbol_pols_ls*_slon*_240.tab")) if pcp_dir else []
    complete = len(files) == 144
    for name, target in cfg.targets("moon").items():
        if name not in SITE_FOR_TARGET or target.lat is None:
            continue
        window = (f"{ts.target_window_km:g} km window, {cfg.target_settings.tile_stride_km:g} km stride, {cfg.features.operators.tile_m / 1000:g} km tiles")
        prof = None
        reason = UNAVAILABLE_REASON
        if files and not complete:
            reason = f"only {len(files)} of 144 Diviner files found in {pcp_dir}; an incomplete seasonal set would bias the means"
        if complete:
            cx, cy = polar_xy_m(target.lat, target.lon_e)
            prof = temperature_features(files, cx, cy, ts.target_window_km * 1000, cfg.features.operators.tile_m,
                                        ts.tile_stride_km * 1000, ts.spread_iqr_divisor)
        for col, (grade, note) in COLUMN_NOTES.items():
            p = prof[col] if prof else None
            ok = bool(p and np.isfinite(p["value"]))
            rows.append({
                "target": name, "column": col, "value": p["value"] if ok else float("nan"), "spread": p["spread"] if ok else float("nan"),
                "source": "LRO Diviner Polar Cumulative Product (PDS urn:nasa:pds:lro_diviner_derived1, data_derived_pcp)",
                "evidence_grade": grade, "window": window, "unavailable": not ok, "note": note if ok else f"unavailable: {reason}",
            })
        rows.append({
            "target": name, "column": "silica_index", "value": float("nan"), "spread": float("nan"),
            "source": "LRO Diviner Christiansen feature", "evidence_grade": "C", "window": window, "unavailable": True,
            "note": "unavailable: the polar product is bolometric only; the Christiansen feature needs the narrow-band channels under suitable illumination, which low polar Sun does not give",
        })
    return rows
