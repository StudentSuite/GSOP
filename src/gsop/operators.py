"""The one terrain operator for Earth, Moon and Mars (numpy). Definition: ``docs/SPEC.md``, section Operators.

Public functions (signatures are stable; other issues import them):

* :func:`baseline_factor`, :func:`effective_baseline_m`
* :func:`block_mean`          resample a DEM to a coarser baseline
* :func:`horn_slope_deg`      Horn (1981) slope in degrees on a grid of known cell size
* :func:`plane_rms`           RMS height after plane removal over a window (NaN-aware)
* :func:`tile_features`       slope_med, slope_p90, relief, rough_a, rough_b for every tile of a DEM window

Conventions: a DEM is a 2-D float array, row 0 at the north, cell size in **metres** (``cell_m``, square cells; give the
metric size for geographic or polar grids). The array's cell [0, 0] must sit on a tile corner; tiles are ``tile_m`` squares
(``stride_m`` apart, default ``tile_m`` for the global grid, 3 km for target windows). Callers read DEM windows with a
one-cell halo of 500 m cells if border tiles must not lose their outermost slope cells.

Missing data: NaN. A tile whose share of missing cells exceeds ``max_nan_fraction`` is NaN for that feature, never a
partial answer. Slope at a cell is NaN if any cell of its 3 x 3 neighbourhood is NaN, and always NaN on the outer border of
the array.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Tiles per call are computed in numpy views; for a window of 15 km at 500 m this is tiny. A global DEM must be
# processed in blocks by the caller (the Earth side uses Earth Engine instead).


def baseline_factor(cell_m: float, baseline_m: float) -> int:
    """Native cells per baseline cell: ``max(1, round(baseline_m / cell_m))``."""
    if cell_m <= 0 or baseline_m <= 0:
        raise ValueError("cell_m and baseline_m must be positive")
    return max(1, int(round(baseline_m / cell_m)))


def effective_baseline_m(cell_m: float, baseline_m: float) -> float:
    """The baseline actually obtained: ``factor * cell_m``. Larger than ``baseline_m`` if the native cell is coarser."""
    return baseline_factor(cell_m, baseline_m) * cell_m


def block_mean(dem: np.ndarray, factor: int) -> np.ndarray:
    """Mean of each ``factor x factor`` block of native cells (finite cells only; NaN if a block has none).

    A ragged edge (rows or columns that do not fill a block) is dropped.
    """
    dem = np.asarray(dem, dtype="float64")
    if factor < 1:
        raise ValueError("factor must be at least 1")
    if factor == 1:
        return dem.copy()
    ny, nx = (dem.shape[0] // factor) * factor, (dem.shape[1] // factor) * factor
    if ny == 0 or nx == 0:
        raise ValueError("DEM is smaller than one block")
    blocks = dem[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    with np.errstate(all="ignore"):
        counts = np.isfinite(blocks).sum(axis=(1, 3))
        sums = np.nansum(blocks, axis=(1, 3))
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def horn_slope_deg(dem: np.ndarray, cell_m: float) -> np.ndarray:
    """Horn slope (degrees). ``dz/dx = ((c+2f+i)-(a+2d+g))/(8 dx)``, ``dz/dy = ((g+2h+i)-(a+2b+c))/(8 dy)``.

    NaN on the outer border and wherever any cell of the 3 x 3 neighbourhood is NaN.
    """
    z = np.asarray(dem, dtype="float64")
    out = np.full(z.shape, np.nan)
    if z.shape[0] < 3 or z.shape[1] < 3:
        return out
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cell_m)
    out[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return out


def plane_rms(window: np.ndarray) -> float:
    """RMS of the residual after a least-squares plane ``z = a + b x + c y`` is removed, over the finite cells of ``window``.

    Coordinates are cell indices (the answer does not depend on their unit offset). NaN if fewer than 3 finite cells or if
    all finite cells are collinear.
    """
    w = np.asarray(window, dtype="float64")
    y, x = np.mgrid[0 : w.shape[0], 0 : w.shape[1]]
    ok = np.isfinite(w)
    if ok.sum() < 3:
        return float("nan")
    design = np.column_stack([np.ones(ok.sum()), x[ok], y[ok]])
    if np.linalg.matrix_rank(design) < 3:
        return float("nan")
    coef, *_ = np.linalg.lstsq(design, w[ok], rcond=None)
    return float(np.sqrt(np.mean((w[ok] - design @ coef) ** 2)))


def _windows(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """All ``size x size`` windows at ``stride``: shape (ny_tiles, nx_tiles, size, size). A view, no copy."""
    if arr.shape[0] < size or arr.shape[1] < size:
        return np.empty((0, 0, size, size))
    return sliding_window_view(arr, (size, size))[::stride, ::stride]


def _nan_ok(win: np.ndarray, max_nan_fraction: float) -> np.ndarray:
    return np.isnan(win).mean(axis=(-1, -2)) <= max_nan_fraction


def _closed_form_rms(win: np.ndarray) -> np.ndarray:
    """Plane-removed RMS for windows without NaN, vectorised, closed form (the Earth Engine formula; see SPEC)."""
    n = win.shape[-1]
    idx = np.arange(n) - (n - 1) / 2.0
    ex2 = (n * n - 1) / 12.0
    mean = win.mean(axis=(-1, -2))
    mean_sq = (win**2).mean(axis=(-1, -2))
    exz = (win * idx[None, :]).mean(axis=(-1, -2))  # x runs along columns
    eyz = (win * idx[:, None]).mean(axis=(-1, -2))
    resid = mean_sq - mean**2 - exz**2 / ex2 - eyz**2 / ex2
    return np.sqrt(np.maximum(resid, 0.0))


def _roughness(demB: np.ndarray, nB: int, strideB: int, max_nan_fraction: float) -> np.ndarray:
    win = _windows(demB, nB, strideB)
    if win.size == 0:
        return np.empty(win.shape[:2])
    out = np.full(win.shape[:2], np.nan)
    has_nan = np.isnan(win).any(axis=(-1, -2))
    full = ~has_nan
    if full.any():
        out[full] = _closed_form_rms(win[full])
    partial = has_nan & _nan_ok(win, max_nan_fraction)
    for iy, ix in zip(*np.nonzero(partial)):
        out[iy, ix] = plane_rms(win[iy, ix])
    return out


def tile_features(
    dem: np.ndarray,
    cell_m: float,
    *,
    tile_m: float = 6000.0,
    baseline_m: float = 500.0,
    rough_baselines_m: tuple[float, float] = (100.0, 1000.0),
    stride_m: Optional[float] = None,
    max_nan_fraction: float = 0.2,
) -> dict:
    """Terrain features for every tile of a DEM window.

    Returns ``{"slope_med", "slope_p90", "relief", "rough_a", "rough_b"}`` as float arrays of shape ``(ny_tiles, nx_tiles)``
    plus ``"effective_baseline_m"``: the baselines actually used (``factor * cell_m``). Defaults come from
    ``config/features.yaml`` (the caller passes the config values). ``stride_m`` defaults to ``tile_m``.
    """
    stride_m = tile_m if stride_m is None else stride_m
    dem = np.asarray(dem, dtype="float64")

    # --- slope and relief on the shared baseline ---
    f = baseline_factor(cell_m, baseline_m)
    eff = f * cell_m
    dem_b = block_mean(dem, f)
    n, step = int(round(tile_m / eff)), int(round(stride_m / eff))
    if step < 1:
        raise ValueError("stride is smaller than one baseline cell")
    slope = horn_slope_deg(dem_b, eff)
    sw = _windows(slope, n, step)
    dw = _windows(dem_b, n, step)
    shape = sw.shape[:2]
    out = {k: np.full(shape, np.nan) for k in ("slope_med", "slope_p90", "relief", "rough_a", "rough_b")}
    if sw.size:
        flat = sw.reshape(-1, n * n)
        ok = (np.isnan(flat).mean(axis=1) <= max_nan_fraction) & (np.isfinite(flat).any(axis=1))
        med, p90 = np.full(flat.shape[0], np.nan), np.full(flat.shape[0], np.nan)
        if ok.any():
            q = np.nanpercentile(flat[ok], [50, 90], axis=1)
            med[ok], p90[ok] = q[0], q[1]
        out["slope_med"], out["slope_p90"] = med.reshape(shape), p90.reshape(shape)
        dflat = dw.reshape(-1, n * n)
        dok = (np.isnan(dflat).mean(axis=1) <= max_nan_fraction) & np.isfinite(dflat).any(axis=1)
        relief = np.full(dflat.shape[0], np.nan)
        if dok.any():
            relief[dok] = np.nanmax(dflat[dok], axis=1) - np.nanmin(dflat[dok], axis=1)
        out["relief"] = relief.reshape(shape)

    # --- roughness at its own baselines ---
    effective = {"slope_relief_m": eff}
    for key, base in zip(("rough_a", "rough_b"), rough_baselines_m):
        fb = baseline_factor(cell_m, base)
        effb = fb * cell_m
        nb, stepb = int(round(tile_m / effb)), stride_m / effb
        if abs(stepb - round(stepb)) > 1e-9 or round(stepb) < 1:
            raise ValueError(f"stride {stride_m} m is not a whole number of {effb:g} m cells (baseline {base:g} m)")
        r = _roughness(block_mean(dem, fb), nb, int(round(stepb)), max_nan_fraction)
        h, w = min(r.shape[0], shape[0]), min(r.shape[1], shape[1])
        if r.shape != shape:
            # the baselines crop the ragged edge differently; keep the tiles every baseline can see
            out[key] = np.full(shape, np.nan)
            out[key][:h, :w] = r[:h, :w]
        else:
            out[key] = r
        effective[key] = effb
    out["effective_baseline_m"] = effective
    return out
