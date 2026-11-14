"""``gsop check-operators``: Earth Engine versus numpy terrain operators on one Earth window (issue B24).

Window: 3 x 3 tiles (18 km square, on the 6 km lattice) centred on Lonar (the primary known site in
``config/known_sites.yaml``; its coordinates are approximate and unconfirmed, which does not matter for choosing a test
window). Both paths start from the same elevation data:

* **Earth Engine path**: the export expressions of ``gee_export`` (slope_med, relief, rough_a) evaluated on the 3 x 3 tile grid.
* **numpy path**: ``operators.tile_features`` run on the DEM fetched from Earth Engine on a 25 m EPSG:6933 grid (25 m divides 100 m,
  500 m and 1 km exactly, so the block means line up).

Compared per feature: relative difference of the mean over tiles (with an absolute floor so flat terrain does not blow up the
ratio). The command exits 1 if any feature differs by more than ``operator_check_tolerance`` (10%). It also prints the
intermediate 500 m DEM and slope grids from both paths so a mismatch can be located.

Acceptable differences, written down in advance: (1) the DEM is resampled to the 25 m grid by Earth Engine (bilinear) for
numpy, while the Earth Engine path block-averages the native 30 m cells directly; (2) Earth Engine's percentile reducer is
histogram based, numpy interpolates; (3) the Earth Engine roughness uses a closed form that is exact only on full tiles.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from gsop import operators as O
from gsop.config import Config
from gsop.earth import gee_export as G

GRID_N = 3  # tiles per side
NATIVE_M = 25.0
COMPARED = ("slope_med", "relief", "rough_a")
# Absolute floors so a flat window does not turn a tiny difference into a huge ratio. TUNABLE, NOT SOURCED.
ABS_FLOOR = {"slope_med": 0.1, "relief": 1.0, "rough_a": 0.1}  # degrees, metres, metres


@dataclass
class WindowData:
    """Everything fetched from Earth Engine for the window (arrays are 2-D, row 0 north)."""

    x0: float
    y_top: float
    dem25: np.ndarray  # elevation on a 25 m grid
    ee_dem500: np.ndarray  # Earth Engine's own 500 m resampled DEM (36 x 36)
    ee_slope500: np.ndarray  # Earth Engine's Horn slope at 500 m
    ee_tiles: dict  # band -> (3, 3) array from the export expressions


@dataclass
class FeatureResult:
    name: str
    ee_mean: float
    np_mean: float
    rel_diff: float
    max_tile_rel_diff: float
    passed: bool


@dataclass
class CheckResult:
    features: list
    intermediate: dict
    tolerance: float
    passed: bool
    window: dict


def window_bounds(cfg: Config, lat: float, lon: float, n: int = GRID_N) -> tuple[float, float, float, float]:
    """(x0, y_top, x1, y_bottom) in EPSG:6933 of an n x n tile window containing (lat, lon) in its central tile."""
    from pyproj import Transformer

    tile = cfg.features.operators.tile_m
    x, y = Transformer.from_crs("EPSG:4326", cfg.features.operators.grid_crs, always_xy=True).transform(lon, lat)
    ix, iy = math.floor(x / tile), math.floor(y / tile)
    half = n // 2
    x0, x1 = (ix - half) * tile, (ix - half + n) * tile
    y_bottom, y_top = (iy - half) * tile, (iy - half + n) * tile
    return float(x0), float(y_top), float(x1), float(y_bottom)


def relative_difference(a: float, b: float, floor: float) -> float:
    return abs(a - b) / max(abs(b), floor)


def compare(window: WindowData, cfg: Config) -> CheckResult:
    """Run the numpy operator on ``window.dem25`` and compare with the Earth Engine tile bands."""
    ops = cfg.features.operators
    base_a = G.roughness_baseline_m(cfg, "rough_a")
    base_b = G.roughness_baseline_m(cfg, "rough_b")
    np_tiles = O.tile_features(
        window.dem25, NATIVE_M, tile_m=ops.tile_m, baseline_m=ops.baseline_m,
        rough_baselines_m=(base_a, base_b), max_nan_fraction=ops.max_nan_fraction,
    )
    tol = cfg.run.operator_check_tolerance
    results = []
    for name in COMPARED:
        e = np.asarray(window.ee_tiles[name], dtype="float64")
        n = np.asarray(np_tiles[name], dtype="float64")
        both = np.isfinite(e) & np.isfinite(n)
        if not both.any():
            results.append(FeatureResult(name, float("nan"), float("nan"), float("inf"), float("inf"), False))
            continue
        floor = ABS_FLOOR[name]
        rel = relative_difference(float(e[both].mean()), float(n[both].mean()), floor)
        per_tile = np.abs(e[both] - n[both]) / np.maximum(np.abs(n[both]), floor)
        results.append(FeatureResult(name, float(e[both].mean()), float(n[both].mean()), rel, float(per_tile.max()), rel <= tol))

    dem_b_np = O.block_mean(window.dem25, O.baseline_factor(NATIVE_M, ops.baseline_m))
    slope_np = O.horn_slope_deg(dem_b_np, ops.baseline_m)
    inner = np.isfinite(slope_np) & np.isfinite(window.ee_slope500[: slope_np.shape[0], : slope_np.shape[1]])
    ee_s = window.ee_slope500[: slope_np.shape[0], : slope_np.shape[1]]
    ee_d = window.ee_dem500[: dem_b_np.shape[0], : dem_b_np.shape[1]]
    intermediate = {
        "dem500 mean (m): Earth Engine / numpy": (float(np.nanmean(ee_d)), float(np.nanmean(dem_b_np))),
        "dem500 max abs difference (m)": float(np.nanmax(np.abs(ee_d - dem_b_np))),
        "dem500 min / max (m): Earth Engine": (float(np.nanmin(ee_d)), float(np.nanmax(ee_d))),
        "dem500 min / max (m): numpy": (float(np.nanmin(dem_b_np)), float(np.nanmax(dem_b_np))),
        "slope500 mean (deg): Earth Engine / numpy": (float(np.nanmean(ee_s[inner])), float(np.nanmean(slope_np[inner]))),
        "slope500 max abs difference (deg)": float(np.nanmax(np.abs(ee_s[inner] - slope_np[inner]))) if inner.any() else float("nan"),
    }
    return CheckResult(results, intermediate, tol, all(r.passed for r in results), {"x0": window.x0, "y_top": window.y_top, "tiles": GRID_N})


def render_markdown(result: CheckResult, when: Optional[str] = None) -> str:
    when = when or datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Operator check: Earth Engine versus numpy",
        "",
        f"Generated {when} by `gsop check-operators`. Result: **{'PASS' if result.passed else 'FAIL'}** (tolerance {result.tolerance:.0%} per feature).",
        "",
        f"Window: {result.window['tiles']} x {result.window['tiles']} tiles of 6 km, north-west corner x = {result.window['x0']:.0f} m, y = {result.window['y_top']:.0f} m (EPSG:6933), centred on Lonar.",
        "",
        "| Feature | Earth Engine (mean over tiles) | numpy | relative difference | worst tile | result |",
        "|---|---|---|---|---|---|",
    ]
    for r in result.features:
        lines.append(f"| {r.name} | {r.ee_mean:.4g} | {r.np_mean:.4g} | {r.rel_diff:.2%} | {r.max_tile_rel_diff:.2%} | {'pass' if r.passed else '**FAIL**'} |")
    lines += ["", "## Intermediate steps", ""]
    for k, v in result.intermediate.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Known acceptable differences",
        "",
        "1. numpy works on a 25 m grid resampled bilinearly by Earth Engine; the Earth Engine path block-averages the native 30 m cells directly.",
        "2. Earth Engine's percentile reducer is histogram based; numpy interpolates.",
        "3. The Earth Engine roughness is a closed form that is exact only on full tiles.",
        "",
        "If this fails, open a fix issue **before any global export**.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Earth Engine fetch (not exercised without a login)
# ---------------------------------------------------------------------------------------------------------------------
def _pixels(ee: Any, image: Any, crs: str, x0: float, y_top: float, scale: float, n: int, band: str) -> np.ndarray:
    request = {
        "expression": image.select([band]),
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": n, "height": n},
            "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -scale, "translateY": y_top},
            "crsCode": crs,
        },
    }
    raw = ee.data.computePixels(request)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return np.asarray(arr[band] if arr.dtype.names else arr, dtype="float64")


def fetch_window(ee: Any, cfg: Config) -> WindowData:
    ops = cfg.features.operators
    lonar = cfg.known_sites.primary["lonar"]
    x0, y_top, x1, y_bottom = window_bounds(cfg, lonar.lat, lonar.lon)
    side = int(round((x1 - x0) / ops.tile_m))
    dem = G.dem_mosaic(ee, cfg)
    n25 = int(round((x1 - x0) / NATIVE_M))
    n500 = int(round((x1 - x0) / ops.baseline_m))
    dem25 = _pixels(ee, dem.resample("bilinear"), ops.grid_crs, x0, y_top, NATIVE_M, n25, "elevation")
    dem500_img = G.resample_mean(ee, dem, cfg, ops.baseline_m)
    ee_dem500 = _pixels(ee, dem500_img, ops.grid_crs, x0, y_top, ops.baseline_m, n500, "elevation")
    ee_slope500 = _pixels(ee, G.horn_slope_deg(ee, dem500_img, ops.baseline_m), ops.grid_crs, x0, y_top, ops.baseline_m, n500, "slope")
    tiles = {}
    for name, image in (
        ("slope_med", G.slope_bands(ee, cfg)), ("relief", G.relief_band(ee, cfg)), ("rough_a", G.rough_band(ee, cfg, "rough_a")),
    ):
        tiles[name] = _pixels(ee, image, ops.grid_crs, x0, y_top, ops.tile_m, side, name)
    return WindowData(x0, y_top, dem25, ee_dem500, ee_slope500, tiles)


def run_check(ee: Any, cfg: Config, out_md: Path, log: Callable[[str], None] = print) -> CheckResult:
    result = compare(fetch_window(ee, cfg), cfg)
    from gsop.atomic import atomic_write_text

    atomic_write_text(out_md, render_markdown(result))
    for r in result.features:
        log(f"{r.name:<10} EE {r.ee_mean:.4g}  numpy {r.np_mean:.4g}  difference {r.rel_diff:.2%}  {'ok' if r.passed else 'FAIL'}")
    log(f"{'PASS' if result.passed else 'FAIL'}; wrote {out_md}")
    return result
