"""Check an exported Earth raster (issue B17): band names, CRS, pixel size, nodata, valid share and ranges.

Used on the Lonar test box before any global export, and on every global chunk later. Pure rasterio and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio

from gsop.earth.sanity import SANITY_RANGES


@dataclass
class BandReport:
    name: str
    valid_share: float
    minimum: float
    median: float
    maximum: float
    out_of_range_share: float
    problems: list[str] = field(default_factory=list)


@dataclass
class RasterReport:
    path: str
    crs: Optional[str]
    pixel_size: tuple[float, float]
    shape: tuple[int, int]
    nodata: Optional[float]
    size_bytes: int
    bands: list[BandReport]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems and not any(b.problems for b in self.bands)


def check_raster(path: Path, expected_bands: Sequence[str], grid_crs: str = "EPSG:6933", tile_m: float = 6000.0) -> RasterReport:
    path = Path(path)
    problems: list[str] = []
    with rasterio.open(path) as ds:
        crs = ds.crs.to_string() if ds.crs else None
        px = (abs(ds.transform.a), abs(ds.transform.e))
        names = [d or f"band_{i + 1}" for i, d in enumerate(ds.descriptions)]
        if crs != grid_crs:
            problems.append(f"CRS is {crs}, expected {grid_crs}")
        if not (np.isclose(px[0], tile_m) and np.isclose(px[1], tile_m)):
            problems.append(f"pixel size is {px}, expected {tile_m} m")
        if (ds.transform.c % tile_m) or (ds.transform.f % tile_m):
            problems.append("grid origin is not a multiple of the tile size (chunks would not line up)")
        missing = [b for b in expected_bands if b not in names]
        extra = [b for b in names if b not in expected_bands]
        if missing:
            problems.append(f"missing bands: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected bands: {', '.join(extra)}")
        reports: list[BandReport] = []
        for i, name in enumerate(names, start=1):
            arr = ds.read(i, masked=True).astype("float64").filled(np.nan)
            finite = np.isfinite(arr)
            share = float(finite.mean()) if arr.size else 0.0
            bp: list[str] = []
            if not finite.any():
                reports.append(BandReport(name, 0.0, np.nan, np.nan, np.nan, 0.0, ["no valid pixels at all"]))
                continue
            vals = arr[finite]
            lo, hi, _ = SANITY_RANGES.get(name, (None, None, ""))
            bad = np.zeros(vals.shape, dtype=bool)
            if lo is not None:
                bad |= vals < lo
            if hi is not None:
                bad |= vals > hi
            bad_share = float(bad.mean())
            if bad_share > 0:
                bp.append(f"{bad_share:.1%} of valid pixels outside the plausible range [{lo}, {hi}]")
            reports.append(BandReport(name, share, float(vals.min()), float(np.median(vals)), float(vals.max()), bad_share, bp))
        nodata = ds.nodata
        shape = (ds.height, ds.width)
    return RasterReport(str(path), crs, px, shape, nodata, path.stat().st_size, reports, problems)


def format_report(report: RasterReport) -> str:
    lines = [
        f"file:   {report.path}  ({report.size_bytes / 1e6:.2f} MB)",
        f"grid:   {report.crs}, pixel {report.pixel_size[0]:g} x {report.pixel_size[1]:g} m, {report.shape[1]} x {report.shape[0]} px, nodata {report.nodata}",
        f"{'band':<24}{'valid':>8}{'min':>12}{'median':>12}{'max':>12}  problems",
    ]
    for b in report.bands:
        lines.append(f"{b.name:<24}{b.valid_share:>8.1%}{b.minimum:>12.4g}{b.median:>12.4g}{b.maximum:>12.4g}  {'; '.join(b.problems)}")
    lines += [f"PROBLEM: {p}" for p in report.problems]
    lines.append("RESULT: " + ("ok" if report.ok else "PROBLEMS FOUND"))
    return "\n".join(lines)
