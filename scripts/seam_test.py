"""Export two narrow strips across the DEM seams and test them (issue B07).

    uv run python scripts/seam_test.py [--out docs/seam]

Strips: Scandinavia across 60 N and Patagonia across 56 S, 1 degree wide, 6 degrees tall, at the 6 km tile grid.
For slope_med and relief it prints the step across the seam against placebo steps and writes a CSV profile per
strip (and a PNG if matplotlib is available: ``uv run --with matplotlib python scripts/seam_test.py``).
Needs an Earth Engine login. The strip longitudes below are a choice made here, not sourced: change them if the
strip hits sea.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STRIPS = {  # name: (seam latitude, west, east)
    "scandinavia_60N": (60.0, 15.0, 16.0),
    "patagonia_56S": (-56.0, -70.5, -69.5),
}
HALF_HEIGHT_DEG = 3.0


def fetch_strip(ee, cfg, seam_lat, west, east):
    """Return ``lat (rows,)``, and ``{band: array(rows, cols)}`` at 6 km for one strip."""
    from gsop.earth import gee_export as G

    ops = cfg.features.operators
    region = ee.Geometry.Rectangle([west, seam_lat - HALF_HEIGHT_DEG, east, seam_lat + HALF_HEIGHT_DEG], "EPSG:4326", False)
    image = ee.Image.cat([G.slope_bands(ee, cfg).select("slope_med"), G.relief_band(ee, cfg)])
    image = image.reproject(crs=ops.grid_crs, scale=ops.tile_m)
    lonlat = ee.Image.pixelLonLat().reproject(crs=ops.grid_crs, scale=ops.tile_m)
    info = image.addBands(lonlat).sampleRectangle(region=region, defaultValue=float("nan")).getInfo()["properties"]
    lat = np.array(info["latitude"], dtype="float64")[:, 0]
    return lat, {b: np.array(info[b], dtype="float64") for b in ("slope_med", "relief")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/seam", help="folder for the CSV profiles and plots")
    args = ap.parse_args()

    from gsop.config import load_config
    from gsop.earth.gee_auth import ensure_ee
    from gsop.earth.seam import seam_test, row_means

    cfg, ee = load_config(), ensure_ee()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (seam_lat, west, east) in STRIPS.items():
        lat, bands = fetch_strip(ee, cfg, seam_lat, west, east)
        for band, values in bands.items():
            res = seam_test(lat, values, seam_lat)
            print(f"{name:<16} {band:<10} step={res.step:+.3f} placebo_sd={res.placebo_sd:.3f} z={res.z:.2f} -> {res.verdict}")
            means = row_means(values)
            np.savetxt(out / f"{name}_{band}.csv", np.column_stack([lat, means]), delimiter=",", header="lat,mean", comments="")
            try:
                import matplotlib.pyplot as plt  # optional
            except ImportError:
                continue
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(lat, means)
            ax.axvline(seam_lat, color="r", ls="--")
            ax.set(title=f"{name}: {band}", xlabel="latitude", ylabel=band)
            fig.tight_layout()
            fig.savefig(out / f"{name}_{band}.png", dpi=120)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
