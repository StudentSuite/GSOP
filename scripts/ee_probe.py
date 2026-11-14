"""Print the bands and projection of any Earth Engine asset (image or collection).

    uv run python scripts/ee_probe.py NASA/NASADEM_HGT/001

Needs a working Earth Engine login (see README, "Earth Engine login"). It only asks Earth Engine for
metadata and one tiny ``getInfo()`` call, so it is cheap. Reused by every "verify the dataset" issue
and by the small-region export test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _bands(info: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for band in info.get("bands", []):
        out.append(
            {
                "id": band.get("id"),
                "data_type": band.get("data_type"),
                "crs": band.get("crs"),
                "crs_transform": band.get("crs_transform"),
                "dimensions": band.get("dimensions"),
            }
        )
    return out


def probe(ee: Any, asset_id: str) -> dict[str, Any]:
    """Return a small dict describing ``asset_id``. ``ee`` must already be initialised."""
    asset = ee.data.getAsset(asset_id)
    kind = asset.get("type", "UNKNOWN")
    result: dict[str, Any] = {
        "id": asset_id,
        "type": kind,
        "title": asset.get("properties", {}).get("title"),
        "start_time": asset.get("startTime"),
        "end_time": asset.get("endTime"),
    }
    if kind == "IMAGE_COLLECTION":
        first = ee.ImageCollection(asset_id).first()
        result["size"] = ee.ImageCollection(asset_id).size().getInfo()
    else:
        first = ee.Image(asset_id)
    info = first.getInfo()
    result["bands"] = _bands(info)
    projection = first.select(0).projection().getInfo()
    result["projection"] = {"crs": projection.get("crs"), "transform": projection.get("transform")}
    result["image_properties"] = sorted(info.get("properties", {}))[:40]
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset_id", help="for example NASA/NASADEM_HGT/001")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)

    from gsop.earth.gee_auth import ensure_ee

    ee = ensure_ee()
    result = probe(ee, args.asset_id)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"asset:      {result['id']} ({result['type']})")
    print(f"title:      {result['title']}")
    print(f"time range: {result['start_time']} .. {result['end_time']}")
    if "size" in result:
        print(f"images:     {result['size']}")
    print(f"projection: {result['projection']['crs']}  transform={result['projection']['transform']}")
    print("bands:")
    for band in result["bands"]:
        print(f"  {band['id']:<22} {str(band['data_type']):<40} {band['dimensions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
