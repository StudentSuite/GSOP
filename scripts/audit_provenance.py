"""Audit: nothing numeric enters the repo without a source (issue B34).

    uv run python scripts/audit_provenance.py [--strict]

Checks
  1. every feature column in config/features.yaml: each source named for a body that is not purely "computed" has a matching
     dataset row in docs/PROVENANCE.md (or an entry in docs/UNVERIFIED.md), and every column is mentioned in UNVERIFIED or SPEC;
  2. every row of data/processed/targets_{moon,mars}.parquet (when the files exist): its source matches a PROVENANCE dataset row, and an
     unavailable row's column is explained in UNVERIFIED.md or SPEC.md;
  3. every column of data/processed/earth_features_6km.parquet (when it exists) is a documented feature or flag.

Prints a table and the number of missing rows. Exit code 0 when nothing is missing; 1 otherwise (CI runs this step as warning-only).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# keyword found in a source name -> keyword that must appear in docs/PROVENANCE.md (or UNVERIFIED.md)
KEYWORDS = {
    "NASADEM": "NASADEM", "Copernicus": "Copernicus", "MODIS": "MOD11A2", "ASTER": "ASTER", "MERRA": "MERRA", "EMIT": "EMIT",
    "LOLA": "LOLA", "MOLA": "MOLA", "Diviner": "Diviner", "THEMIS": "THEMIS", "TES": "TES", "SWIM": "SWIM", "TerraClimate": "TerraClimate",
    "permafrost": "Permafrost", "PSR": "permanent shadow", "Glaser": None, "Barker": None, "Grant": None,
}
SPECIAL = {"MODIS": {"NDVI": "MOD13A2", "land cover": "MCD12Q1", "land-cover": "MCD12Q1"}}
NOT_A_DATASET = ("computed", "standard-atmosphere", "gsop.", "exponential", "PSR share", "Mars climate data", "none verified", "not built", "no dataset")


def read(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def needed_keywords(source_name: str) -> list[str]:
    out: list[str] = []
    for key, kw in KEYWORDS.items():
        if key.lower() in source_name.lower() and kw:
            sub = SPECIAL.get(key, {})
            hit = [v for k, v in sub.items() if k.lower() in source_name.lower()]
            out.extend(hit or [kw])
    return out


def audit() -> tuple[list[tuple[str, str, str]], list[str]]:
    from gsop.config import load_config

    prov, unv, spec = read("PROVENANCE.md"), read("UNVERIFIED.md"), read("SPEC.md")
    cfg = load_config(ROOT / "config")
    rows: list[tuple[str, str, str]] = []  # (what, status, detail)
    skipped: list[str] = []

    def in_docs(token: str) -> bool:
        return token.lower() in (prov + unv).lower()

    # 1. feature columns
    for name, f in cfg.features.columns.items():
        for body, src in f.sources.items():
            if src is None:
                continue
            if any(m in src["name"] for m in NOT_A_DATASET) and not needed_keywords(src["name"]):
                rows.append((f"{name}.{body}", "ok", f"computed or formula ({src['name'][:50]})"))
                continue
            need = needed_keywords(src["name"])
            if not need:
                rows.append((f"{name}.{body}", "MISSING", f"no recognised dataset in source '{src['name'][:60]}'"))
                continue
            missing = [m for m in need if not in_docs(m)]
            rows.append((f"{name}.{body}", "MISSING" if missing else "ok", f"needs {', '.join(need)}" + (f"; absent from PROVENANCE/UNVERIFIED: {', '.join(missing)}" if missing else "")))
        mentioned = name in unv or name in spec or name in prov
        rows.append((f"{name} (column documented)", "ok" if mentioned else "MISSING", "named in SPEC, PROVENANCE or UNVERIFIED" if mentioned else "not named in any doc"))

    # 2. target tables
    import pandas as pd

    for body in ("moon", "mars"):
        path = ROOT / "data" / "processed" / f"targets_{body}.parquet"
        if not path.is_file():
            skipped.append(f"targets_{body}.parquet does not exist (run `gsop s2 targets --body {body}`)")
            continue
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            label = f"targets_{body}: {r['target']}.{r['column']}"
            if r["unscored"]:
                ok = "unscored" in unv.lower() or "arcadia" in unv.lower()
                rows.append((label, "ok" if ok else "MISSING", "unscored target explained in UNVERIFIED" if ok else "unscored target not explained"))
            elif r["unavailable"]:
                ok = r["column"] in unv or r["column"] in spec
                rows.append((label, "ok" if ok else "MISSING", "unavailable value explained in UNVERIFIED/SPEC" if ok else "unavailable value not explained"))
            else:
                need = needed_keywords(r["source"])
                if not need and any(m in r["source"] for m in NOT_A_DATASET):
                    rows.append((label, "ok", "computed from documented formula"))
                    continue
                missing = [m for m in need if not in_docs(m)]
                rows.append((label, "MISSING" if (missing or not need) else "ok", f"source '{r['source'][:50]}'" + (f" absent: {missing}" if missing else "")))

    # 3. the Earth file
    ep = ROOT / "data" / "processed" / "earth_features_6km.parquet"
    if ep.is_file():
        import pyarrow.parquet as pq

        known = set(cfg.features.columns) | {"tile_id", "lat", "lon", "x_m", "y_m", "land", "elevation_mean", "t_low_obs", "ti_proxy_fallback", "land_frac"}
        known |= set(cfg.features.gate_layers) | {f"{c}_unavailable" for c in cfg.features.columns}
        for col in pq.ParquetFile(ep).schema_arrow.names:
            rows.append((f"earth_features_6km.{col}", "ok" if col in known else "MISSING", "documented column" if col in known else "column not in features.yaml"))
    else:
        skipped.append("earth_features_6km.parquet does not exist (the Earth file is not built yet, issue B18)")
    return rows, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="accepted for clarity; the exit code already reflects missing rows")
    ap.parse_args(argv)
    rows, skipped = audit()
    width = max(len(r[0]) for r in rows)
    for what, status, detail in rows:
        if status != "ok":
            print(f"{status:<8} {what:<{width}}  {detail}")
    missing = [r for r in rows if r[1] != "ok"]
    for note in skipped:
        print(f"SKIPPED  {note}")
    print(f"audited {len(rows)} item(s): {len(rows) - len(missing)} documented, {len(missing)} missing, {len(skipped)} check(s) skipped")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
