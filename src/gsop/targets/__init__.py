"""Target profiles: ``gsop s2 targets --body moon|mars`` (issue B32).

Assembles the rows produced by the per-body modules into ``data/processed/targets_{body}.parquet`` and a readable
``targets_{body}.md``. One row per (target, column):

``target, column, value, spread, source, evidence_grade (A/B/C), window, unavailable, unscored, note``

* A scored target has a row for every column its body's weights use. A value that could not be built has ``unavailable = True``,
  ``value = NaN`` and the reason in ``note``; it is never filled in.
* A target without coordinates (Arcadia, decision D10) gets one row with ``column = "*"`` and ``unscored = True``.
* Malapert has no data and is not a target (it is not in ``config/targets.yaml``).
* Rows for columns that are not scored (for example ``thermal_inertia_themis``, the input of the Mars target check) are kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gsop.atomic import atomic_write_bytes, atomic_write_text
from gsop.config import Config

SCHEMA = pa.schema([
    ("target", pa.string()), ("column", pa.string()), ("value", pa.float64()), ("spread", pa.float64()), ("source", pa.string()),
    ("evidence_grade", pa.string()), ("window", pa.string()), ("unavailable", pa.bool_()), ("unscored", pa.bool_()), ("note", pa.string()),
])
GRADES = ("A", "B", "C")


class TargetError(ValueError):
    pass


def collect_rows(cfg: Config, body: str, cache_dir: Optional[Path] = None, diviner_dir: Optional[Path] = None) -> list[dict]:
    """Run the per-body modules. Reads data over HTTP (windowed) unless a cache already holds the window."""
    if body == "moon":
        from gsop.targets import diviner, moon

        return (moon.moon_terrain_rows(cfg, cache_dir=cache_dir) + moon.moon_illumination_rows(cfg)
                + moon.psr_share_rows(cfg, cache_dir=cache_dir) + diviner.moon_temperature_rows(cfg, diviner_dir))
    if body == "mars":
        from gsop.targets import mars

        return (mars.mars_terrain_rows(cfg, cache_dir=cache_dir) + mars.mars_thermal_rows(cfg, cache_dir=cache_dir)
                + mars.swim_water_index_rows(cfg, cache_dir=cache_dir) + mars.mars_climate_rows(cfg, cache_dir=cache_dir))
    raise TargetError(f"unknown body '{body}'")


def to_frame(rows: list[dict], cfg: Config, body: str) -> pd.DataFrame:
    """Rows -> DataFrame with the fixed columns; adds one ``unscored`` row per target without coordinates."""
    out = []
    for r in rows:
        out.append({**{"unscored": False}, **r})
    for name, t in cfg.targets(body).items():
        if t.lat is None or t.lon_e is None:
            out.append({
                "target": name, "column": "*", "value": np.nan, "spread": np.nan, "source": t.source, "evidence_grade": "C",
                "window": "", "unavailable": True, "unscored": True, "note": f"unscored: no target data ({t.status})",
            })
    df = pd.DataFrame(out, columns=[f.name for f in SCHEMA])
    df["unavailable"] = df["unavailable"].astype(bool)
    df["unscored"] = df["unscored"].astype(bool)
    return df


def used_columns(cfg: Config, body: str) -> list[str]:
    seen: dict[str, None] = {}
    for fam in cfg.weights(body).values():
        for c in fam.columns:
            seen.setdefault(c)
    return list(seen)


def validate(df: pd.DataFrame, cfg: Config, body: str) -> list[str]:
    """Problems that make the file wrong (not merely incomplete). Unavailable columns are allowed, silent gaps are not."""
    problems: list[str] = []
    if not set(df["evidence_grade"]).issubset(GRADES):
        problems.append(f"evidence_grade must be one of {GRADES}")
    scored = [n for n, t in cfg.targets(body).items() if t.lat is not None and t.lon_e is not None]
    cols = used_columns(cfg, body)
    for target in scored:
        have = set(df.loc[df.target == target, "column"])
        for c in cols:
            if c not in have:
                problems.append(f"{target}: no row for column '{c}' (it must be a value or an explicit unavailable row)")
        for _, r in df[(df.target == target)].iterrows():
            if not r["unavailable"] and not np.isfinite(r["value"]):
                problems.append(f"{target}.{r['column']}: marked available but the value is not finite")
            if r["unavailable"] and not str(r["note"]).startswith("unavailable"):
                problems.append(f"{target}.{r['column']}: unavailable rows must say why in note")
            if not r["unavailable"] and not (np.isfinite(r["spread"]) and r["spread"] >= 0):
                problems.append(f"{target}.{r['column']}: spread must be finite and not negative")
            if not str(r["source"]).strip():
                problems.append(f"{target}.{r['column']}: missing source")
    for target in df.loc[df.unscored, "target"].unique():
        if target in scored:
            problems.append(f"{target}: marked unscored but has coordinates")
    unknown = set(df["target"]) - set(cfg.targets(body))
    if unknown:
        problems.append(f"rows for targets that are not in targets.yaml: {sorted(unknown)}")
    return problems


def family_coverage(df: pd.DataFrame, cfg: Config, body: str) -> dict[str, dict[str, dict]]:
    """For each scored target and family: the columns that have a value and the ones that do not."""
    out: dict[str, dict[str, dict]] = {}
    for target, t in cfg.targets(body).items():
        if t.lat is None:
            continue
        sub = df[df.target == target].set_index("column")
        fams = {}
        for fam, w in cfg.weights(body).items():
            ok = [c for c in w.columns if c in sub.index and not bool(sub.loc[c, "unavailable"])]
            fams[fam] = {"available": ok, "missing": [c for c in w.columns if c not in ok], "weight": w.published}
        out[target] = fams
    return out


def render_markdown(df: pd.DataFrame, cfg: Config, body: str) -> str:
    lines = [f"# Target profiles: {body}", "", "Generated by `gsop s2 targets`. One row per target and column; `unavailable` rows were not built and say why.", ""]
    for target in dict.fromkeys(df["target"]):
        sub = df[df.target == target]
        lines += [f"## {target}", ""]
        if bool(sub.unscored.any()):
            lines += [f"**Unscored.** {sub.iloc[0].note}", ""]
            continue
        lines += ["| column | value | spread | grade | status | source |", "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            val = "n/a" if not np.isfinite(r["value"]) else f"{r['value']:.5g}"
            spr = "n/a" if not np.isfinite(r["spread"]) else f"{r['spread']:.4g}"
            status = "UNAVAILABLE" if r["unavailable"] else "ok"
            lines.append(f"| {r['column']} | {val} | {spr} | {r['evidence_grade']} | {status} | {r['source']} |")
        lines += ["", "Unavailable, with reasons:", ""]
        for _, r in sub[sub.unavailable].iterrows():
            lines.append(f"- `{r['column']}`: {r['note']}")
        lines.append("")
    cov = family_coverage(df, cfg, body)
    lines += ["## Families with no usable target column", "", "A family whose columns are all unavailable cannot be scored. Under the missing-data gate (`config/gates.yaml`) a target is rejected for any weights that use a missing feature, so these must be decided before a run (see `docs/SPEC.md` and `docs/UNVERIFIED.md`).", ""]
    any_missing = False
    for target, fams in cov.items():
        gone = [f"{f} (weight {v['weight']})" for f, v in fams.items() if not v["available"]]
        part = [f"{f}: missing {', '.join(v['missing'])}" for f, v in fams.items() if v["available"] and v["missing"]]
        if gone or part:
            any_missing = True
            lines.append(f"- **{target}**: no usable column: {', '.join(gone) or 'none'}" + (f"; partly missing: {'; '.join(part)}" if part else ""))
    if not any_missing:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_targets(df: pd.DataFrame, cfg: Config, body: str, out_dir: Path) -> tuple[Path, Path]:
    problems = validate(df, cfg, body)
    if problems:
        raise TargetError("targets file is invalid:\n  " + "\n  ".join(problems))
    out_dir = Path(out_dir)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    import io

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    parquet = atomic_write_bytes(out_dir / f"targets_{body}.parquet", buf.getvalue())
    md = atomic_write_text(out_dir / f"targets_{body}.md", render_markdown(df, cfg, body))
    return parquet, md


def build_targets(cfg: Config, body: str, out_dir: Path, cache_dir: Optional[Path] = None, diviner_dir: Optional[Path] = None,
                  collect: Optional[Callable[..., list[dict]]] = None) -> pd.DataFrame:
    collect = collect or collect_rows  # resolved at call time so it can be replaced in tests
    df = to_frame(collect(cfg, body, cache_dir, diviner_dir), cfg, body)
    write_targets(df, cfg, body, out_dir)
    return df
