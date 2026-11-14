"""Earth feature QA: statistics, histograms, maps, sanity checks. Writes docs/EARTH_QA.md.

    uv run python scripts/earth_qa.py [--parquet data/processed/earth_features_6km.parquet] [--out docs]
    uv run --with matplotlib python scripts/earth_qa.py        # also writes the histogram and map PNGs

Exit code 1 if any check fails: a failing check blocks the Moon and Mars runs until it is resolved. The verdict is
also written to data/interim/earth_qa.json so later stages can refuse to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def render_markdown(stats, checks, areas, plots_dir_rel, plots_written, rows, parquet_name) -> str:
    lines = [
        "# Earth feature QA",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/earth_qa.py` from `{parquet_name}` ({rows:,} land tiles).",
        "",
        "**A failing check blocks the Moon and Mars runs until it is resolved** (verdict in `data/interim/earth_qa.json`).",
        "",
        f"## Result: {'PASS' if all(c.passed for c in checks) else 'FAIL'}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {c.detail} |")
    lines += ["", "## Feature statistics", "", "| Feature | NaN % | at nodata sentinel % | min | median | max |", "|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(f"| {s.name}{' (UNAVAILABLE)' if s.unavailable else ''} | {s.pct_nan:.1f} | {s.pct_sentinel:.2f} | {s.minimum:.4g} | {s.median:.4g} | {s.maximum:.4g} |")
    lines += ["", "## Known bad areas", ""]
    if not areas:
        lines.append("None found by the per-tile checks.")
    for name, cells in areas.items():
        lines.append(f"- **{name}**: " + ", ".join(f"lat {a}..{a + 10} lon {b}..{b + 10} ({n:,} tiles)" for a, b, n in cells))
    if plots_written:
        lines += ["", "## Plots", ""]
        for s in stats:
            if s.name in plots_written:
                lines.append(f"### {s.name}\n\n![histogram]({plots_dir_rel}/hist_{s.name}.png) ![map]({plots_dir_rel}/map_{s.name}.png)\n")
    else:
        lines += ["", "## Plots", "", "Not written (matplotlib is not installed). Run `uv run --with matplotlib python scripts/earth_qa.py`.", ""]
    lines += ["", "For anything wrong, open an issue in the tracker and link it here."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", default=str(ROOT / "data" / "processed" / "earth_features_6km.parquet"))
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    from gsop.atomic import atomic_write_text
    from gsop.config import load_config
    from gsop.earth import qa

    path = Path(args.parquet)
    if not path.is_file():
        print(f"{path} does not exist: the Earth file has not been built yet (issue B18). Nothing was checked.", file=sys.stderr)
        return 2
    cfg = load_config()
    df = pd.read_parquet(path)
    stats = qa.feature_stats(df)
    checks = qa.run_qa(df, cfg)
    areas = qa.bad_areas(df, checks)

    out = Path(args.out)
    plots_dir = out / "earth_qa"
    written: set[str] = set()
    try:
        import matplotlib.pyplot as plt

        plots_dir.mkdir(parents=True, exist_ok=True)
        sample = df.sample(min(len(df), 200_000), random_state=0)
        for s in stats:
            if s.unavailable:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df[s.name].dropna(), bins=80)
            ax.set(title=s.name, ylabel="tiles")
            fig.tight_layout()
            fig.savefig(plots_dir / f"hist_{s.name}.png", dpi=100)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 3))
            sc = ax.scatter(sample.lon, sample.lat, c=sample[s.name], s=0.5)
            fig.colorbar(sc, ax=ax)
            ax.set(title=s.name, xlabel="lon", ylabel="lat")
            fig.tight_layout()
            fig.savefig(plots_dir / f"map_{s.name}.png", dpi=100)
            plt.close(fig)
            written.add(s.name)
    except ImportError:
        pass
    atomic_write_text(out / "EARTH_QA.md", render_markdown(stats, checks, areas, "earth_qa", written, len(df), path.name))
    passed = qa.overall_pass(checks)
    atomic_write_text(ROOT / "data" / "interim" / "earth_qa.json",
                      json.dumps({"passed": passed, "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "failed": [c.name for c in checks if not c.passed]}, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(c.passed for c in checks)}/{len(checks)} checks passed; report in {out / 'EARTH_QA.md'}")
    for c in checks:
        if not c.passed:
            print(f"  FAIL {c.name}: {c.detail}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
