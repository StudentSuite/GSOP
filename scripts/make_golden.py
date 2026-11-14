"""Regenerate the golden slice inputs (issue C29). Only needed when the synthetic generator or the input file format changes.

    uv run python scripts/make_golden.py            # write tests/golden/<body>/*.parquet
    uv run gsop golden --update                     # then rewrite tests/golden/golden.json

The inputs are synthetic (planted analog in Gaussian background, see tests/synth.py), 5,000 tiles per body, fixed seed. Commit both together:
changing an input file without rewriting golden.json makes ``gsop golden`` report an input hash mismatch.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

SEEDS = {"moon": 20261114, "mars": 20261115}  # fixed, only for the synthetic generator


def main() -> None:
    import stage_inputs as SI

    from gsop.config import load_config
    from gsop.golden import GOLDEN_DIR, input_files

    cfg = load_config(ROOT / "config")
    for body, seed in SEEDS.items():
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = SI.write_inputs(Path(tmp), body, n=cfg.run.golden_tiles, seed=seed, cfg=cfg)
            for dst in input_files(GOLDEN_DIR, body):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "data" / "processed" / dst.name, dst)
                print(f"wrote {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
