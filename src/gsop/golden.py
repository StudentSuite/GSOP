"""The golden slice (issue C29): a fixed 5,000-tile input per body, run end to end, compared with a committed expectation.

It exists so two machines (the Mac and the Windows laptop) can show they run the same maths without comparing multi-hour runs.

* Inputs are committed under ``tests/golden/<body>/`` and hashed **exactly** (sha256 of the file bytes); a different input hash is reported as an input problem, not a maths difference.
* Outputs are **not** required to be bit-identical (a different BLAS or CPU can move the last bits of a float32): they are compared with the tolerances in ``config/run.yaml`` (``golden``).
* The inputs are **synthetic** (a planted analog in Gaussian background, ``scripts/make_golden.py``): the golden test shows the code agrees with itself across machines and over time, **not** that the real data are right.

``golden.json`` holds, per body and target: the number of gated tiles, the scores of every gated tile (``score_<profile>``), the top-1% set of each profile,
the Monte Carlo ``mc_mean`` / frequencies of the tiles that reached the top 10%, and the top-1% set of the Monte Carlo ranking.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.hashing import sha256_file
from gsop.paths import REPO_ROOT, Paths

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
EXPECTED = GOLDEN_DIR / "golden.json"
BODIES = ("moon", "mars")
SCHEMA = 1
DECIMALS = 6
CONFIG_PARTS = ["weights", "purposes", "features", "run", "gates", "targets"]
INPUT_NAMES = ("earth_features_6km.parquet", "targets_{body}.parquet")


@dataclass(frozen=True)
class Failure:
    check: str  # which check failed
    where: str  # body/target/profile
    detail: str  # by how much

    def __str__(self) -> str:
        return f"{self.check} [{self.where}]: {self.detail}"


def input_files(golden_dir: Path, body: str) -> list[Path]:
    return [golden_dir / body / n.format(body=body) for n in INPUT_NAMES]


def input_hashes(golden_dir: Path, bodies: Iterable[str] = BODIES) -> dict[str, str]:
    out = {}
    for body in bodies:
        for p in input_files(golden_dir, body):
            if not p.is_file():
                raise FileNotFoundError(f"golden input missing: {p} (regenerate with scripts/make_golden.py)")
            out[f"{body}/{p.name}"] = sha256_file(p)
    return out


def versions() -> dict[str, str]:
    """Informational only (never compared): what the numbers were produced with."""
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for pkg in ("numpy", "pandas", "pyarrow", "scipy", "typer"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _clean(a: np.ndarray) -> list:
    """Rounded floats for JSON; NaN becomes null."""
    return [None if not math.isfinite(v) else round(float(v), DECIMALS) for v in a]


def top_fraction_ids(tile_ids: np.ndarray, score: np.ndarray, fraction: float) -> list[int]:
    """The ``ceil(fraction * n)`` tiles with the highest score (ties: lower tile_id first), NaN scores excluded; sorted ascending."""
    ok = np.isfinite(score)
    ids, s = tile_ids[ok], score[ok]
    if len(ids) == 0:
        return []
    k = max(1, math.ceil(fraction * len(ids)))
    order = np.lexsort((ids, -s.astype("float64")))
    return sorted(int(i) for i in ids[order[:k]])


def _summarise_body(cfg: Config, body: str, out: Path) -> dict:
    scores = pd.read_parquet(out / "scores_base.parquet")
    profile_cols = [c for c in scores.columns if c.startswith("score_")]
    top_fraction = min(cfg.run.top_fractions)
    res: dict = {"gated": {}, "scores": {}, "top1": {}, "mc": {}, "mc_top1": {}}
    for target, sub in scores.groupby("target", sort=True):
        sub = sub.sort_values("tile_id")
        ids = sub["tile_id"].to_numpy("int64")
        res["gated"][target] = int(len(sub))
        res["scores"][target] = {"tile_id": [int(i) for i in ids], **{c: _clean(sub[c].to_numpy("float64")) for c in profile_cols}}
        res["top1"][target] = {c.removeprefix("score_"): top_fraction_ids(ids, sub[c].to_numpy("float64"), top_fraction) for c in profile_cols}
        res["mc"][target], res["mc_top1"][target] = {}, {}
        for path in sorted(out.glob(f"mc_{target}_*.parquet")):
            profile = path.stem[len(f"mc_{target}_"):]
            t = pd.read_parquet(path).sort_values("tile_id")
            tid = t["tile_id"].to_numpy("int64")
            res["mc"][target][profile] = {"tile_id": [int(i) for i in tid], "mc_mean": _clean(t["mc_mean"].to_numpy("float64")),
                                          "freq_top1": _clean(t["freq_top1"].to_numpy("float64")), "freq_top5": _clean(t["freq_top5"].to_numpy("float64")),
                                          "freq_top10": _clean(t["freq_top10"].to_numpy("float64")), "robust": [int(i) for i in tid[t["robust"].to_numpy(bool)]]}
            res["mc_top1"][target][profile] = top_fraction_ids(tid, t["mc_mean"].to_numpy("float64"), top_fraction)
    return res


def run_golden(cfg: Config, golden_dir: Path = GOLDEN_DIR, bodies: Iterable[str] = BODIES, runs: Optional[int] = None, workers: int = 1,
               work_dir: Optional[Path] = None, log: Callable[[str], None] = print) -> dict:
    """Run s3, s4 and s5 on the committed slice of every body in a scratch folder and return the ``golden.json`` content."""
    from gsop.stages import s3, s4, s5

    runs = cfg.run.golden_runs if runs is None else runs
    bodies = tuple(bodies)
    hashes = input_hashes(golden_dir, bodies)
    own = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="gsop_golden_")) if own else Path(work_dir)
    try:
        paths = Paths(root)
        result = {"schema": SCHEMA, "runs": runs, "tiles": cfg.run.golden_tiles, "master_seed": cfg.run.seed, "config_hash": cfg.config_hash(CONFIG_PARTS),
                  "inputs": hashes, "versions": versions(), "bodies": {}}
        for body in bodies:
            (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
            for src in input_files(golden_dir, body):
                shutil.copyfile(src, root / "data" / "processed" / src.name)
            out = paths.results(body)
            log(f"[golden] {body}: s3 gates")
            s3.run_gates(cfg, body, paths.earth_parquet, paths.targets_parquet(body), out, log=lambda m: None)
            log(f"[golden] {body}: s4 scores")
            s4.run_scores(cfg, body, paths.earth_parquet, paths.targets_parquet(body), out / "gates_passed.parquet", out, log=lambda m: None)
            log(f"[golden] {body}: s5 simulate, {runs} runs")
            s5.run_simulate(cfg, body, paths, runs=runs, workers=workers, log=lambda m: None)
            result["bodies"][body] = _summarise_body(cfg, body, out)
        return result
    finally:
        if own:
            shutil.rmtree(root, ignore_errors=True)


def write_golden(result: dict, path: Path) -> Path:
    return atomic_write_text(path, json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def read_golden(path: Path = EXPECTED) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def _max_diff(ids_a: list[int], va: list, ids_b: list[int], vb: list) -> tuple[float, int]:
    """Largest absolute difference over the tile ids both lists share, and its tile id (NaN against a number counts as infinite)."""
    da = dict(zip(ids_a, va))
    worst, at = 0.0, -1
    for i, y in zip(ids_b, vb):
        if i not in da:
            continue
        x = da[i]
        d = 0.0 if (x is None and y is None) else (math.inf if (x is None or y is None) else abs(x - y))
        if d > worst:
            worst, at = d, i
    return worst, at


def compare(expected: dict, actual: dict, score_tolerance: float, top1_jaccard_min: float) -> list[Failure]:
    """Every difference that matters, as readable failures; an empty list means the two golden files agree within tolerance.

    Exact: the input hashes, the run count, the master seed and the set of tiles that passed the gates. Tolerance ``score_tolerance`` (absolute, on the 0-100
    scale): every score and every ``mc_mean``. ``top1_jaccard_min``: the overlap of every top-1% set (base score and Monte Carlo).
    A different ``config_hash`` is reported first: the expected file is then stale, and the other differences are probably its consequence.
    """
    fails: list[Failure] = []
    for key in ("schema", "runs", "master_seed", "tiles"):
        if expected.get(key) != actual.get(key):
            fails.append(Failure(key, "run", f"expected {expected.get(key)!r}, got {actual.get(key)!r}"))
    if expected.get("config_hash") != actual.get("config_hash"):
        fails.append(Failure("config_hash", "run", "the scoring config changed since golden.json was written; if on purpose run `gsop golden --update`"))
    for name in sorted(set(expected["inputs"]) | set(actual["inputs"])):
        if expected["inputs"].get(name) != actual["inputs"].get(name):
            fails.append(Failure("input_hash", name, "input file differs byte for byte from the one the expectation was made with"))
    for body in sorted(set(expected["bodies"]) & set(actual["bodies"])):
        e, a = expected["bodies"][body], actual["bodies"][body]
        for target in sorted(set(e["gated"]) | set(a["gated"])):
            where = f"{body}/{target}"
            if target not in e["gated"] or target not in a["gated"]:
                fails.append(Failure("targets", where, "target present in only one of the two files"))
                continue
            if e["scores"][target]["tile_id"] != a["scores"][target]["tile_id"]:
                fails.append(Failure("gated_tiles", where, f"{e['gated'][target]} expected, {a['gated'][target]} got, or different tile ids"))
                continue
            ids = e["scores"][target]["tile_id"]
            for col in sorted(c for c in e["scores"][target] if c.startswith("score_")):
                d, at = _max_diff(ids, e["scores"][target][col], ids, a["scores"][target].get(col, [None] * len(ids)))
                if d > score_tolerance:
                    fails.append(Failure("score", f"{where}/{col}", f"max abs difference {d:.3g} at tile {at} (tolerance {score_tolerance:g})"))
            for profile, expect_ids in e["top1"][target].items():
                j = _jaccard(expect_ids, a["top1"][target].get(profile, []))
                if j < top1_jaccard_min:
                    fails.append(Failure("top1_jaccard", f"{where}/{profile}", f"Jaccard {j:.4f} (minimum {top1_jaccard_min:g})"))
            for profile, em in e["mc"][target].items():
                am = a["mc"][target].get(profile)
                if am is None:
                    fails.append(Failure("mc", f"{where}/{profile}", "Monte Carlo output missing"))
                    continue
                d, at = _max_diff(em["tile_id"], em["mc_mean"], am["tile_id"], am["mc_mean"])
                if d > score_tolerance:
                    fails.append(Failure("mc_mean", f"{where}/{profile}", f"max abs difference {d:.3g} at tile {at} (tolerance {score_tolerance:g})"))
                j = _jaccard(e["mc_top1"][target][profile], a["mc_top1"][target][profile])
                if j < top1_jaccard_min:
                    fails.append(Failure("mc_top1_jaccard", f"{where}/{profile}", f"Jaccard {j:.4f} (minimum {top1_jaccard_min:g})"))
    for body in sorted(set(expected["bodies"]) ^ set(actual["bodies"])):
        fails.append(Failure("bodies", body, "body present in only one of the two files"))
    return fails
