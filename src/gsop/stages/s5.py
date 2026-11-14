"""s5 simulate: the Monte Carlo for every target x profile (issue C22).

For each scored target and each purpose profile it runs ``gsop.montecarlo.simulate`` and writes ``results/<body>/mc_<target>_<profile>.parquet``. At the end
``convergence.json`` (all target x profile reports) and ``mc_summary.json`` (timings for the time budget) are written. Sigma is read from s4's
``sigma_table.json`` so the Monte Carlo and the base score use exactly the same values.

Resume: every finished (target, profile) writes a small marker in ``results/<body>/.s5/`` (with a fingerprint of its inputs); a rerun skips it. Inside a
profile an interrupted run resumes at the block level (``results/<body>/.mc_work/``). The landing_mobility profile scores only the tiles whose extra gate passed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from gsop import montecarlo as MC
from gsop import similarity as S
from gsop import target_check as TC
from gsop.atomic import atomic_write_bytes, atomic_write_text
from gsop.config import Config
from gsop.paths import Paths
from gsop.purposes import apply_purpose
from gsop.runner import Stage
from gsop.stages import s3, s4
from gsop.targets import profiles_from_frame, read_targets
from gsop.weights import derive_weights, weight_vector


def memory_chunk_rows(cfg: Config, n_columns: int) -> int:
    """Rows per scoring chunk: the default, or less if the free RAM budget of ``gsop doctor`` (cell chunk at most ``chunk_ram_fraction`` of free RAM) is smaller.
    Scores do not depend on this (tested), so two machines may choose differently."""
    import psutil

    from gsop.doctor import chunk_cells

    budget = chunk_cells(int(psutil.virtual_memory().available), max(n_columns, 1) * 4, cfg.run.chunk_ram_fraction)
    return int(max(4096, min(S.DEFAULT_CHUNK_ROWS, budget)))


def memory_numbers() -> tuple[int, int]:
    """``(free RAM now, the budget `gsop doctor` prints)`` in bytes: the budget is 40% of free RAM (``chunk_ram_fraction``)."""
    from gsop.doctor import collect

    info = collect()
    return int(info["ram_free_gb"] * 1e9), int(info["chunk"]["megabytes"] * 1e6)


def _sigma_from_file(path: Path, target: str, setup: S.ScoreSetup) -> np.ndarray:
    table = json.loads(Path(path).read_text(encoding="utf-8"))["targets"].get(target)
    if table is None:
        raise ValueError(f"{path} has no sigma table for target {target}: run s4 first")
    return np.array([table[c]["sigma"] for c in setup.columns], dtype="float64")


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    import io

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    atomic_write_bytes(path, buf.getvalue())


def run_simulate(cfg: Config, body: str, paths: Paths, runs: Optional[int] = None, workers: Optional[int] = None, target_check: Optional[str] = None,
                 block_runs: int = MC.DEFAULT_BLOCK_RUNS, log: Callable[[str], None] = print) -> dict:
    from gsop.parallel import default_workers

    out = paths.results(body)
    out.mkdir(parents=True, exist_ok=True)
    markers = out / ".s5"
    markers.mkdir(exist_ok=True)
    workers = default_workers() if workers is None else int(workers)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"
    params = MC.MCParams.from_config(cfg, runs=runs, block_runs=block_runs)
    profiles = profiles_from_frame(read_targets(paths.targets_parquet(body)))
    scored, info = s3.scored_targets(cfg, body, profiles, mode, log)
    passed = pd.read_parquet(out / "gates_passed.parquet")
    purposes = list(cfg.purposes(body))
    derived = weight_vector(derive_weights(cfg, body))
    convergence: dict = {}
    timings: dict = {}
    for name, profile, check in scored:
        setup = S.make_setup(cfg, body, profile)
        sub = passed[passed.target == name].sort_values("tile_id")
        ids, x = s4.load_gated_features(paths.earth_parquet, sub["tile_id"].to_numpy("int64"), list(setup.columns))
        landing_ok = sub["landing_ok"].to_numpy(bool)
        setup = setup.with_sigma(_sigma_from_file(out / "sigma_table.json", name, setup))
        base_w = dict(zip(setup.families, S.renormalise(derived, setup.families)))
        chunk_rows = memory_chunk_rows(cfg, len(setup.columns))
        for p in purposes:
            res = apply_purpose(cfg, base_w, p, body)
            sel = landing_ok if res.extra_gate is not None else np.ones(len(ids), dtype=bool)
            xs, ts = x[sel], ids[sel]
            key = f"{name}/{p}"
            mc_path = out / f"mc_{name}_{p}.parquet"
            marker = markers / f"{name}__{p}.json"
            mcs = MC.make_mc_setup(cfg, setup, p)
            fp = MC.fingerprint(xs, mcs, params)
            if marker.exists() and mc_path.exists() and json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fp:
                done = json.loads(marker.read_text(encoding="utf-8"))
                convergence[key], timings[key] = done["convergence"], done["timing"]
                log(f"[s5] {key}: already done (marker matches), skipped")
                continue
            if len(ts) == 0:
                log(f"[s5] {key}: no tile passes this profile's gate, nothing to simulate")
                continue
            free, budget = memory_numbers()
            MC.check_memory_budget(len(ts), len(setup.columns), workers, len(MC.scaled_checkpoints(params.runs, params.checkpoints)), free, budget, chunk_rows, log)
            t0 = time.perf_counter()
            result = MC.simulate(xs, ts, mcs, params, work_dir=out / ".mc_work" / f"{name}__{p}", workers=workers, chunk_rows=chunk_rows, log=log)
            tbl = result.table.copy()
            tbl.insert(0, "target", name)
            tbl.insert(1, "profile", p)
            tbl.insert(2, "confidence", res.confidence)
            tbl["fails_own_gate"] = check.fails_own_gate
            tbl["own_gate_reasons"] = ";".join(check.reasons)
            _atomic_parquet(tbl, mc_path)
            timing = {**result.timing, "total_seconds": round(time.perf_counter() - t0, 3), "candidates": result.candidates, "robust": int(result.table.robust.sum())}
            convergence[key], timings[key] = result.convergence, timing
            atomic_write_text(marker, json.dumps({"fingerprint": fp, "convergence": result.convergence, "timing": timing}, indent=2) + "\n")
            log(f"[s5] {key}: done in {timing['total_seconds']} s")
    atomic_write_text(out / "convergence.json", json.dumps(convergence, indent=2, sort_keys=True) + "\n")
    summary = {"body": body, "runs": params.runs, "workers": workers, "timing": timings, **info}
    atomic_write_text(out / "mc_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage(cfg: Config, body: str, paths: Paths, runs: Optional[int] = None, workers: Optional[int] = None, target_check: Optional[str] = None,
          log: Callable[[str], None] = print) -> Stage:
    out = paths.results(body)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"
    n_runs = cfg.run.runs if runs is None else int(runs)

    def run(ctx) -> None:
        run_simulate(cfg, body, paths, runs, workers, target_check, log=log)

    return Stage("s5", run, inputs=[paths.earth_parquet, paths.targets_parquet(body), out / "gates_passed.parquet", out / "sigma_table.json"],
                 outputs=[out / "convergence.json", out / "mc_summary.json"],
                 config_parts=["weights", "purposes", "features", "run", "gates", "targets"], code_version=f"s5:{n_runs}:{_code_hash()}:{mode}")


def _code_hash() -> str:
    from gsop.hashing import sha256_file

    here = Path(__file__).resolve().parent.parent
    return "".join(sha256_file(p)[:8] for p in (__file__, here / "montecarlo.py", here / "similarity.py"))
