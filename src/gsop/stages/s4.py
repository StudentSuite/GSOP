"""s4 score: the deterministic base similarity, 0-100, per target x purpose profile (issue C14).

Outputs in ``results/<body>/``:

* ``scores_base.parquet``  one row per (target, tile) that passed the base gates, in ``tile_id`` order. Columns: ``target``, ``tile_id``,
  ``sim_<family>`` (family similarity, profile-independent, float32), ``illumination_match`` (the illumination family's similarity alone:
  Moon ``illumination``, Mars ``sunlight``), ``score_<profile>`` (0-100, float32; NaN where the tile does not pass that profile's gate, which
  only landing_mobility has), ``fails_own_gate``, ``own_gate_reasons``. (A wide layout instead of one row per profile: the family similarities do
  not depend on the profile, so this is several times smaller.)
* ``sigma_table.json``     sigma per target and column with its three terms.
* ``weights_used.json``    the weights each (target, profile) actually used, the profile's confidence and extra gate, and anything dropped.
* ``scores_summary.json``  run time and peak memory per target.

Tiles are sorted by ``tile_id`` before scoring so that output order, and the tie-breaking of later stages, never depend on the row order of the input.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gsop import similarity as S
from gsop import target_check as TC
from gsop import weights as W
from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.paths import Paths
from gsop.purposes import apply_purpose
from gsop.runner import Stage
from gsop.stages import s3
from gsop.targets import profiles_from_frame, read_targets

ILLUMINATION_FAMILY = {"moon": "illumination", "mars": "sunlight"}


class ScoreError(RuntimeError):
    pass


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def load_gated_features(earth_parquet: Path, tile_ids: np.ndarray, columns: list[str], batch_rows: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Features of exactly ``tile_ids`` (sorted ascending) from the Earth file: returns ``(ids sorted, X float32 (n, K))`` in tile_id order."""
    pf = pq.ParquetFile(earth_parquet)
    ids_out, parts = [], []
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["tile_id", *columns]):
        t = batch.to_pandas()
        tid = t["tile_id"].to_numpy("int64")
        pos = np.searchsorted(tile_ids, tid)
        hit = (pos < len(tile_ids)) & (tile_ids[np.minimum(pos, len(tile_ids) - 1)] == tid)
        if hit.any():
            ids_out.append(tid[hit])
            parts.append(t.loc[hit, columns].to_numpy("float32"))
    if not parts:
        return np.empty(0, dtype="int64"), np.empty((0, len(columns)), dtype="float32")
    ids, x = np.concatenate(ids_out), np.concatenate(parts)
    order = np.argsort(ids, kind="stable")
    ids, x = ids[order], x[order]
    if len(ids) != len(tile_ids):
        raise ScoreError(f"{len(tile_ids) - len(ids)} gated tile id(s) are missing from {earth_parquet.name}: the Earth file changed after s3")
    return ids, x


def run_scores(cfg: Config, body: str, earth_parquet: Path, targets_parquet: Path, gates_passed: Path, out_dir: Path,
               target_check: Optional[str] = None, chunk_rows: int = S.DEFAULT_CHUNK_ROWS, log: Callable[[str], None] = print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"
    profiles = profiles_from_frame(read_targets(targets_parquet))
    scored, info = s3.scored_targets(cfg, body, profiles, mode, log)
    passed = pq.read_table(gates_passed).to_pandas()
    purposes = list(cfg.purposes(body))
    derived = W.derive_weights(cfg, body)

    sigma_tables: dict = {}
    weights_used: dict = {"body": body, "derived_family_weights": W.weight_vector(derived), "targets": {}}
    timing: dict = {}
    fam_names_all: list[str] = []
    tmp = out_dir / ".scores_base.parquet.tmp"
    writer: Optional[pq.ParquetWriter] = None
    peak = _rss_mb()
    try:
        for name, profile, check in scored:
            t0 = time.perf_counter()
            setup = S.make_setup(cfg, body, profile)
            sub = passed[passed.target == name].sort_values("tile_id")
            tile_ids = sub["tile_id"].to_numpy("int64")
            landing_ok = sub["landing_ok"].to_numpy(bool)
            ids, x = load_gated_features(earth_parquet, tile_ids, list(setup.columns))
            if len(ids) == 0:
                log(f"[s4] {name}: no tile passed the gates, nothing to score")
                continue
            table = S.build_sigma_table(cfg, setup, {c: x[:, i] for i, c in enumerate(setup.columns)})
            sigma_tables[name] = table
            setup = setup.with_sigma(S.sigma_vector(setup, table))
            base_w = dict(zip(setup.families, S.renormalise(W.weight_vector(derived), setup.families)))
            prof_w, meta = {}, {}
            for p in purposes:
                res = apply_purpose(cfg, base_w, p, body)
                prof_w[p] = np.array([res.weights[f] for f in setup.families], dtype="float64")
                meta[p] = {"weights": res.weights, "confidence": res.confidence, "extra_gate": res.extra_gate.text if res.extra_gate else None}
            weights_used["targets"][name] = {"families": list(setup.families), "dropped": setup.dropped, "profiles": meta}
            fam_names = list(setup.families)
            fam_names_all = fam_names
            illum = ILLUMINATION_FAMILY[body]
            illum_idx = fam_names.index(illum) if illum in fam_names else None
            applies_landing = {p: (meta[p]["extra_gate"] is not None) for p in purposes}

            n = len(ids)
            sims = np.empty((n, len(fam_names)), dtype=np.float32)
            scores = {p: np.empty(n, dtype=np.float32) for p in purposes}
            for i in range(0, n, chunk_rows):
                blk = x[i : i + chunk_rows]
                fam = S.family_similarity(S.column_similarity(blk, setup.target_values, setup.sigma), setup.family_cols)
                sims[i : i + chunk_rows] = fam
                for p in purposes:
                    scores[p][i : i + chunk_rows] = S.weighted_score(fam, prof_w[p])
            for p in purposes:
                if applies_landing[p]:
                    scores[p][~landing_ok] = np.nan  # the landing profile's extra gate
            cols = {"target": np.full(n, name), "tile_id": ids}
            for j, f in enumerate(fam_names):
                cols[f"sim_{f}"] = sims[:, j]
            cols["illumination_match"] = sims[:, illum_idx] if illum_idx is not None else np.full(n, np.nan, dtype=np.float32)
            for p in purposes:
                cols[f"score_{p}"] = scores[p]
            cols["fails_own_gate"] = np.full(n, check.fails_own_gate)
            cols["own_gate_reasons"] = np.full(n, ";".join(check.reasons))
            tbl = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            peak = max(peak, _rss_mb())
            timing[name] = {"tiles": n, "seconds": round(time.perf_counter() - t0, 3), "peak_rss_mb": round(peak, 1)}
            log(f"[s4] {name}: {n:,} tiles x {len(purposes)} profiles in {timing[name]['seconds']} s (peak {peak:,.0f} MB); best base score {float(np.nanmax(scores['base'])):.2f}")
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    if writer is None:
        raise ScoreError("no target had any tile to score (all unscored, rejected by the target check, or fully gated out)")
    writer.close()
    tmp.replace(out_dir / "scores_base.parquet")
    S.write_sigma_table(out_dir / "sigma_table.json", sigma_tables, cfg.run.sigma_iqr_multiplier)
    atomic_write_text(out_dir / "weights_used.json", json.dumps(weights_used, indent=2, sort_keys=True) + "\n")
    summary = {"body": body, "timing": timing, "peak_rss_mb": round(peak, 1), "profiles": purposes, "families": fam_names_all, **info}
    atomic_write_text(out_dir / "scores_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage(cfg: Config, body: str, paths: Paths, target_check: Optional[str] = None, log: Callable[[str], None] = print) -> Stage:
    out = paths.results(body)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"

    def run(ctx) -> None:
        run_scores(cfg, body, paths.earth_parquet, paths.targets_parquet(body), out / "gates_passed.parquet", out, target_check, log=log)

    return Stage("s4", run, inputs=[paths.earth_parquet, paths.targets_parquet(body), out / "gates_passed.parquet"],
                 outputs=[out / "scores_base.parquet", out / "sigma_table.json", out / "weights_used.json", out / "scores_summary.json"],
                 config_parts=["weights", "purposes", "features", "run", "gates", "targets"], code_version=f"s4:{_code_hash()}:{mode}")


def _code_hash() -> str:
    from gsop.hashing import sha256_file

    here = Path(__file__).resolve().parent.parent
    return "".join(sha256_file(p)[:8] for p in (__file__, here / "similarity.py", here / "weights.py", here / "purposes.py"))
