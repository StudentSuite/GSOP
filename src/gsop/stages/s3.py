"""s3 gates: ``results/<body>/gates_passed.parquet``, ``gates_rejected.parquet`` and ``gates_summary.json`` (issue C13).

For every scored target (Arcadia, which has no data, is skipped) the Earth tiles are gated with ``gsop.gates``:

* ``gates_rejected.parquet``  tiles that failed a **base** gate (water/ice, vegetation, built-up/cropland, missing data):
  ``target, tile_id, reasons, near, fails_own_gate, own_gate_reasons``. ``reasons`` is the bitmask of ``gsop.gates.BITS``.
* ``gates_passed.parquet``    tiles that passed every base gate: ``target, tile_id, near, landing_ok, fails_own_gate, own_gate_reasons``.
  ``landing_ok`` is False where the body's landing_mobility slope gate fails (those tiles pass the other profiles but not landing_mobility).

So per target ``passed + rejected`` is every land tile. ``fails_own_gate`` and ``own_gate_reasons`` (decision D6) are on every row of the target.
``gates_summary.json`` counts tiles per reason code and per target and records the Mars target check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gsop import gates as G
from gsop import target_check as TC
from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.paths import Paths
from gsop.runner import Stage
from gsop.similarity import make_setup
from gsop.targets import profiles_from_frame, read_targets

REJECTED_SCHEMA = pa.schema([("target", pa.string()), ("tile_id", pa.int64()), ("reasons", pa.uint8()), ("near", pa.uint8()),
                             ("fails_own_gate", pa.bool_()), ("own_gate_reasons", pa.string())])
PASSED_SCHEMA = pa.schema([("target", pa.string()), ("tile_id", pa.int64()), ("near", pa.uint8()), ("landing_ok", pa.bool_()),
                           ("fails_own_gate", pa.bool_()), ("own_gate_reasons", pa.string())])
GATE_LAYER_COLUMNS = list(G.GATE_LAYER_COLUMNS.values())


def gate_columns(cfg: Config, body: str, required: list[str]) -> list[str]:
    """Earth columns s3 must read: the id, the gate layers, the weight columns and the landing gate column."""
    cols = ["tile_id", *GATE_LAYER_COLUMNS, *required]
    gate = G.landing_gate(cfg, body)
    if gate is not None:
        cols.append(gate.column)
    return list(dict.fromkeys(cols))


def run_gates(cfg: Config, body: str, earth_parquet: Path, targets_parquet: Path, out_dir: Path, target_check: Optional[str] = None,
              batch_rows: int = 500_000, log: Callable[[str], None] = print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"
    profiles = profiles_from_frame(read_targets(targets_parquet))
    pf = pq.ParquetFile(earth_parquet)
    n_tiles = pf.metadata.num_rows
    summary: dict = {"body": body, "tiles": n_tiles, "target_check_mode": mode, "bit_table": G.BITS, "reason_codes": G.reason_codes(cfg),
                     "targets": {}, "unscored": [], "not_scored_by_target_check": [], "validation": []}
    rej_tmp, pas_tmp = out_dir / ".gates_rejected.parquet.tmp", out_dir / ".gates_passed.parquet.tmp"
    rej_w, pas_w = pq.ParquetWriter(rej_tmp, REJECTED_SCHEMA, compression="zstd"), pq.ParquetWriter(pas_tmp, PASSED_SCHEMA, compression="zstd")
    try:
        for name, profile in profiles.items():
            if profile.unscored:
                summary["unscored"].append(name)
                log(f"[s3] {name}: unscored (no target data), skipped")
                continue
            check = TC.check_target(cfg, body, name, profile.values, mode)
            summary["validation"].append(check.validation_entry())
            if check.fails_own_gate:
                log(f"[s3] {name}: fails its own gate ({'; '.join(check.reasons)}); mode {check.mode}")
            if not check.scored:
                summary["not_scored_by_target_check"].append(name)
                log(f"[s3] {name}: not scored (strict target check)")
                continue
            setup = make_setup(cfg, body, profile)  # raises a clear error when a family has no usable target value
            flags = check.flags()
            counts = {"reasons": {}, "tiles": 0, "rejected": 0, "passed": 0, "passed_landing": 0, "near": {}}
            agg_reasons = np.zeros(len(G.BITS), dtype="int64")
            for batch in pf.iter_batches(batch_size=batch_rows, columns=gate_columns(cfg, body, list(setup.columns))):
                t = batch.to_pandas()
                cols = {c: t[c].to_numpy("float64") for c in t.columns if c != "tile_id"}
                res = G.evaluate_gates(cfg, body, cols, list(setup.columns))
                base_fail = (res.reasons & G.BASE_MASK) != 0
                tid = t["tile_id"].to_numpy("int64")
                k = int(base_fail.sum())
                m = len(t)
                if k:
                    rej_w.write_table(pa.table({"target": np.full(k, name), "tile_id": tid[base_fail], "reasons": res.reasons[base_fail], "near": res.near[base_fail],
                                                "fails_own_gate": np.full(k, flags["fails_own_gate"]), "own_gate_reasons": np.full(k, flags["own_gate_reasons"])}, schema=REJECTED_SCHEMA))
                ok = ~base_fail
                if ok.any():
                    pas_w.write_table(pa.table({"target": np.full(int(ok.sum()), name), "tile_id": tid[ok], "near": res.near[ok],
                                                "landing_ok": (res.reasons[ok] & G.LANDING_BIT) == 0,
                                                "fails_own_gate": np.full(int(ok.sum()), flags["fails_own_gate"]),
                                                "own_gate_reasons": np.full(int(ok.sum()), flags["own_gate_reasons"])}, schema=PASSED_SCHEMA))
                counts["tiles"] += m
                counts["rejected"] += k
                counts["passed"] += int(ok.sum())
                counts["passed_landing"] += int((ok & ((res.reasons & G.LANDING_BIT) == 0)).sum())
                for gate_name, bit in G.BITS.items():
                    code = cfg.gates.reason_codes[gate_name]
                    counts["reasons"][code] = counts["reasons"].get(code, 0) + int(((res.reasons & bit) != 0).sum())
                    counts["near"][code] = counts["near"].get(code, 0) + int(((res.near & bit) != 0).sum())
            assert counts["passed"] + counts["rejected"] == counts["tiles"] == n_tiles
            counts["fails_own_gate"], counts["own_gate_reasons"] = flags["fails_own_gate"], list(check.reasons)
            counts["dropped_columns"], counts["dropped_families"] = setup.dropped["columns"], setup.dropped["families"]
            summary["targets"][name] = counts
            log(f"[s3] {name}: {counts['passed']:,} passed, {counts['rejected']:,} rejected of {counts['tiles']:,}; "
                + ", ".join(f"{c}={v:,}" for c, v in counts["reasons"].items()))
    except BaseException:
        rej_w.close(), pas_w.close()
        rej_tmp.unlink(missing_ok=True), pas_tmp.unlink(missing_ok=True)
        raise
    rej_w.close()
    pas_w.close()
    rej_tmp.replace(out_dir / "gates_rejected.parquet")
    pas_tmp.replace(out_dir / "gates_passed.parquet")
    atomic_write_text(out_dir / "gates_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage(cfg: Config, body: str, paths: Paths, target_check: Optional[str] = None, log: Callable[[str], None] = print) -> Stage:
    out = paths.results(body)
    mode = TC.effective_mode(cfg, target_check) if body == "mars" else "warn"

    def run(ctx) -> None:
        run_gates(cfg, body, paths.earth_parquet, paths.targets_parquet(body), out, target_check, log=log)

    return Stage("s3", run, inputs=[paths.earth_parquet, paths.targets_parquet(body)],
                 outputs=[out / "gates_passed.parquet", out / "gates_rejected.parquet", out / "gates_summary.json"],
                 config_parts=["gates", "purposes", "weights", "features", "run", "targets"], code_version=f"s3:{_code_hash()}:{mode}")


def _code_hash() -> str:
    from gsop.hashing import sha256_file

    here = Path(__file__).resolve().parent.parent
    return sha256_file(__file__)[:12] + sha256_file(here / "gates.py")[:12] + sha256_file(here / "target_check.py")[:12]
