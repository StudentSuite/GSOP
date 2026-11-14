"""s1 tiles: the tile universe, ``data/interim/tiles.parquet`` (issue C02).

**Land mask source (documented choice).** The land flag is not computed again: the Earth Engine export carries ``land_frac``, the share of
500 m MODIS MCD12Q1 cells in a tile whose land/water mask ``LW`` says land (see ``docs/PROVENANCE.md``), and ``gsop s2 earth --build-from``
keeps the tiles with ``land_frac >= earth_build.land_min_fraction`` (0.5). So s1 runs **after** the Earth file exists and reads the tile list
from it; the whole-grid cell count comes from ``gsop.grid`` (about 14 million cells, about 4 million of them land). A verified land-mask source
independent of the Earth file was not found, so the Earth file's flag is the single source.

Columns: ``tile_id, lon, lat, x, y, land_frac``. Memory-safe: the Earth parquet is read in record batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gsop.atomic import atomic_write_bytes
from gsop.config import Config
from gsop.grid import Grid
from gsop.runner import Stage

COLUMNS = ["tile_id", "lon", "lat", "x", "y", "land_frac"]
SOURCE_COLUMNS = ["tile_id", "lon", "lat", "x_m", "y_m", "land_frac"]
ID_CHECK_SAMPLE = 20_000


class TilesError(RuntimeError):
    pass


def build_tiles(cfg: Config, earth_parquet: Path, out_parquet: Path, batch_rows: int = 500_000,
                log: Callable[[str], None] = print) -> dict:
    earth_parquet, out_parquet = Path(earth_parquet), Path(out_parquet)
    if not earth_parquet.is_file():
        raise TilesError(f"{earth_parquet} does not exist: build the Earth file first (`gsop s2 earth`, issues B16 to B18)")
    pf = pq.ParquetFile(earth_parquet)
    missing = [c for c in SOURCE_COLUMNS if c not in pf.schema_arrow.names]
    if missing:
        raise TilesError(f"{earth_parquet} lacks column(s) {missing}; rebuild it with the current `gsop s2 earth --build-from`")
    grid = Grid.from_config(cfg)
    schema = pa.schema([("tile_id", pa.int64()), ("lon", pa.float64()), ("lat", pa.float64()), ("x", pa.float64()), ("y", pa.float64()), ("land_frac", pa.float32())])
    tmp = out_parquet.with_name("." + out_parquet.name + ".tmp")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    n, checked, bad_ids, last_id, unique_ok = 0, 0, 0, -1, True
    try:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=SOURCE_COLUMNS):
            t = batch.to_pandas()
            tid = t["tile_id"].to_numpy("int64")
            if checked < ID_CHECK_SAMPLE:  # the ids stored in the Earth file must be the ids the grid assigns to their coordinates
                k = min(len(t), ID_CHECK_SAMPLE - checked)
                bad_ids += int((grid.lonlat_to_tile_id(t["lon"].to_numpy()[:k], t["lat"].to_numpy()[:k]) != tid[:k]).sum())
                checked += k
            if len(tid) and (tid.min() <= last_id or not (np.diff(tid) > 0).all()):
                unique_ok = False  # not sorted ascending: uniqueness is then checked below on the full set
            last_id = int(tid.max()) if len(tid) else last_id
            out = pa.table({"tile_id": tid, "lon": t["lon"].to_numpy(), "lat": t["lat"].to_numpy(), "x": t["x_m"].to_numpy(),
                            "y": t["y_m"].to_numpy(), "land_frac": t["land_frac"].to_numpy("float32")}, schema=schema)
            writer.write_table(out)
            n += len(t)
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    if bad_ids:
        tmp.unlink(missing_ok=True)
        raise TilesError(f"{bad_ids} of the first {checked} tile ids in {earth_parquet.name} do not match the grid: the Earth file and the grid disagree")
    ids = pq.read_table(tmp, columns=["tile_id"]).column(0).to_numpy()
    if len(np.unique(ids)) != n:
        tmp.unlink(missing_ok=True)
        raise TilesError("duplicate tile ids in the Earth file")
    tmp.replace(out_parquet)
    exp = cfg.features.earth_build["expected_land_tiles"]
    tol = cfg.features.earth_build["expected_land_tolerance"]
    off = abs(n - exp) / exp
    flag = off > tol
    log(f"[s1] {n:,} land tiles of {grid.n_cells:,} grid cells ({n / grid.n_cells:.1%}); expected about {exp:,}")
    if flag:
        log(f"[s1] WARNING: the land tile count is {off:.0%} away from the expected {exp:,} (limit {tol:.0%}): check the land mask before using this file")
    return {"land_tiles": n, "grid_cells": grid.n_cells, "expected": exp, "off_fraction": off, "flagged": bool(flag), "path": str(out_parquet)}


def stage(cfg: Config, earth_parquet: Path, out_parquet: Path, log: Callable[[str], None] = print) -> Stage:
    def run(ctx) -> None:
        build_tiles(cfg, earth_parquet, out_parquet, log=log)

    return Stage("s1", run, inputs=[earth_parquet], outputs=[out_parquet], config_parts=["features"])
