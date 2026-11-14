from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import test_build_parquet as TB
from gsop.cli import app
from gsop.config import load_config
from gsop.earth import build_parquet as B
from gsop.grid import Grid
from gsop.runner import Runner
from gsop.stages import s1

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


@pytest.fixture
def earth(tmp_path):
    x0, y_top = 600000.0, 300000.0
    TB.make_raster(tmp_path / "a.tif", x0, y_top, seed=1)
    TB.make_raster(tmp_path / "b.tif", x0 + 10 * 6000.0, y_top, seed=2)
    out = tmp_path / "processed"
    B.build_earth_parquet([tmp_path / "a.tif", tmp_path / "b.tif"], CFG, out, max_cells=25, log=lambda m: None)
    return out / "earth_features_6km.parquet"


def test_earth_file_now_carries_land_frac(earth):
    df = pd.read_parquet(earth)
    assert "land_frac" in df.columns and df.land_frac.between(0.5, 1.0).all()


def test_tiles_parquet_columns_counts_and_ids(earth, tmp_path):
    logs = []
    out = tmp_path / "interim" / "tiles.parquet"
    info = s1.build_tiles(CFG, earth, out, batch_rows=17, log=logs.append)  # tiny batches exercise the chunked path
    t = pd.read_parquet(out)
    assert list(t.columns) == s1.COLUMNS and len(t) == info["land_tiles"] == 120
    assert t.tile_id.is_unique and t.land_frac.dtype == np.float32
    grid = Grid.from_config(CFG)
    assert np.array_equal(grid.lonlat_to_tile_id(t.lon, t.lat), t.tile_id.to_numpy())
    assert info["grid_cells"] == grid.n_cells and any("120 land tiles of" in m for m in logs)


def test_a_count_far_from_the_expectation_is_flagged(earth, tmp_path):
    logs = []
    info = s1.build_tiles(CFG, earth, tmp_path / "t.parquet", log=logs.append)
    assert info["flagged"] and info["off_fraction"] > 0.99 and any("WARNING" in m and "land mask" in m for m in logs)


def test_missing_earth_file_and_old_earth_file_are_refused(tmp_path):
    with pytest.raises(s1.TilesError, match="build the Earth file first"):
        s1.build_tiles(CFG, tmp_path / "nope.parquet", tmp_path / "t.parquet")
    old = tmp_path / "old.parquet"
    pd.DataFrame({"tile_id": [1], "lon": [0.0], "lat": [0.0]}).to_parquet(old)
    with pytest.raises(s1.TilesError, match="lacks column"):
        s1.build_tiles(CFG, old, tmp_path / "t.parquet")


def test_ids_that_disagree_with_the_grid_are_refused(earth, tmp_path):
    df = pd.read_parquet(earth)
    df["tile_id"] = df["tile_id"] + 1
    bad = tmp_path / "bad.parquet"
    df.to_parquet(bad)
    with pytest.raises(s1.TilesError, match="do not match the grid"):
        s1.build_tiles(CFG, bad, tmp_path / "t.parquet")
    assert not (tmp_path / "t.parquet").exists()


def test_runner_skips_the_second_run(earth, tmp_path):
    out = tmp_path / "interim" / "tiles.parquet"
    st = s1.stage(CFG, earth, out, log=lambda m: None)
    runner = lambda: Runner(tmp_path / "interim", CFG, base_dir=tmp_path, log=lambda m: None, write_meta=False)  # noqa: E731
    assert [r.action for r in runner().run([st])] == ["ran"]
    assert [r.action for r in runner().run([st])] == ["skipped"]
    assert not (tmp_path / "interim" / "run_meta.json").exists()
    assert [r.action for r in runner().run([st], force=["s1"])] == ["ran"]


def test_cli_reports_a_clear_error_when_the_earth_file_is_missing():
    result = CliRunner().invoke(app, ["s1", "tiles"])
    if not (Path(__file__).resolve().parents[1] / "data" / "processed" / "earth_features_6km.parquet").exists():
        assert result.exit_code == 1 and "build the Earth file first" in result.output
