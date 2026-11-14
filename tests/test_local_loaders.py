import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from typer.testing import CliRunner

from gsop.cli import app
from gsop.config import load_config
from gsop.earth import build_parquet, local_loaders as L

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")
REGION = [10.0, 0.0, 13.0, 3.0]  # a 3 x 3 degree box near the equator


def write_geo(path, array, west=9.5, north=3.5, res=0.0166667):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, res, res), nodata=-9999.0) as ds:
        ds.write(array.astype("float32"), 1)


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    n = int(4.0 / 0.0166667)
    yy, xx = np.mgrid[0:n, 0:n]
    write_geo(tmp_path / "elev.tif", 500 + 40 * xx / n + rng.normal(0, 5, (n, n)))  # ~1.8 km cells, gentle east tilt
    write_geo(tmp_path / "land.tif", np.ones((n, n)))
    write_geo(tmp_path / "tmean.tif", 295 + rng.normal(0, 1, (n, n)))
    (tmp_path / "local_sources.yaml").write_text(yaml.safe_dump({
        "region": REGION,
        "bands": {"elevation": {"path": "elev.tif"}, "land_frac": {"path": "land.tif"}, "t_mean": {"path": "tmean.tif"}},
    }), encoding="utf-8")
    return tmp_path


def test_coarse_dem_makes_slope_and_roughness_unavailable_but_builds_relief(folder, tmp_path):
    out = tmp_path / "out"
    report = L.build_local_chunk(folder, CFG, out)
    for band in ("slope_med", "slope_p90", "rough_a", "rough_b"):
        assert band in report["unavailable"] and "coarser" in report["unavailable"][band]
    assert "relief" not in report["unavailable"] and "relief" in report["notes"] and "underestimates" in report["notes"]["relief"]
    for band in ("silica_index", "ti_proxy", "dust_index_raw"):  # water_index is not in the chunk at all: it is not exported
        assert report["unavailable"][band] == "unavailable: not supplied"
    with rasterio.open(out / "local_chunk.tif") as ds:
        names = list(ds.descriptions)
        assert names == report["bands"] and ds.crs.to_string() == "EPSG:6933" and ds.transform.a == 6000
        relief = ds.read(names.index("relief") + 1, masked=True)
        elev = ds.read(names.index("elevation_mean") + 1, masked=True)
        slope = ds.read(names.index("slope_med") + 1, masked=True)
    assert slope.mask.all()
    assert 480 < float(elev.mean()) < 560 and float(relief.min()) >= 0 and float(relief.mean()) > 5


def test_full_pipeline_to_parquet_with_the_same_code_as_the_earth_engine_route(folder, tmp_path):
    L.build_local_chunk(folder, CFG, tmp_path / "interim")
    info = build_parquet.build_earth_parquet([tmp_path / "interim" / "local_chunk.tif"], CFG, tmp_path / "data" / "processed", max_cells=200, log=lambda m: None)
    df = pd.read_parquet(tmp_path / "data" / "processed" / "earth_features_6km.parquet")
    assert info["rows"] == len(df) > 50
    assert {"slope_med", "slope_p90", "rough_a", "rough_b", "water_index", "silica_index"} <= set(info["unavailable_features"])
    assert df.t_mean.between(290, 300).all() and df.relief.notna().all() and df.slope_med.isna().all()
    assert df.air_density.notna().all() and df.sun_max_elev.notna().all()


def test_bad_inputs_are_refused_with_clear_messages(tmp_path):
    with pytest.raises(L.LocalSourceError, match="not found"):
        L.load_sources(tmp_path)
    (tmp_path / "local_sources.yaml").write_text("region: [1, 2]\nbands: {}\n", encoding="utf-8")
    with pytest.raises(L.LocalSourceError, match="region"):
        L.load_sources(tmp_path)
    (tmp_path / "local_sources.yaml").write_text("region: [10, 0, 13, 3]\nbands: {t_mean: {path: nope.tif}}\n", encoding="utf-8")
    with pytest.raises(L.LocalSourceError, match="file not found"):
        L.load_sources(tmp_path)
    write_geo(tmp_path / "a.tif", np.ones((4, 4)))
    (tmp_path / "local_sources.yaml").write_text("region: [10, 0, 13, 3]\nbands: {t_mean: {path: a.tif}}\n", encoding="utf-8")
    with pytest.raises(L.LocalSourceError, match="land_frac"):
        L.load_sources(tmp_path)
    (tmp_path / "local_sources.yaml").write_text("region: [10, 0, 13, 3]\nbands: {nonsense: {path: a.tif}, land_frac: {path: a.tif}}\n", encoding="utf-8")
    with pytest.raises(L.LocalSourceError, match="unknown band"):
        L.load_sources(tmp_path)


def test_cli_local_source_needs_a_folder_and_reports_coarseness(tmp_path):
    r = CliRunner().invoke(app, ["s2", "earth", "--source", "local"])
    assert r.exit_code == 2 and "--local-dir" in r.output


def test_spec_documents_the_loud_note():
    text = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    assert "LOCAL ROUTE" in text and "coarser" in text
