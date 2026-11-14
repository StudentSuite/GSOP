import json
from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from gsop.config import load_config
from gsop.earth import gee_export as G
from gsop.grid import EARTH_MEAN_RADIUS_KM, Grid, haversine_km, write_grid_json

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
GRID = Grid.from_config(CFG)


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))  # uniform on the sphere


def test_grid_matches_the_lattice_of_the_earth_engine_export():
    chunks = G.plan_chunks(CFG)
    tile = 6000
    assert GRID.x0 == min(c.x0 for c in chunks) and GRID.y_top == max(c.y1 for c in chunks)
    assert GRID.nx == (max(c.x1 for c in chunks) - GRID.x0) // tile and GRID.ny == (GRID.y_top - min(c.y0 for c in chunks)) // tile
    assert GRID.x0 % tile == 0 and GRID.y_top % tile == 0
    assert 5000 < GRID.nx < 6500 and 2000 < GRID.ny < 2900 and 13e6 < GRID.n_cells < 16e6
    assert GRID.cell_area_km2 == 36.0


def test_round_trip_for_10000_random_points():
    lon, lat = random_points(10_000)
    ids = GRID.lonlat_to_tile_id(lon, lat)
    assert (ids >= 0).all() and ids.dtype == np.int64 and (ids < GRID.n_cells).all()
    clon, clat = GRID.tile_centroid_lonlat(ids)
    assert np.array_equal(GRID.lonlat_to_tile_id(clon, clat), ids)  # centroid maps back to the same tile
    x, y = GRID.lonlat_to_xy(lon, lat)
    cx, cy = GRID.tile_centroid_xy(ids)
    assert (np.abs(x - cx) <= 3000.0 + 1e-6).all() and (np.abs(y - cy) <= 3000.0 + 1e-6).all()  # the point lies inside its tile


def test_row_col_conversion_is_exact_and_the_inverse():
    iy, ix = np.array([0, 5, GRID.ny - 1]), np.array([0, 7, GRID.nx - 1])
    ids = GRID.rowcol_to_id(iy, ix)
    back_y, back_x = GRID.id_to_rowcol(ids)
    assert np.array_equal(back_y, iy) and np.array_equal(back_x, ix) and ids[-1] == GRID.n_cells - 1


def test_poles_and_antimeridian_are_handled():
    ids = GRID.lonlat_to_tile_id([0.0, 0.0, 179.999, -180.0, 180.0, -179.999], [90.0, -90.0, 10.0, 10.0, 10.0, 10.0])
    iy, ix = GRID.id_to_rowcol(ids)
    assert iy[0] == 0 and iy[1] == GRID.ny - 1  # north and south pole rows
    assert ix[2] == GRID.nx - 1 and ix[3] == 0 and ix[4] == ix[3] and ix[5] == 0  # 180 wraps to the west edge tile
    assert (GRID.lonlat_to_tile_id([0.0, 0.0], [90.1, np.nan]) == -1).all()  # outside the globe / not a number
    # tiles on either side of the antimeridian are distinct and far apart in the grid
    assert ix[2] != ix[5]


def test_wrapped_longitudes_are_the_same_tile():
    lon, lat = random_points(500, seed=2)
    assert np.array_equal(GRID.lonlat_to_tile_id(lon, lat), GRID.lonlat_to_tile_id(lon + 360.0, lat))


def test_polygon_is_a_6_km_square_in_projected_space_and_has_equal_true_area_everywhere():
    lon, lat = random_points(40, seed=4)
    lon = np.concatenate([lon, [0.0, 0.0, 100.0, -100.0]])
    lat = np.concatenate([lat, [0.0, 80.0, -60.0, 85.0]])
    ids = GRID.lonlat_to_tile_id(lon, lat)
    geod = Geod(ellps="WGS84")
    areas = []
    for tid in ids:
        sq = GRID.tile_polygon(int(tid))
        assert sq.area == pytest.approx(36e6) and sq.bounds[2] - sq.bounds[0] == 6000 and sq.bounds[3] - sq.bounds[1] == 6000
        poly = GRID.tile_polygon(int(tid), lonlat=True, densify=40)
        areas.append(abs(geod.geometry_area_perimeter(poly)[0]) / 1e6)
    # EPSG:6933 is equal-area on the WGS84 ellipsoid: every cell is 36 km2 on the ground (densified edges leave a tiny residual)
    assert np.allclose(areas, 36.0, rtol=2e-3), (min(areas), max(areas))


def test_haversine_matches_known_distances_within_0_1_percent():
    quarter = np.pi / 2 * EARTH_MEAN_RADIUS_KM
    assert haversine_km(0, 0, 90, 0) == pytest.approx(quarter, rel=1e-9)  # equator to the point at longitude 90
    assert haversine_km(0, 0, 0, 90) == pytest.approx(quarter, rel=1e-9)  # equator to the pole
    assert haversine_km(0, 0, 180, 0) == pytest.approx(2 * quarter, rel=1e-9)
    # London to Paris, published as about 344 km great-circle
    assert haversine_km(-0.1276, 51.5072, 2.3522, 48.8566) == pytest.approx(343.5, rel=0.002)
    # one degree of latitude is 111.19 km
    assert haversine_km(10, 20, 10, 21) == pytest.approx(111.195, rel=1e-3)
    # vectorised and broadcasting
    d = haversine_km(np.zeros(3), np.zeros(3), np.array([0, 90, 180]), np.zeros(3))
    assert d.shape == (3,) and d[0] == 0.0


def test_grid_json_documents_origin_extent_and_cell_count(tmp_path):
    path = write_grid_json(tmp_path / "grid.json", GRID)
    info = json.loads(path.read_text(encoding="utf-8"))
    assert info["crs"] == "EPSG:6933" and info["cell_size_m"] == 6000 and info["cell_count"] == GRID.n_cells
    assert info["origin"] == {"x_west_m": GRID.x0, "y_north_m": GRID.y_top}
    assert info["extent_m"]["x_max"] - info["extent_m"]["x_min"] == GRID.nx * 6000
    assert info["shape"] == {"ny": GRID.ny, "nx": GRID.nx} and "iy * nx + ix" in info["tile_id"]
