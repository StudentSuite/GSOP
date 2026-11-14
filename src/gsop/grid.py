"""The equal-area tile grid (decision D1): EPSG:6933, 6000 m cells, one fixed origin.

* Origin: the west edge ``x0`` and north edge ``y_top`` of the lattice, in EPSG:6933 metres. They are the extent of the whole globe
  (longitude -180 to 180, latitude -90 to 90) snapped **outward** to a multiple of the tile size, so every point of the globe falls
  inside the grid and tile boundaries sit at multiples of 6000 m (the same lattice the Earth Engine export uses).
* Tile ids: ``tile_id = iy * nx + ix`` (int64), ``iy`` counted from the north edge, ``ix`` from the west edge. The inverse is exact.
* A point exactly on the antimeridian is assigned to the western edge tile (longitude is wrapped into [-180, 180)); the poles belong to
  the northernmost and southernmost rows.
* EPSG:6933 is a cylindrical equal-area projection, so every cell has the same true area (about 36 km2) at every latitude; cells are
  squares only in projected space, not on the ground.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box

from gsop.atomic import atomic_write_text
from gsop.config import Config

EARTH_MEAN_RADIUS_KM = 6371.0088  # IAU / IUGG mean radius, for the haversine distance only (not for tile geometry)
WGS84 = "EPSG:4326"


@lru_cache(maxsize=4)
def _transformers(crs: str) -> tuple[Transformer, Transformer]:
    return (Transformer.from_crs(WGS84, crs, always_xy=True), Transformer.from_crs(crs, WGS84, always_xy=True))


@dataclass(frozen=True)
class Grid:
    x0: int  # west edge, metres
    y_top: int  # north edge, metres
    nx: int
    ny: int
    tile_m: int
    crs: str = "EPSG:6933"

    # ---- construction ------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config) -> "Grid":
        ops = cfg.features.operators
        tile = int(ops.tile_m)
        fwd, _ = _transformers(ops.grid_crs)
        x_west, _ = fwd.transform(-180.0, 0.0)
        x_east, _ = fwd.transform(180.0, 0.0)
        _, y_south = fwd.transform(0.0, -90.0)
        _, y_north = fwd.transform(0.0, 90.0)
        x0 = math.floor(x_west / tile) * tile
        x1 = math.ceil(x_east / tile) * tile
        y_top = math.ceil(y_north / tile) * tile
        y_bot = math.floor(y_south / tile) * tile
        return cls(int(x0), int(y_top), int((x1 - x0) // tile), int((y_top - y_bot) // tile), tile, ops.grid_crs)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area_km2(self) -> float:
        return (self.tile_m / 1000.0) ** 2

    # ---- projection helpers ------------------------------------------------------------------------------
    def lonlat_to_xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = _transformers(self.crs)[0].transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
        return np.asarray(x), np.asarray(y)

    def xy_to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = _transformers(self.crs)[1].transform(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))
        return np.asarray(lon), np.asarray(lat)

    # ---- ids ---------------------------------------------------------------------------------------------
    def rowcol_to_id(self, iy, ix) -> np.ndarray:
        return np.asarray(iy, dtype="int64") * self.nx + np.asarray(ix, dtype="int64")

    def id_to_rowcol(self, tile_id) -> tuple[np.ndarray, np.ndarray]:
        tid = np.asarray(tile_id, dtype="int64")
        return tid // self.nx, tid % self.nx

    def lonlat_to_tile_id(self, lon, lat, wrap_lon: bool = True) -> np.ndarray:
        """Tile id of each point; -1 for points outside the grid (latitude beyond +-90, non-finite input)."""
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        if wrap_lon:
            lon = ((lon + 180.0) % 360.0) - 180.0
        x, y = self.lonlat_to_xy(lon, np.clip(lat, -90.0, 90.0))
        ix = np.floor((x - self.x0) / self.tile_m)
        iy = np.floor((self.y_top - y) / self.tile_m)
        ok = np.isfinite(ix) & np.isfinite(iy) & (np.abs(lat) <= 90.0) & (ix >= 0) & (iy >= 0)
        ix = np.minimum(np.where(ok, ix, 0), self.nx - 1)
        iy = np.minimum(np.where(ok, iy, 0), self.ny - 1)
        out = self.rowcol_to_id(iy, ix)
        return np.where(ok, out, -1)

    def tile_centroid_xy(self, tile_id) -> tuple[np.ndarray, np.ndarray]:
        iy, ix = self.id_to_rowcol(tile_id)
        return self.x0 + (ix + 0.5) * self.tile_m, self.y_top - (iy + 0.5) * self.tile_m

    def tile_centroid_lonlat(self, tile_id) -> tuple[np.ndarray, np.ndarray]:
        return self.xy_to_lonlat(*self.tile_centroid_xy(tile_id))

    def tile_polygon(self, tile_id: int, lonlat: bool = False, densify: int = 20) -> Polygon:
        """Polygon of one tile, built on demand: projected metres, or longitude/latitude with each edge densified."""
        iy, ix = (int(v) for v in self.id_to_rowcol(int(tile_id)))
        west, north = self.x0 + ix * self.tile_m, self.y_top - iy * self.tile_m
        poly = box(west, north - self.tile_m, west + self.tile_m, north)
        if not lonlat:
            return poly
        t = np.linspace(0.0, 1.0, densify, endpoint=False)
        south = north - self.tile_m
        east = west + self.tile_m
        xs = np.concatenate([west + t * self.tile_m, np.full(densify, east), east - t * self.tile_m, np.full(densify, west)])
        ys = np.concatenate([np.full(densify, south), south + t * self.tile_m, np.full(densify, north), north - t * self.tile_m])
        lon, lat = self.xy_to_lonlat(xs, ys)
        return Polygon(np.column_stack([lon, lat]))

    # ---- grid.json ---------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "crs": self.crs, "cell_size_m": self.tile_m,
            "origin": {"x_west_m": self.x0, "y_north_m": self.y_top},
            "shape": {"ny": self.ny, "nx": self.nx}, "cell_count": self.n_cells,
            "extent_m": {"x_min": self.x0, "x_max": self.x0 + self.nx * self.tile_m, "y_max": self.y_top, "y_min": self.y_top - self.ny * self.tile_m},
            "cell_area_km2": self.cell_area_km2,
            "tile_id": "iy * nx + ix, iy counted from the north edge, ix from the west edge (int64)",
        }


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km between points (vectorised, numpy broadcasting, degrees in)."""
    p1, p2 = np.radians(np.asarray(lat1, dtype="float64")), np.radians(np.asarray(lat2, dtype="float64"))
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype="float64")) - np.radians(np.asarray(lon1, dtype="float64"))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_MEAN_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def write_grid_json(path: Union[str, Path], grid: Grid, extra: Union[dict, None] = None) -> Path:
    info = {"schema_version": 1, **grid.to_dict(), **(extra or {})}
    return atomic_write_text(path, json.dumps(info, indent=2) + "\n")
