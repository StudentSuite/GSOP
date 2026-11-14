# Operator check: Earth Engine versus numpy

**Status: NOT RUN.** It needs an Earth Engine login (issue B01), so no number is recorded. Run `uv run gsop check-operators`; it overwrites this file with the real table and exits with code 1 if any of `slope_med`, `relief` or `rough_a` differs by more than 10% (`operator_check_tolerance` in `config/run.yaml`).

If it fails, open a fix issue **before any global export**.

What it compares: the Earth Engine export expressions and `operators.tile_features` on a 3 x 3 tile window (18 km) centred on Lonar, both starting from the same elevation data (the numpy path uses the DEM fetched on a 25 m grid), plus the intermediate 500 m DEM and slope grids so a mismatch can be located. Known acceptable differences: bilinear resampling to the 25 m grid for numpy against direct block averaging in Earth Engine; histogram-based percentiles in Earth Engine against interpolated percentiles in numpy; the closed-form roughness is exact only on full tiles.
