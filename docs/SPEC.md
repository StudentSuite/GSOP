# GSOP specification

This file is written as the code is built. It must be enough for a person outside the team to re-implement the score: every formula, every assumption, every deviation from the published plan.

**Rule: every pull request that changes behaviour updates this file in the same pull request.**

Source of truth for numbers and decisions: `docs/source/` and `docs/DECISIONS.md` (D1 to D10). Unverified items: `docs/UNVERIFIED.md`. Datasets: `docs/PROVENANCE.md`.

Each section below says which pipeline stage it belongs to. A stage with no text yet says `TODO` and the issue that fills it.

## Pipeline map

| Stage | Command | Section |
|---|---|---|
| s1 | `gsop s1 tiles` | Grid |
| s2 | `gsop s2 earth`, `gsop s2 targets` | Features, Operators |
| s3 | `gsop s3 gates` | Gates |
| s4 | `gsop s4 score` | Similarity, Sigma rule, Weights, Purposes |
| s5 | `gsop s5 simulate` | Monte Carlo |
| s6 | `gsop s6 ml` | ML |
| s7 | `gsop s7 validate` | Validation |
| s8 | `gsop s8 export` | Export |
| s9 | `gsop s9 discover` | Discovery |

## Grid (s1)

Decision D1: 6 km x 6 km tiles on the equal-area grid EPSG:6933 (6000 m cells), land only, polar ice gated out. `src/gsop/grid.py` defines it once.

- **Origin and extent.** The grid is the extent of the whole globe (longitude -180 to 180, latitude -90 to 90 projected to EPSG:6933) snapped *outward* to multiples of 6000 m: west edge `x0`, north edge `y_top`, `nx` columns, `ny` rows (about 5.8 thousand by 2.4 thousand, about 14 million cells, of which about 4 million are land; the exact numbers are in `grid.json`). It is the same lattice the Earth Engine export uses (tested).
- **Tile id.** `tile_id = iy * nx + ix` as int64, with `iy` counted from the north edge and `ix` from the west edge; the inverse is exact. Centroid longitude and latitude come from the projection inverse (pyproj), polygons are built on demand.
- **Edges.** A longitude is wrapped into [-180, 180), so a point exactly on the antimeridian goes to the western edge tile; the poles belong to the first and last row; points beyond +-90 degrees or non-finite give id -1.
- **Equal area.** EPSG:6933 is cylindrical equal-area, so every cell has the same true area on the WGS84 ellipsoid (36 km2) at every latitude, but cells are squares only in projected space (tested against a geodesic area on densified polygons, within 0.2%).
- **Tile universe (`gsop s1 tiles`, `src/gsop/stages/s1.py`).** Writes `data/interim/tiles.parquet` with `tile_id, lon, lat, x, y, land_frac`, one row per land tile. **Land mask source:** the Earth file's `land_frac` (share of 500 m MODIS MCD12Q1 `LW` cells that are land, tile is land at 0.5 or more), not recomputed here, so s1 runs after the Earth file exists; no independent verified land mask was found. It checks the stored ids against the grid, prints `N land tiles of M cells`, and flags a count more than 25% (tunable, unsourced) from the expected 4 million. The runner skips it when the Earth file and config are unchanged.
- **Distance.** The haversine helper uses the mean Earth radius 6371.0088 km and is used only for the 20 km site matching and the 100 km novelty radius (not for tile geometry).

## Features (s2)

Columns, sources and uncertainty floors are in `config/features.yaml`. `water_index` and `silica_index` are cross-instrument proxies and are the two weakest comparisons in the index. Other feature definitions are added below by the issue that builds each one.

### Earth export orchestration (issue B16)

`gsop s2 earth --source gee [--region W,S,E,N] [--dry-run]`. All feature bands (`gee_export.FEATURE_BUILDERS`, plus `elevation_mean`) form one multi-band image exported at 6000 m on EPSG:6933 as float32 GeoTIFFs. The globe is cut into 15 x 15 degree lat/lon chunks (config `earth_export`, 288 chunks) whose edges are snapped to the global 6 km lattice by one function, so chunks neither overlap nor leave gaps and share one pixel grid (`crsTransform`). Task names are deterministic (`gsop_earth_<chunk id>`). A task ledger (`data/interim/ee_export_ledger.json`, written after every task start) records task ids and states: a restart polls tasks already in flight and never re-exports completed chunks. A failed chunk is retried up to `max_retries` (3) more times, then reported (exit code 1). `--dry-run` lists chunks and bands and starts nothing (no login needed). Groups that are unavailable (currently `water_index`, see above) are left out with a logged reason.

**Expected size** printed before the run is the uncompressed float32 upper bound: about 5.8k x 2.4k tiles x bands x 4 bytes, a few GB for all bands, before ocean and GeoTIFF compression. The plan's "roughly 12 bands well under 1 GB" holds for 12 compressed bands, not for the full list (about 25 bands incl. flags and diagnostics). Compare with the actual size after the first real export.

**Not yet run against Earth Engine.** The orchestration is tested with a fake task API (start, poll, retry, resume); the Earth Engine expressions are not (issue B17 is the first real run).

### Fallback route without Earth Engine (issue B21): LOCAL ROUTE, terrain is coarser

**Loud note: if Earth Engine is not available the Earth file can still be built from rasters you downloaded yourself, but terrain features are coarser or missing.** `gsop s2 earth --source local --local-dir DIR` reads `DIR/local_sources.yaml` (region and one raster per band, see the docstring of `src/gsop/earth/local_loaders.py`), resamples each onto the 6 km EPSG:6933 lattice, writes one chunk raster with the same band names an Earth Engine export has, and then runs the same parquet build, QA and manifest. Nothing is downloaded by this route.

Limits, stated plainly: (1) a global ~1.8 km relief grid cannot give slope at the **500 m baseline**, so `slope_med`, `slope_p90`, `rough_a` and `rough_b` become `unavailable` (all NaN; the missing-data gate then rejects every tile for targets whose weights use them, which is every target); (2) `relief` is built from the maximum and minimum of the native cells in the tile and **underestimates** relief when cells are coarser than 500 m; (3) a band that is not supplied is `unavailable`; (4) the local route currently has no tile operator wired in for fine DEMs even if supplied (it would reuse `operators.py` windows; not built); (5) the reasons are written to `local_report.json` and printed. In practice the local route cannot score terrain, which is 0.53 of the Moon score, so it is a documented fallback and not a substitute.

### The Earth file (issue B18)

`gsop s2 earth --build-from DIR` turns the downloaded chunk GeoTIFFs into `data/processed/earth_features_6km.parquet` and `grid.json` (`src/gsop/earth/build_parquet.py`). One row per **land** tile (land = at least `land_min_fraction`, 0.5, of its 500 m cells are land in MCD12Q1 `LW`; expect about 4 million rows, not yet measured). `tile_id = iy * nx + ix` on the global 6 km lattice (iy from the north edge), so ids do not depend on which chunks were exported. Nodata becomes NaN. Computed here, not in Earth Engine: `sun_max_elev`, `sun_low_share`, `insol_annual` from tile latitude (flat ground, Earth constants), `air_density` from `elevation_mean`, and `dust_index` (raw dust scaled with land-tile percentiles). Columns: `tile_id, lat, lon, x_m, y_m, land`, the 15 feature columns, `elevation_mean`, the three gate columns (`water_ice_fraction`, `ndvi_median`, `builtup_cropland_fraction`), flags `t_low_obs` and `ti_proxy_fallback`, and `<feature>_unavailable` (True where NaN). A feature that was not exported at all is all NaN and listed in `grid.json` under `unavailable_features`; a feature with more than `nan_warn_fraction` of land tiles missing is reported. Blocks are read in row slabs sized from the `gsop doctor` chunk budget. The build ends with a check that five random tiles equal their raster pixels.

### Gate layers (issue B15)

Three Earth-only bands are exported at 6 km and **no threshold is applied at export** (stage s3 does that with `config/gates.yaml`): `gate_water_ice_frac` (share of 500 m MCD12Q1 cells that are water bodies, class 17, or permanent snow and ice, class 15), `gate_ndvi_med` (tile median of the per-pixel 2015-2024 median NDVI of MOD13A2 with SummaryQA good or marginal), `gate_built_crop_frac` (share that is cropland 12, urban and built-up 13, or cropland/natural mosaic 14). Class numbers, the land-cover year (2023) and the QA values are in `config/features.yaml`, block `land_cover` (tunable, unsourced). Reference definitions: `src/gsop/gate_layers.py`. Class 14 is only partly farmed, so the farmland share may be overstated.

### `air_density` (issue B14, Mars family only)

One pure function (`src/gsop/features_misc.py`), no constants in code; all in `config/features.yaml` block `air_density` with sources. **Earth**: International Standard Atmosphere troposphere, `T = T0 - L h`, `p = p0 (T/T0)^(g/(R L))`, `rho = p / (R T)`, from the DEM elevation, valid to 11 km (tested against 1.225 kg/m3 at sea level and 0.3639 at 11,000 m, both printed on the ISA page). **Mars**: isothermal exponential, `rho = rho0 exp(-(h - h_ref)/H)` with `rho0 = p0 / (R_specific T)`; constants from the NASA Mars Fact Sheet (6.36 mb, scale height 11.0 km, 214 K, molecular weight 43.49), which reproduce the sheet's own surface density (about 0.0155 against 0.016 +/- 0.006 kg/m3). The Mars pressure datum is an open TODO (see `docs/UNVERIFIED.md`). The Moon has no air; it does not use this column.

### Earth `silica_index` (issue B13)

The weakest cross-instrument comparison with `water_index`. **Earth**: ASTER Quartz Index of Ninomiya et al. (2005, *Remote Sensing of Environment*, as quoted in arXiv 1107.3499), `QI = (e11 * e11) / (e10 * e12)`, with `eN` the emissivity in ASTER thermal-infrared band N from ASTER GED v003 (`emissivity_band10`..`12`, scale 0.001), averaged to the 6 km tile (`src/gsop/silica.py`, `gee_export.silica_bands`). **Direction: higher = more silica-rich.** Quartz-rich rock has emissivity minima near bands 10 and 12 and a higher value at band 11.

- **Moon**: the Diviner Christiansen feature. Its wavelength moves **shorter** as silica increases, so the lunar index must be a decreasing function of the Christiansen-feature wavelength to keep the same direction (higher = more silica-rich). Defined in issue B26.
- **Mars**: TES-derived silica/surface-composition index, with higher = more silica-rich. Defined in issue B29.

Limits: the published QI was defined for radiance-at-sensor data, here it is applied to emissivity, so values are not comparable with the paper's maps; QI is a quartz index correlated with bulk SiO2, not SiO2 content (the 2002 SiO2 content index was not retrieved); ASTER GED is 2000-2008, has gaps (see Known weaknesses), and EMIT is not used. Whether the three instruments' indices share a scale is **not established**: the similarity score treats them as comparable, which is the assumption to challenge first.

### Earth `dust_index` (issue B12)

Mars thermal-inertia family column (it shares the family's weight with `ti_proxy`). **Status: built with a caveat, not unavailable.** Earth Engine exports `dust_index_raw`, the annual-mean MERRA-2 dust extinction aerosol optical thickness at 550 nm (`NASA/GSFC/MERRA/aer/2`, band `DUEXTTAU`, noon-UTC value of each day, 2015-2024). At build time (issue B18) it is scaled to 0..1 as `clip((raw - lo) / (hi - lo), 0, 1)` with `lo`, `hi` the 1st and 99th percentile over land tiles, so no cap is invented (`src/gsop/dust.py`); higher = more dust. Caveats: MERRA-2 cells are 69375 x 55000 m, about ten times coarser than a tile, so neighbouring tiles share a value; and it measures atmospheric dust while the Mars column (TES) is surface albedo and dust cover, so the two are not the same quantity.

If the column is ever marked `unavailable`, the Mars `thermal_inertia` family uses `ti_proxy` **alone**: a family's score is the mean of its columns' similarities, so dropping a column renormalises inside the family and the family keeps its weight (`Config.family_columns(body, family, unavailable)`); a family with no usable column left is an error and the missing-data gate acts.

### Earth `water_index` (issue B11)

`water_index` is the Moon's `cold_dry` family and the Mars `water` family. On Earth it is a **cross-instrument proxy and one of the two weakest comparisons in GSOP**: `water_index = mean(permafrost probability, dryness)`, both 0 to 1 and higher = more ice-like. Dryness is `1 - clip(AI / ai_cap, 0, 1)` with `AI = P / PET` from TerraClimate annual sums (UNEP aridity index): **0 = fully humid** (AI at or above `ai_cap`, default 1), **1 = no precipitation**. Reference code: `src/gsop/water_index.py`. A tile missing either input gets NaN (never 0). Hot deserts get about 0.5 (dry, no permafrost); cold deserts with permafrost approach 1; tropical wet ground gets 0.

**Current status: unavailable.** No permafrost-probability layer is in the Earth Engine catalog (issue B04), so while `water_index.permafrost_asset` in `config/features.yaml` is null the Earth `water_index` is **not built**: the column is NaN everywhere and the missing-data gate rejects every tile for any target whose weights use it (all Moon and Mars targets use it). This blocks the runs until Anay decides: provide the Northern-Hemisphere permafrost map as an Earth Engine asset and choose `permafrost_outside_domain` (`nan` leaves the whole Southern Hemisphere, including McMurdo and the Atacama, without a value; `zero` scores it as no permafrost).

### Earth temperature: `t_mean`, `t_coldest` (issue B09)

From MODIS Terra LST `MODIS/061/MOD11A2` (1 km, 8-day), 2015-2024, config block `temperature`. Per composite the temperature is the **mean of day and night LST, kept only where both have QA 'good'** (QC bits 0-1 equal 0); a day-only value would be biased warm. `t_mean` is the mean of all valid composites; `t_coldest` is the lowest of the 12 calendar-month means (each averaged over all ten years), using only months with at least `min_valid_obs_per_month` valid composites. 1 km values are averaged to the 6 km tile. A tile with fewer than `min_valid_obs_total` valid composites gets `t_low_obs = 1` and **NaN** temperatures (never a fake number); both thresholds are tunable and unsourced. Known bias: cloudy seasons (monsoon, polar night) drop out of the composites, so the 'annual' mean over-represents clear seasons there.

### Earth thermal-inertia proxy: `ti_proxy` (issue B10)

`ti_proxy` is the **day-night surface-temperature swing (K, day LST minus night LST) in the dry season**: the three consecutive (cyclic) calendar months with the least precipitation (TerraClimate `pr`, 2015-2024), averaged over those months and years. It is a proxy for thermal inertia (high inertia gives a small swing), **not a measurement** of it. Reference implementation: `src/gsop/ti_proxy.py` (numpy, unit tested); the Earth Engine build in `gee_export.ti_proxy_bands` must match it.

**Fallback.** When no dry season can be defined (precipitation missing, or the driest three months are not clearly drier than average: driest-window precipitation is not below `dry_season_ratio_max` = 0.5 times the average three-month precipitation, or a dry month has no valid swing), `ti_proxy` is the **median swing over all valid composites** and `ti_proxy_fallback = 1`. A place with essentially no precipitation has a trivially defined dry season. The Moon and Mars have no dry season: their `ti_proxy` is defined per body in the target issues. **How many global tiles use the fallback is not known yet**: it must be counted after the global export and written here.

### Illumination: `sun_max_elev`, `sun_low_share`, `insol_annual` (issue B08)

One analytic numpy implementation (`src/gsop/illumination.py`) for Earth, Moon and Mars. Per-body inputs are the obliquity and solar irradiance in `config/features.yaml` (block `illumination`, each with a source; the Moon obliquity 1.54 degrees is not on the NASA fact sheet and is marked unverified). Over a year the Sun's declination `d` satisfies `sin d = sin(eps) sin(L)`; at latitude `phi` and hour angle `h`, `sin e = sin(phi) sin(d) + cos(phi) cos(d) cos(h)`.

- `sun_max_elev` (degrees) = `90 - max(|phi| - eps, 0)`, exact.
- `sun_low_share` (0-1) = share of all daylight (`e > 0`) over the year with `e` below 5 degrees, integrated analytically per day over 360 values of `L`.
- `insol_annual` (W/m2) = annual mean flux on the ground `S0 max(0, cos theta)`; flat ground uses the closed-form daily mean, a given slope and aspect use numeric quadrature (72 days x 96 hour angles).
- Earth tiles use flat ground (the tile table has no aspect). Moon and Mars target values come from the published Glaser 2018 numbers (issue B27), not from this code.

**What it cannot capture:** (1) **terrain shadowing**: an unobstructed flat horizon is assumed, but at the lunar poles neighbouring ridges and crater rims decide what is lit, so these three numbers say nothing about that (the `illumination_match` output and the fidelity score carry the consequence: Earth cannot match polar lighting); (2) orbital eccentricity: `L` is sampled uniformly, equal to uniform in time only for a circular orbit (worst for Mars, eccentricity about 0.09); (3) atmosphere: top-of-atmosphere geometry only.

## Operators (s2)

One terrain operator is used for slope, relief and roughness on Earth, Moon and Mars. It is implemented twice, in numpy (`src/gsop/operators.py`, issue B22) and in Earth Engine (`src/gsop/earth/gee_export.py`), and `gsop check-operators` compares the two. Settings are in `config/features.yaml`, block `operators`.

### Grid and tiles

Tiles are 6000 m x 6000 m cells of the equal-area grid EPSG:6933 (decision D1), aligned to the projection origin, so a tile boundary falls at a multiple of 6000 m. The terrain baseline is 500 m, and 6000 / 500 = 12 exactly, so a tile is a 12 x 12 block of 500 m cells. For Moon and Mars target windows the same tile is used with a stride of 3 km (half a tile), so tiles overlap.

### Step 1: DEM source (Earth only)

Inside the latitude range `nasadem_lat_range` (-56 to 60 degrees) the elevation is NASADEM (`NASA/NASADEM_HGT/001`, band `elevation`); outside it, Copernicus GLO-30 (`COPERNICUS/DEM/GLO30_2024_1`, band `DEM`). The switch is a **hard cut** at the configured latitudes with no blending. Deviation from the setup prompt: it says "NASADEM between 56S and 60N, Copernicus above 60N" and says nothing south of 56S, where SRTM does not exist (Antarctica, so McMurdo needs it); Copernicus is used there too. The vertical datums differ (EGM96 vs EGM2008, see `docs/PROVENANCE.md`); slope and relief use height differences, so a constant offset cancels. Issue B07 tests the seam.

### Step 2: resample to the 500 m baseline

Block mean: each 500 m cell is the mean of the native cells inside it (Earth Engine: `reduceResolution(mean)` then `reproject(EPSG:6933, scale=500)`; numpy: mean over `factor x factor` native cells, `factor = round(500 / native_cell_m)`, at least 1). If the native cell is coarser than 500 m (Mars MOLA is about 463 m), the factor is 1 and the **effective baseline is the native cell size**; it is reported, never hidden.

### Step 3: slope

Horn (1981) 3 x 3 finite difference on the 500 m grid. With neighbours

    a b c
    d e f
    g h i

    dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * dx)
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * dy)
    slope = atan( sqrt( dz/dx^2 + dz/dy^2 ) )   in degrees

`dx` and `dy` are the 500 m cell size in metres. Horn weights the nearer neighbours double and averages three finite differences, which makes it less sensitive to single-cell noise than a plain central difference; it is also the usual choice for DEM slope, so Moon and Mars tools agree with it. A cell with **any missing cell in its 3 x 3 neighbourhood has no slope** (masked in Earth Engine, NaN in numpy). The slope grid is computed over the whole DEM window first and cut into tiles afterwards, so cells on a tile border use their neighbours in the next tile (callers read windows with a one-cell halo).

### Step 4: reduce to the tile

`slope_med` is the median and `slope_p90` the 90th percentile of the slopes of the (up to 144) 500 m cells in the tile. The numpy percentile uses linear interpolation; Earth Engine's `percentile` reducer is histogram based, so small differences are expected (the 10% tolerance of `gsop check-operators` covers them). A tile with more than `max_nan_fraction` (0.2, tunable and unsourced) of its cells missing is NaN: a partial answer is never returned.

### numpy implementation and edge handling (issue B22)

`src/gsop/operators.py`: `baseline_factor`, `effective_baseline_m`, `block_mean`, `horn_slope_deg`, `plane_rms`, `tile_features(dem, cell_m, *, tile_m, baseline_m, rough_baselines_m, stride_m, max_nan_fraction)`. Inputs: a 2-D DEM (row 0 at the north), the **metric** cell size `cell_m` (square cells; give the metric size for geographic or polar grids, so LOLA, MOLA and Earth tiles all go through the same code), and values from `config/features.yaml`. Cell [0, 0] of the array must sit on a tile corner; tiles are `tile_m` squares `stride_m` apart (`tile_m` on the global grid, 3000 m for the Moon and Mars target windows). The result is one array per feature of shape `(tile rows, tile columns)` plus the baselines actually used.

Edge handling: (1) the outer border of the slope grid is NaN (no 3 x 3 neighbourhood); (2) a slope cell with any NaN neighbour is NaN; (3) a ragged edge that does not fill a block or a tile is dropped; (4) a tile with more than `max_nan_fraction` missing cells is NaN for that feature, so a tile at the array border keeps its slope only while the lost border cells stay under that share (callers read windows with a one-cell halo of baseline cells to avoid it); (5) roughness tiles without missing cells use the same closed form as Earth Engine; tiles with some missing cells (within the limit) use a least-squares plane on the valid cells.


`relief` is the maximum minus the minimum of the 500 m cells (block means, Step 2) inside the tile. Same missing-cell rule as slope.

### Roughness

`rough_a` and `rough_b` are the RMS height after plane removal at two baselines. The baselines are **config, not code** (`baseline_m` of each column in `config/features.yaml`): about 100 m and 1 km, defaults only (`TODO confirm with Vihaan`, listed in `docs/UNVERIFIED.md`).

1. Block-mean the DEM to the baseline B (Step 2 with B in place of 500 m). If the native cell is coarser than B the effective baseline is the native cell size (Mars MOLA, about 463 m, cannot give 100 m: `rough_a` on Mars is then at about 463 m. This is a real mismatch and is flagged, not hidden).
2. In each tile (n x n cells of size B, n = 6000 / B) fit a plane `z = a + b x + c y` by least squares and take the RMS of the residual: `rough = sqrt(mean((z - plane)^2))`.

Earth Engine does not run a regression. On a full square grid with centred coordinates `x`, `y` (cell index minus (n-1)/2) the constant, `x` and `y` are mutually orthogonal, so the fitted plane has a closed form and

    rough^2 = E[z^2] - E[z]^2 - E[x z]^2 / E[x^2] - E[y z]^2 / E[y^2],      E[x^2] = E[y^2] = (n^2 - 1) / 12

with every `E[.]` a tile mean (`reduceResolution(mean)` of `z`, `z^2`, `x z`, `y z`). This equals the least-squares residual exactly for a tile without missing cells (tested in `tests/test_gee_export.py`). For a tile with some missing cells it is an approximation: the numpy operator fits the plane by least squares on the valid cells only. The difference is expected to be small below the 20% missing-cell limit but has not been measured (`gsop check-operators`, issue B24, compares on one window).

## Gates (s3)

`src/gsop/gates.py`. Thresholds live in `config/gates.yaml` (never in code); the landing_mobility extra gate lives in `config/purposes.yaml`. A tile can fail several gates, so the result is a per-tile bitmask (bits: water_ice 1, vegetation 2, builtup_cropland 4, landing_mobility_slope 8, missing_data 16); reason codes are the stable strings in `gates.reason_codes` (`WATER_OR_ICE`, `NDVI_HIGH`, `BUILTUP_OR_CROPLAND`, `LANDING_SLOPE`, `MISSING_DATA`).

- `water_ice`: open water or permanent ice covers **more than** 50% of the tile. `vegetation`: median NDVI **above** 0.2. `builtup_cropland`: built-up or farmed land covers **more than** 20%. A value equal to the threshold passes.
- `landing_mobility_slope`: Moon `slope_med <= 8`, Mars `slope_p90 <= 30` (equal passes). It is evaluated for every tile, but **applied only by the landing_mobility profile**.
- `missing_data`: any feature column the target's weights use is NaN, **or a gate layer is NaN** (a tile whose water, vegetation or land use is unknown cannot be shown to be an analog candidate, so it is rejected, not waved through).
- **Near misses.** A tile that *passes* a gate within `near_miss_fraction` (10%) of its threshold gets the matching bit in a second mask, for the discovery cards (for the upper-limit gates: value above 90% of the threshold and not above it).
- **Mars target check (decision D6, `src/gsop/target_check.py`).** Applied to the target profile, not to tiles: latitude within +-50 degrees (`targets.yaml`), elevation below 2000 m (`elevation_mean`, MOLA above the areoid; equal fails), thermal inertia at least 100 (`thermal_inertia_themis`, THEMIS; equal passes), all thresholds from `gates.yaml`. In `warn` mode (default) a failing target is still scored and every output row carries `fails_own_gate: true` and `own_gate_reasons` (listed in `validation.json`); `--target-check strict` (mode `reject`) does not score a failing target. An input that is missing cannot pass: it fails with a reason `...:INPUT_MISSING_...`. Moon targets have no such check. Jezero passes with real values: elevation -2587 m above the areoid, thermal inertia 313, latitude 18.44 (2026-11-14); Arcadia is unscored (no data), so its check is not evaluated.
- Gating is vectorised and gives the same answer for any chunking of the rows (tested). The Mars *target* check is not a tile gate (it is applied to the target: `gsop.target_check`).

## Similarity (s4)

`src/gsop/similarity.py`. Per column `s_k = exp(-0.5 * ((x_tile - x_target) / sigma_k)^2)` in [0, 1]; a family's score `S_f` is the mean of its columns' `s_k`; tile score `100 * sum_f w_f * S_f`, in [0, 100].

- **Missing is missing.** A NaN column makes its family NaN and a NaN family makes the tile score NaN; it is never treated as 0. A family with weight 0 (alternative weight sets) is skipped so it cannot poison the score. The gate stage decides what to do with missing tiles.
- **No overflow.** The squared distance is clipped at 170 before the exponential, so a tile 13 or more sigma away scores about 1e-37 (a normal float32), never NaN or inf. Such tiles tie with each other, which does not matter for ranking.
- **Deterministic across machines.** float32 element-wise arithmetic with a fixed left-to-right order over columns and families: no BLAS matmul and no layout-dependent mean, so the score of a tile is bit-identical whatever the chunk size (the chunk size depends on each machine's RAM).
- **One function** (`score_tiles(x, target, sigma, family_cols, weights)`) serves a single scoring run and the Monte Carlo inner loop; it also returns the per-family similarity matrix (needed for `illumination_match` and for "which families drive the match").
- **Target set-up (`make_setup`).** The columns are those the body's weights use. A column the target has no value for (an `unavailable` row) is dropped from its family, so the family mean uses the remaining columns (the family keeps its weight). A family left with no column follows `unavailable_target_families` in `config/run.yaml`: `error` (default; stops and names the family) or `drop_and_renormalise` (leaves the family out and renormalises the rest; this departs from the published weights and is for Anay to decide).
- **Stage s4 (`gsop s4 score`, `src/gsop/stages/s4.py`).** For each scored target it takes the tiles that passed the base gates (s3), computes sigma over exactly those tiles, then the family similarities once and a score for every profile from the profile's weights (base weights renormalised over the families kept, multiplied by the profile multipliers, renormalised). Tiles are sorted by `tile_id` first, so outputs and later tie-breaking never depend on input row order. The landing_mobility score is NaN where its extra gate fails. Output files: see `docs/RESULTS_SCHEMA.md`.
- **Throughput** (measured 2026-11-14 on the Apple M2 with `scripts/bench_similarity.py`, one core, quiet process): about 17 million tiles/s for the Moon (12 columns, 8 families) and about 14 million tiles/s for Mars (14 columns, 9 families). One scoring pass over 4 million tiles is therefore about 0.3 s per core.

## Sigma rule (s4)

`compute_sigma` in `src/gsop/similarity.py`: `sigma_k = max(target_spread_k, measurement_uncertainty_k, m * IQR_k)` with `m = sigma_iqr_multiplier` (0.25, `config/run.yaml`) and `IQR_k` the interquartile range of column `k` over the finite values of the **gated** Earth tiles of that target. The IQR term stops the score collapsing to 0 everywhere for variables Earth cannot match (polar lighting): without it a tile 20 target-spreads away on illumination would score 0 for every tile on Earth.

- **`target_spread`**: IQR / 1.349 of the target's window tiles (the target profile).
- **`measurement_uncertainty`**: the larger of the sourced target-side and sourced Earth-side floors in `config/features.yaml` (the `max` of each range, the conservative end; today only Moon slope, relief, roughness from Barker 2021, Mars relief and roughness from MOLA, and Earth `t_mean` and `t_coldest` from the MODIS QC classes). A `TODO` contributes 0; if both sides are `TODO` the value is 0 and one warning per column and body is logged, and the IQR term carries the sigma.
- **Guard**: if all three terms are 0 (a constant column) sigma is 1e-9 and a warning is logged, so the similarity never divides by zero.
- `sigma_table.json` (written by s4) lists, for every target and column, the sigma and the three terms with the winner, so the choice is visible.

## Weights (s4)

`src/gsop/weights.py`. **Derivation:** `weight_f = (paper_score_f / sum(paper_score)) * matchability_f`, rescaled so the weights sum to 1 (no weight is typed anywhere in code; the inputs are `paper_score` and `matchability` in `config/weights.yaml`). Result, pinned by tests to 4 decimals: Moon slope 0.3042, illumination 0.1806, cold_dry 0.1483, roughness 0.1198, relief 0.1141, temperature 0.0608, composition 0.0532, thermal_inertia 0.0190; Mars composition 0.2289, water 0.1937, sunlight 0.1497, slope 0.1132, thermal_inertia 0.0906, roughness 0.0792, relief 0.0755, elevation_density 0.0377, temperature 0.0314. Each is within 0.012 of the published number.

**Decision D3.** The Moon `thermal_inertia` derived weight 0.019 drives the run; the published 0.03 is a rounding-up artifact so the published column sums to 1. 0.019 sits just *below* its own Monte Carlo range [0.02, 0.04]: the range is the sampling range for the Monte Carlo, the derived weight is the base score, and the one case is explicitly excluded from the range test.

**Column weights.** A family's weight is split equally over its columns because the family score is the mean of its columns' similarities.

**Alternative weight sets (`alt_weight_sets`, issue C06).** Each sums to 1 and is run once (no Monte Carlo) as a sensitivity check. `liu2025` (Moon: roughness 0.39, slope 0.26, cold_dry 0.16, illumination 0.11, composition 0.08) and `mashal2025` (Mars: composition 0.75, water 0.125, sunlight 0.125) do not name every family: **families not named get weight 0**, and the set is then renormalised only if the listed numbers do not already sum to 1 (they do, so nothing is rescaled). That means Liu's set zeroes relief, temperature and thermal_inertia, and Mashal's zeroes six of the nine Mars families. `no_illumination` (Moon, decision D4) is the primary with `illumination` set to 0 and the rest renormalised, so ratios among the other families are unchanged. `equal` gives every family 1/n. `no_illumination` is refused for Mars.

## Purposes (s4)

TODO (phase C). Multiply base weights by the purpose multipliers, then renormalise to sum 1 (`config/purposes.yaml`).

## Monte Carlo (s5)

`src/gsop/montecarlo.py`. 2,000 runs, master seed 20261114 (D2); the Monte Carlo jitters weights and data (D8).

**Seeds.** Run `i` uses `SeedSequence(master_seed, spawn_key=(i,))`, which equals child `i` of `SeedSequence(master_seed).spawn(n)` for any `n` (tested), so a run never depends on the worker count or on how many runs there are. The run's two children, `spawn_key=(i, 0)` (weight stream) and `(i, 1)` (data stream), never overlap. Every function that depends on `i` is pure (no global random state): any worker can reproduce any run from the set-up and the index.

**Weights, per run (issue C15).** Draw each family weight uniformly within its `range` in `config/weights.yaml`; renormalise to sum 1; multiply by the purpose multipliers; renormalise again. (Because renormalising is scale-invariant the first renormalisation does not change the final vector; the order is kept because it states what the weights mean at each step.) **Data jitter, per run (issue C16, decision D8).** From the run's data stream, two independent normals per column: (1) the **target value** of each column is perturbed by N(0, target measurement uncertainty), taken as the `max` of the sourced target-side range in `config/features.yaml` (today Moon slope 2.5 degrees and height 0.5 m from Barker 2021, Mars height 3 m from MOLA); (2) a **systematic Earth bias** per column, N(0, `earth_bias`). Since (x + b - t') / sigma = (x - (t' - b)) / sigma, both terms are one shift of the target value per column (`shift = jitter - bias`), so the jitter costs nothing per tile. Every column whose uncertainty or bias is `TODO` gets sd 0 (no jitter) and one warning per column; **today all fifteen `earth_bias` entries are `TODO`, so only the target-side terms with a source actually jitter**. With zero uncertainty the shift is exactly 0 (tested). *Not jittered:* sigma (fixed per column), the weight ranges, and per-tile noise: the issue's optional per-tile noise for candidate tiles is not implemented because it would make pass 2 scores differ from pass 1 scores for the same run (`CANDIDATE_TILE_NOISE_SD = 0`).

**Streaming core (issue C17).** Scoring a run for all tiles is `gsop.similarity.score_tiles` in chunks, so peak extra memory is one chunk, not the tile block. Per-tile statistics over runs are kept online (`OnlineStats`: count, mean, M2 for the variance, min, max; Welford update, Chan merge in a fixed order) instead of storing an N x 2000 matrix. `GSOP_MC_DEBUG=1` turns on an assertion that refuses any buffer that looks like (tiles x runs) and prints its size. The score of a run depends only on the run index (tested), so blocks of runs can be computed anywhere and merged.

**Top-set membership (issue C18).** For every run the top 1%, 5% and 10% of the gated tiles (`ceil(fraction * N)` tiles each, nested) get +1 in a uint16 counter per tile (2,000 runs cannot overflow it; the limit is 65,535). The cut is found in O(N) with `np.partition`, and only the top 10% is then sorted. **Tie rule:** tiles are ordered by score descending and, among equal scores, by lower index (tile order is `tile_id` ascending), so the sets never depend on the algorithm's internal order. NaN scores are refused (a NaN would sort as the largest); missing tiles must be gated out first. Counters are kept per **segment** between convergence checkpoints so that cumulative counts at 500, 1,000, 1,500 and 2,000 runs are exact.

**Pass 2 (issue C19).** After pass 1, **candidate tiles** are the union of the top 3% (`candidate_fraction`) of tiles by mean score and the top 3% by membership count of the top-5% set. For each run the candidates are scored again with the same seeds and code (the stored pass 2 score is bit-identical to the pass 1 score, tested) and the run's scores are sorted once; the candidate's **rank percentile** in that run is `100 * (tiles with a strictly lower score) / N` (`np.searchsorted`, stored in hundredths of a point). Per-run values are written to disk-backed `.npy` arrays (runs x candidates, float32 for scores, uint16 for ranks) opened as memmaps, because 2,000 runs x 120,000 candidates would not fit in RAM; the percentiles are then exact numpy linear percentiles over the runs: `mc_median`, `mc_p05`, `mc_p95` of the score and `rank_pct_p05`, `rank_pct_p50`, `rank_pct_p95`. A finished block has a marker file and is skipped on resume. The cost is one more scoring pass plus one sort of N scores per run.

**Outputs (issue C20).** `robust` = the tile was in the top 5% in at least 90% of the runs (`robust.top_fraction`, `robust.min_run_share` in `config/run.yaml`). The results table lists every tile that ever reached the top 10% or is a candidate (tiles not listed never entered the top 10% in any run): `mc_mean`, `mc_std`, `freq_top1`, `freq_top5`, `freq_top10`, `robust`, `is_candidate`, and for candidates `mc_median`, `mc_p05`, `mc_p95`, `rank_pct_p05`, `rank_pct_p50`, `rank_pct_p95`. **Convergence:** the top-1% set (tiles ranked by membership count of the top-1%, then 5%, then 10% set, then lower index) after the first 500, 1,000, 1,500 and 2,000 runs is compared with the final set by Jaccard overlap; for a different `--runs` the checkpoints scale in proportion. **Blocks:** the runs are cut into blocks of 125 (split at every checkpoint) that depend only on `runs`, never on the number of workers, and merged in run order, so the merged statistics are the same for any worker count.

**Parallelism and resume (issue C21).** Blocks of runs are the unit of work and of saving. Workers are spawned processes (the same start method on Windows and macOS) that attach the feature matrix from shared memory (`gsop.parallel`), never a pickled copy; `iter_parallel` yields block results in order with a bounded number in flight, and the parent merges and discards each. Seeds belong to runs, not workers, and the blocks and the merge order do not depend on the worker count, so **1, 4 or 8 workers give exactly the same table** (tested with a full frame comparison: counters and floating-point statistics alike). With a work directory every finished block is written at once; after a crash or Ctrl-C a restart runs only the missing blocks (pass 1: one file per block; pass 2: one marker per block, rows written straight into the memmap). Saved partial results carry a fingerprint (settings, set-up arrays, a sample of the features) and are discarded if it does not match. Progress is a `tqdm` bar on a terminal and a log line per profile otherwise. Temporary files are removed after success.

**Stage s5 (`gsop s5 simulate --body --runs N [--workers W]`, issue C22).** For every scored target and every profile: gated tiles from `gates_passed.parquet` (the landing_mobility profile only those whose extra gate passed), features and **the sigma written by s4**, then `simulate`, then `mc_<target>_<profile>.parquet` with the target, profile, confidence and own-gate columns added. A finished (target, profile) writes a marker with a fingerprint of its inputs and is skipped on rerun; inside a profile, interrupted runs resume at the block level. `--runs` is part of the stage hash (a different run count reruns the stage) and of every marker. Chunk size for scoring is bounded by the free-RAM budget of `gsop doctor`; scores do not depend on it.

The range, not the derived weight, is what is sampled: the Moon `thermal_inertia` range is 0.02 to 0.04 although its derived weight is 0.019.

## ML (s6)

TODO (phase D). Second opinion only (D9); it never produces the score.

## Validation (s7)

TODO (phase D). Pass criteria are pre-registered in `config/known_sites.yaml` before anything is computed. Known-site coordinates are unconfirmed until issue D01 is closed.

## Export (s8)

TODO (phase E). See `docs/RESULTS_SCHEMA.md`. Required by issue C28: the exported scores must keep the `illumination_match` column (the illumination family's similarity alone; Earth cannot match polar lighting, so it is shown beside the total and never used to rank).

## Discovery (s9)

TODO (phase E). Robust tiles more than 100 km from every known site. Each candidate card shows `illumination_match` beside the score: it is already a column of every `mc_<target>_<profile>.parquet` row (issue C28).

## Target profiles (s2 targets)

### Moon terrain (issue B25)

`src/gsop/targets/moon.py`. Source: the Barker et al. (2021) 5 m LOLA site DEMs on NASA PGDA (`Site01` = Connecting Ridge, `Site04` = Shackleton rim). A 15 km window centred on the target (shifted, not shrunk, to stay inside the 16 km tile; Shackleton rim sits 1.9 km off its tile centre) is read with a windowed `rasterio` read, never the full-planet DEM. The one shared operator (`operators.tile_features`) gives `slope_med`, `slope_p90`, `relief`, `rough_a`, `rough_b` on 6 km tiles at 3 km stride inside the window (16 overlapping tiles, so they are not independent). **Value = median over tiles, spread = IQR / 1.349.** Every row carries `source`, `evidence_grade` (A) and `window`. The window is read without a halo, so border tiles lose their outermost slope cells (under the 20% missing-cell limit).

First real values (2026-11-14, 5 m LOLA through the shared operator): Connecting Ridge `slope_med` 8.75 deg (spread 1.2), `slope_p90` 17.7, `relief` 1045 m, `rough_a` 142 m, `rough_b` 130 m; Shackleton rim `slope_med` 23.3 deg (spread 13.2), `slope_p90` 31.6, `relief` 2404 m (spread 1412), `rough_a` 184 m, `rough_b` 169 m.

**Two consequences to read before trusting any Moon result.** (1) Connecting Ridge's own median tile slope (8.75 deg) is already above the Moon landing-mobility extra gate (`slope_med <= 8`), so the target would fail the gate that applies to the candidates (decision D6 only covers the Mars target check). (2) For every tile `rough_b <= rough_a` is the normal order: both are RMS over the same 6 km tile and the coarser baseline averages away short wavelengths (tested in `tests/test_operators.py`). The issue text (B06, B20) expects `rough_b >= rough_a`; that expectation contradicts the operator definition and the Earth QA check was changed accordingly.

### The target file (issue B32)

`gsop s2 targets --body moon|mars [--diviner-dir DIR] [--out-dir DIR]` (`src/gsop/targets/__init__.py`) assembles every per-body row into `data/processed/targets_{body}.parquet` and a readable `targets_{body}.md`. Columns: `target, column, value, spread, source, evidence_grade (A/B/C), window, unavailable, unscored, note`. Rules enforced before anything is written: every column the body's weights use has a row for every scored target (a value, or an explicit `unavailable` row with `value = NaN` and the reason in `note`); an available row has a finite value and a non-negative spread; no row without a source. Arcadia (no coordinates, D10) is one row with `column = "*"` and `unscored = True`. Malapert is not a target. Rows for unscored columns are kept (for example `thermal_inertia_themis`, the input of the D6 check).

**State on 2026-11-14 (real run):** Moon 24 rows (8 unavailable), Mars 16 rows (6 unavailable). Families with no usable target column, which the missing-data gate would reject on: **Moon** `temperature` (0.06), `composition` (0.05), `thermal_inertia` (0.019, derived weight); **Mars** `composition` (0.23), `thermal_inertia` (0.09), `temperature` (0.03). That is roughly 13% of the Moon weight and 35% of the Mars weight with no target value; the file's summary lists them. How to proceed (drop and renormalise, or supply the missing products) is a decision for Anay and is not made here. The Diviner part (Moon temperature) needs 144 downloaded files passed through `--diviner-dir`.

### Mars temperature, `air_density` and insolation (issue B31)

`mars_climate_rows` in `src/gsop/targets/mars.py`. **Built (grade B):** `air_density` from the shared exponential-law function (`gsop.features_misc`, constants from the NASA Mars Fact Sheet in `features.yaml`) at the mean MOLA elevation of each window tile, median over tiles; `insol_annual` (W/m2, annual mean on flat ground) and `sun_max_elev` from the shared analytic module (`gsop.illumination`, Mars obliquity 25.19 and irradiance 586.2 from `features.yaml`) at the latitude of each tile centre. Jezero (18.4 N) is inside the tropics of an obliquity-25.19 planet, so `sun_max_elev` is 90 degrees. No physical constant is in the code. **Not built, `unavailable` with the reason:** `t_mean` and `t_coldest`: one search found no gridded annual-mean or coldest-month surface-temperature file (only single-time TES snapshot images and thermal-inertia maps); the Mars Climate Database is a model with its own access route that was not verified or used. Consequence: the Mars `temperature` family (0.03) has no target value and, like the thermal-inertia and composition families (see B29), is rejected by the missing-data gate until someone supplies a verified source.

### Mars `water_index` composite and Arcadia (issue B30, decisions D7 and D10)

`swim_water_index_rows` in `src/gsop/targets/mars.py`. **Definition (ONE composite, D7):** `water_index = (C + 1) / 2`, where `C` is the SWIM 1.0 combined ice consistency (range -1, inconsistent with ice, to +1, consistent), median over the 16 window tiles, spread IQR / 1.349, grade B. The SWIM combined map is itself the "SWIM equation" combination of five maps: neutron (water-equivalent hydrogen, including its depth to a lower layer), thermal (TES and THEMIS thermal inertia), radar surface power, geomorphology and radar dielectric, over a sensing depth of about 5 m. **Ice depth, ice consistency and thermal inertia are therefore already inside this one number and are not added again**, which is what D7 forbids; the separate SWIM depth map has only 100,180 valid pixels and no stated unit, so it could not stand alone anyway. A cross-instrument proxy like the Earth `water_index`, with a different construction (Earth: mean of permafrost probability and dryness), so equal numbers do not mean equal ice. The product pixel is 3 km, so a 6 km tile holds 2 x 2 pixels.

**Jezero: `water_index` 0.288 (spread 0.035)** (2026-11-14; combined consistency about -0.42, low ice consistency, as expected at 18 N).

**Arcadia stays `unscored: no target data` (D10).** The SWIM pilot-study raster for Arcadia covers a swath from 172 W to 152 W and 0 to 60 N (about 20 degrees of longitude by 60 of latitude): the extent of a study area, not the position of a site, and no SWIM file contains a site centre. Its midpoint (30 N, 162 W) would be an arbitrary number presented as data, so it is **not** written to `config/targets.yaml` (`swim_arcadia_footprint()` returns the footprint as information only). A person must pick and justify an Arcadia centre.

### Mars thermal inertia, dust and silica (issue B29)

`mars_thermal_rows` in `src/gsop/targets/mars.py`. **Built:** THEMIS thermal inertia at Jezero from the 100 m quantitative mosaic (tile `00N060E`), median over the 16 window tiles: **313 J m-2 K-1 s-1/2, spread 15.7** (2026-11-14), stated absolute accuracy about 20%. It is stored as `thermal_inertia_themis` and is the input of the D6 Mars target check (`min_thermal_inertia` 100: Jezero passes). **Not built, each `unavailable` with its reason:** `ti_proxy`, `dust_index`, `silica_index`.

**This exposes a design problem the issue text did not anticipate.** The Earth `ti_proxy` is a day-night temperature swing in kelvin, where high thermal inertia means a *small* swing; THEMIS gives thermal inertia itself, where high inertia means a *large* number. Direction and units differ and nothing converts one into the other without a thermal model, so putting the THEMIS value in the `ti_proxy` column would break the rule that direction conventions are identical to Earth. The same applies to `dust_index` (Earth: MERRA-2 atmospheric dust optical thickness scaled with Earth percentiles; Mars: no TES dust cover index was found, only an 8 pixel/degree albedo map) and `silica_index` (no TES silica product found). **Consequence: the Mars `thermal_inertia` family (weight 0.09) and the `composition` family (0.23, the largest Mars weight) currently have no usable target value**, and under the missing-data gate every Mars target would be rejected for any weights that use them. This is a decision for Anay (options: define a Mars-side swing from a thermal model; find and verify a TES silica product; or drop the families for Mars and renormalise, which changes decision-level weights).

### Mars terrain at Jezero (issue B28)

`src/gsop/targets/mars.py`. Source: the USGS MOLA 463 m DEM mosaic, read by window over HTTP (never the 2 GB file). Because the file is equirectangular its pixels are 463.09 m north-south but only 463.09 x cos(latitude) east-west (439 m at Jezero), so the Horn slope cannot be applied to it as if cells were square. The 15 km window is therefore resampled to a **local Lambert azimuthal equal-area grid centred on the target at exactly 500 m** (bilinear interpolation at the cell centres using exact pyproj coordinates; area-averaging 463 m pixels into 500 m cells and GDAL's approximate warp transformer both added several percent of slope noise in a tilted-plane test), and then the same `operators.tile_features` runs as for Moon and Earth: 6 km tiles, 3 km stride, 16 tiles; value = median, spread = IQR / 1.349; grade A. The test (`tests/test_mars_terrain.py`) builds a synthetic MOLA tile with a plane tilted 3 and 6 degrees in true metres and requires 1% agreement, which a naive square-cell reading fails.

First real values (2026-11-14): Jezero `slope_med` 0.85 deg (spread 0.37), `slope_p90` 1.74, `relief` 113 m (spread 64), `rough_a` 10.1 m, `rough_b` 9.4 m.

**Comparability limits, stated plainly.** (1) MOLA is itself a 463 m product whose track gaps of 1 to 2 km are interpolated, so Mars slope, relief and roughness at the 500 m baseline contain less fine-scale information than the 5 m LOLA (Moon) and 30 m (Earth) data block-averaged to the same baseline: the same operator does not make the three inputs equally detailed. (2) `rough_a` (100 m baseline) cannot be met on Mars: its effective baseline is 500 m, written in each row's note. (3) The cross-machine check (Apple M1 and Windows give the same numbers) has not been done; every row carries the first 16 hex characters of the window's sha256 so the two machines can compare.

### Moon illumination targets and PSR `water_index` (issue B27)

**Mapping rule for `sun_max_elev` and `sun_low_share`.** These two columns describe the Sun's geometry in the sky (highest elevation over the year; share of daylight below 5 degrees). They are computed with the shared analytic module (`gsop.illumination`, Moon obliquity from `features.yaml`) at the centres of the 16 window tiles and summarised as median and IQR / 1.349 (`moon_illumination_rows`). Results: about 2 degrees for `sun_max_elev` and exactly 1.0 for `sun_low_share` at both targets, consistent with the setup prompt's "Sun within about 1.5 deg of the horizon". **Glaser et al. (2018) numbers (`lit_surface_pct`, `lit_2m_pct`, shadow hours, nearest PSR distance, receiver height 2 m) are not converted into either column**: they measure how much of the time terrain leaves the site lit, a different quantity; they are carried in each row's note for comparison and enter the later `illumination_match` and fidelity outputs, not these two columns. Grade B (computed, geometry only, no terrain shadowing).

**PSR `water_index` (Moon) = share of the 6 km tile in permanent shadow**, from the LOLA permanent-shadow map `LPSR_85S_060M_201608.TIF` (NASA PGDA, from Mazarico et al. 2011; 60 m, 85 to 90 S, int16 with scale 2.5e-05 and offset 0.5, nodata -32768), median over the window's tiles, IQR / 1.349, grade B (modelled from LOLA topography over an 18.6-year cycle, not measured shadow). **Encoding verified from data, not from the documentation (which does not say): scaled values are exactly 0.0 or 1.0 and 1.0 = permanent shadow**, because the PGDA average-illumination map `AVGVISIB_85S_060M_201608.TIF` is exactly 0.0000 on every pixel where the PSR map reads 1.0 (and 0.28 and 0.23 elsewhere in the two windows). First real values (2026-11-14): `water_index` 0.042 (spread 0.018) at Connecting Ridge and 0.131 (spread 0.245) at Shackleton rim; `sun_max_elev` 2.11 and 1.80 degrees; `sun_low_share` 1.0 at both. The `_COG.TIF` links listed on the PGDA page return an HTML page, not a file; the plain `.TIF` works (BigTIFF, 656 KB).

### Moon temperature, `ti_proxy`, `silica_index` (issue B26)

`src/gsop/targets/diviner.py`. **Correction to the issue's premise:** Diviner does cover the pole: the Polar Cumulative Product has bolometric brightness temperatures at 240 m for 80 to 90 S. The obstacle is size and format: the south polar seasonal set is **144 ASCII files of about 214 MB (30.7 GB)**, not readable by window, so **no Diviner value exists yet and every Moon temperature row is `unavailable` with that reason**. The module computes the values from files a person has downloaded to a folder (it refuses an incomplete set of fewer than 144 files, which would bias the means) and is tested on synthetic tables written in the real layout.

Definitions used instead of the Earth ones: `t_mean` = mean of the 6 Ls-bin means (each the mean over 24 subsolar-longitude files); `t_coldest` = the lowest Ls-bin mean (**about two months, not the coldest calendar month used on Earth**: a mismatch); `ti_proxy` = all-year swing of the local-time curve (1 h bins, max minus min), which replaces the Earth dry-season swing because the Moon has no rainy and dry seasons (a proxy, not thermal inertia). Bolometric brightness temperature is not kinetic temperature. Evidence grades: `t_mean`, `t_coldest` B (measured, different quantity from Earth LST), `ti_proxy` C. `silica_index` (Christiansen feature) is `unavailable` for the Moon targets: the polar product is bolometric only and the Christiansen feature needs narrow-band channels under suitable illumination, which the low polar Sun does not give.

### Memory (issue C25)

`simulate` never holds more than one block result in flight: block statistics are merged into the running totals as they arrive and then dropped.
(The first version kept every block result until the end, about 1.9 GB at 2,000 runs on 4,000,000 tiles; found by measuring and fixed.)

Measured on an Apple M2, 4,000,000 tiles x 12 columns, 5 runs, 1 worker: feature matrix 192 MB, peak resident memory added by the run 866 MB, 4 s.
Python allocations were 452 MB of that; the rest is allocator retention. That run used random data, which puts about 1.35 million tiles in the results table, so it is a worst case for table size.
`estimate_peak_bytes` is that measurement rounded up 25% and scaled by tiles, columns, checkpoints and extra workers (each extra worker adds its own statistics and counters).
`gsop s5 simulate` compares the estimate with the free RAM and the `gsop doctor` chunk budget before every profile: not enough free RAM stops the run with `MemoryBudgetError`; only the budget being exceeded logs a warning.

The slow test `tests/test_memory.py` runs the real 4,000,000-tile case and asserts the measured peak against the estimate and the budget. **It is skipped when the doctor budget is below the estimate**,
as on the 8 GB development laptop with about 1.8 GB free; so the literal assertion has not been checked since the estimate was recalibrated. Run it with `--runslow` on a machine with more free RAM.
The estimate for 8 workers is untested; worker count is the one knob that raises memory.

## Ledger and fidelity

Implemented in `src/gsop/ledger.py` (issue C27); `gsop ledger --body moon|mars` writes `results/<body>/fidelity.json`.

The limits ledger `config/ledger.yaml` is reported beside every score and **never multiplied into it**. `fidelity = 100 * mean(status_value over the ledger rows relevant to the purpose)`, Broken 0, Partly 0.5, Matchable 1.
The relevant rows are the ones the relevance matrix lists under the purpose. Hand-worked values (base 25, landing_mobility 16.7, power_night 25, ice_resources 33.3, geology_life 40) are asserted in `tests/test_ledger.py`.

`fidelity.json` has one entry per target x purpose (the rows behind the number, each with its reason, status value and provisional flag) and a table of all 12 ledger rows for the site.
`relevance_provisional` is true until Anay and Vihaan review the matrix; two rows (`electrostatic_dust`, `regolith_mechanics`) are provisional because they come from an unverified report; every entry carries both flags.

Guard: only `cli.py` imports `gsop.ledger`, and no scoring stage lists `ledger` among its `config_parts`, so editing the ledger cannot change a score or force a re-run of s3/s4/s5 (tests assert both).

Limits: the ledger has no body or target column, so **fidelity depends on the purpose only**: both Moon targets, and Moon and Mars, get the same number for the same purpose name, and rows that do not apply to a body (the Moon has no perchlorate) still count.
Making it body- or target-specific needs new sourced ledger rows; none were invented.

## Golden slice (issue C29)

`gsop golden` runs s3, s4 and s5 on a fixed slice of 5,000 tiles per body (50 runs, master seed from `config/run.yaml`, 1 worker), writes `results/golden.json` and compares it with the committed `tests/golden/golden.json`;
it exits 1 and prints each difference if they disagree. `gsop golden --update` rewrites the committed expectation (do this only after a deliberate change, and commit it with the change).
Takes about one minute on an Apple M2.

* Inputs: `tests/golden/<body>/earth_features_6km.parquet` and `targets_<body>.parquet` (about 600 kB each), made by `scripts/make_golden.py` from the synthetic generator of the tests (planted analog in Gaussian background, fixed seed). They are hashed byte for byte (sha256 in `golden.json`); `.gitattributes` marks parquet as binary so Windows never rewrites line endings. The whole folder is 2.3 MB (limit 5 MB).
* `golden.json` keeps: input hashes, config hash, run count, seed, library versions (informational), and per body and target the number of gated tiles, the `score_<profile>` of every gated tile, the top-1% set of each profile, the Monte Carlo `mc_mean` and frequencies of the tiles that reached the top 10%, and the top-1% set of the Monte Carlo ranking. Floats are rounded to 6 decimals.
* Outputs are not required to be bit-identical (a different BLAS or CPU may move the last bits of a float32); inputs are compared exactly and outputs with tolerance (see the tolerance paragraph under Golden test).
* **What it does not prove:** the inputs are synthetic, so a pass means the code gives the same answer on two machines (and over time), not that the real data or the real scores are right. Changing any file in `config/` that the scoring uses changes `config_hash` and needs `gsop golden --update`.

## Known weaknesses

TODO (grows as results arrive). Start list:

- `water_index` and `silica_index` compare different instruments on different bodies.
- Earth cannot match polar lighting, gravity, air pressure or cold traps (see the ledger).
- Match-ability values (the 0 to 1 column in `config/weights.yaml`) are expert judgement, not measured, and are the weakest link in the weights.
- Known-site coordinates are approximate and unconfirmed.
- **DEM seam (56 S / 60 N): NOT YET MEASURED.** Earth elevation switches from NASADEM to Copernicus GLO-30 at a hard cut. The test (`scripts/seam_test.py`, `src/gsop/earth/seam.py`) is built and checked on synthetic strips, but needs an Earth Engine login to run on Scandinavia and Patagonia. Until it runs, the verdict is neither `seam OK` nor `seam handled by <rule>`: treat any terrain feature near 60 N and 56 S as suspect.
- **ASTER GED coverage limits (`silica_index`).** The Earth silica proxy comes from ASTER GED v003, built from clear-sky ASTER data of **2000-2008**, not the 2015-2024 window used for MODIS. Its catalog bounding box is -59 to 80 degrees latitude and the page states no no-data value, so high-latitude, persistently cloudy, icy and open-water tiles may be missing. A tile with no silica value is `unavailable` and the missing-data gate rejects it for any target whose weights use `silica_index`. EMIT (2022 onwards, surface reflectance only, no mineral product in the Earth Engine collection) is not a substitute.
