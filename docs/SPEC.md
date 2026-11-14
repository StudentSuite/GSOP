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

TODO (phase B). Fixed by decision D1: 6 km x 6 km tiles on the equal-area grid EPSG:6933 (6000 m cells), land only, polar ice gated out.

## Features (s2)

TODO (phases B and C). Columns, sources and uncertainty floors are in `config/features.yaml`. `water_index` and `silica_index` are cross-instrument proxies and are the two weakest comparisons in the index.

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

### Relief

`relief` is the maximum minus the minimum of the 500 m cells (block means, Step 2) inside the tile. Same missing-cell rule as slope.

### Roughness

`rough_a` and `rough_b` are the RMS height after plane removal at two baselines. The baselines are **config, not code** (`baseline_m` of each column in `config/features.yaml`): about 100 m and 1 km, defaults only (`TODO confirm with Vihaan`, listed in `docs/UNVERIFIED.md`).

1. Block-mean the DEM to the baseline B (Step 2 with B in place of 500 m). If the native cell is coarser than B the effective baseline is the native cell size (Mars MOLA, about 463 m, cannot give 100 m: `rough_a` on Mars is then at about 463 m. This is a real mismatch and is flagged, not hidden).
2. In each tile (n x n cells of size B, n = 6000 / B) fit a plane `z = a + b x + c y` by least squares and take the RMS of the residual: `rough = sqrt(mean((z - plane)^2))`.

Earth Engine does not run a regression. On a full square grid with centred coordinates `x`, `y` (cell index minus (n-1)/2) the constant, `x` and `y` are mutually orthogonal, so the fitted plane has a closed form and

    rough^2 = E[z^2] - E[z]^2 - E[x z]^2 / E[x^2] - E[y z]^2 / E[y^2],      E[x^2] = E[y^2] = (n^2 - 1) / 12

with every `E[.]` a tile mean (`reduceResolution(mean)` of `z`, `z^2`, `x z`, `y z`). This equals the least-squares residual exactly for a tile without missing cells (tested in `tests/test_gee_export.py`). For a tile with some missing cells it is an approximation: the numpy operator fits the plane by least squares on the valid cells only. The difference is expected to be small below the 20% missing-cell limit but has not been measured (`gsop check-operators`, issue B24, compares on one window).

## Gates (s3)

TODO (phase C). Thresholds are in `config/gates.yaml`; the landing_mobility extra gate is in `config/purposes.yaml`. Every rejection logs a stable reason code per tile.

## Similarity (s4)

TODO (phase C). Per column `s_k = exp(-0.5 * ((x_tile - x_target) / sigma_k)^2)`; a family's score is the mean of its columns' similarities; tile score `100 * sum_f w_f * S_f`.

## Sigma rule (s4)

TODO (phase C). `sigma_k = max(target_spread_k, measurement_uncertainty_k, 0.25 * IQR_of_k_across_all_gated_Earth_tiles)`.

## Weights (s4)

TODO (phase C). `derived = (paper_score / sum(paper_score)) * matchability`, rescaled to sum 1. Decision D3: the derived Moon thermal-inertia weight (0.019) drives the run; the published 0.03 is kept for comparison.

## Purposes (s4)

TODO (phase C). Multiply base weights by the purpose multipliers, then renormalise to sum 1 (`config/purposes.yaml`).

## Monte Carlo (s5)

TODO (phase C). 2,000 runs, master seed 20261114 (D2); jitters weights and data (D8); batched float32, online statistics, no N x 2000 matrix.

## ML (s6)

TODO (phase D). Second opinion only (D9); it never produces the score.

## Validation (s7)

TODO (phase D). Pass criteria are pre-registered in `config/known_sites.yaml` before anything is computed. Known-site coordinates are unconfirmed until issue D01 is closed.

## Export (s8)

TODO (phase E). See `docs/RESULTS_SCHEMA.md` once it exists.

## Discovery (s9)

TODO (phase E). Robust tiles more than 100 km from every known site.

## Ledger and fidelity

TODO (phase C). Limits ledger `config/ledger.yaml`, reported beside every score and never multiplied into it. `fidelity = 100 * mean(status_value over the ledger rows relevant to that purpose)`, Broken 0, Partly 0.5, Matchable 1. The relevance matrix is provisional until Anay and Vihaan review it.

## Known weaknesses

TODO (grows as results arrive). Start list:

- `water_index` and `silica_index` compare different instruments on different bodies.
- Earth cannot match polar lighting, gravity, air pressure or cold traps (see the ledger).
- Match-ability values (the 0 to 1 column in `config/weights.yaml`) are expert judgement, not measured, and are the weakest link in the weights.
- Known-site coordinates are approximate and unconfirmed.
- **ASTER GED coverage limits (`silica_index`).** The Earth silica proxy comes from ASTER GED v003, built from clear-sky ASTER data of **2000-2008**, not the 2015-2024 window used for MODIS. Its catalog bounding box is -59 to 80 degrees latitude and the page states no no-data value, so high-latitude, persistently cloudy, icy and open-water tiles may be missing. A tile with no silica value is `unavailable` and the missing-data gate rejects it for any target whose weights use `silica_index`. EMIT (2022 onwards, surface reflectance only, no mineral product in the Earth Engine collection) is not a substitute.
