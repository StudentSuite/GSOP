# Provenance

Where every dataset, paper value and software version comes from. A dataset is only listed as verified after someone has opened the live catalog or archive page and filled in every column. Until then it is a **lead**: it stays `NOT VERIFIED` here and has a row in `docs/UNVERIFIED.md`. Never fill an id, version, date range or licence from memory.

## Datasets

Columns: dataset, id/version, date range, resolution, licence, how accessed, checked on (date), checked by (who).

| Dataset | Id / version | Date range | Resolution | Licence | How accessed | Checked on | Checked by |
|---|---|---|---|---|---|---|---|
| NASADEM (SRTM), Earth DEM 56S-60N | `NASA/NASADEM_HGT/001` (Image). Provider NASA / USGS / JPL-Caltech | 2000-02-11 to 2000-02-22 (the SRTM flight) | 30 m, all bands | NASA data policy: may be used for any purpose without prior permission (exceptions on the NASA Data & Information Policy page) | Earth Engine; bands `elevation` (m), `num` (source/scene count), `swb` (water bodies, 255 = water). Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). Re-confirm with `scripts/ee_probe.py` once logged in. | 2026-11-14 | Claude (catalog page only) |
| Copernicus DEM GLO-30, Earth DEM above 60N | `COPERNICUS/DEM/GLO30_2024_1` (ImageCollection, release 2024_1; each tile has `tile_version`). The older id `COPERNICUS/DEM/GLO30` is not mentioned on this page | 2010-12-01 to 2020-11-13 (TanDEM-X acquisition) | 30 m, all bands; a collection because tiles have several resolutions, so mosaic and set the default projection before terrain analysis | Free worldwide licence with required attribution notice (full terms in the Copernicus licence PDF linked from the page) | Earth Engine; bands `DEM` (m above EGM2008), `EDM`, `FLM`, `HEM` (height error, m), `WBM`. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). Re-confirm with `scripts/ee_probe.py` once logged in. | 2026-11-14 | Claude (catalog page only) |
| MODIS land-surface temperature (2015-2024 climatology) | `MODIS/061/MOD11A2` (Terra, version 6.1; DOI 10.5067/MODIS/MOD11A2.061). Aqua twin `MODIS/061/MYD11A2` exists but is not planned | 2000-02-18 to 2026-11-01 on the page, so 2015-2024 is fully covered | 1000 m, 8-day | LP DAAC: no restrictions on subsequent use, sale or redistribution (cite via LP DAAC guidance) | Earth Engine; `LST_Day_1km`, `LST_Night_1km` (K, scale 0.02), QA `QC_Day`/`QC_Night` (bits 0-1: 0 good, 1 unreliable, 2 cloud, 3 other; bits 6-7 LST error class), `Clear_sky_days`/`Clear_sky_nights`. No filtering by QA is applied in the 8-day product, so we must mask. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). | 2026-11-14 | Claude (catalog page only) |
| MODIS NDVI | `MODIS/061/MOD13A2` (Terra, version 6.1; DOI 10.5067/MODIS/MOD13A2.061) | 2000-02-18 to 2026-10-16 on the page | 1000 m, 16-day | LP DAAC: no restrictions on subsequent use, sale or redistribution | Earth Engine; `NDVI` (scale 0.0001, valid -2000..10000), `SummaryQA` (0 good, 1 marginal, 2 snow/ice, 3 cloudy), `DetailedQA`. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). | 2026-11-14 | Claude (catalog page only) |
| ASTER GED (EMIT where available) | ASTER: `NASA/ASTER_GED/AG100_003` (Image, v003, DOI 10.5067/COMMUNITY/ASTER_GED/AG100.003). EMIT cross-check: `NASA/EMIT/L2A/RFL` (ImageCollection; the id is inferred from the page heading, the page has no field labelled collection id) | ASTER: 2000-01-01 to 2008-12-31 (clear-sky pixels, so it does NOT overlap the 2015-2024 MODIS window). EMIT: 2022-08-09 to 2026-08-31 | ASTER: 111.32 m, all bands. EMIT: 60 m, 285 reflectance bands (380-2500 nm), about 75 km swath | ASTER: LP DAAC, no restrictions except the ASTER GDEM v2 dataset (the page does not say whether that exception covers the `elevation` band, which we do not use). EMIT: not stated on the page | Earth Engine; ASTER bands `emissivity_band10` .. `emissivity_band14` (scale 0.001), `emissivity_sdev_band*`, `ndvi`, `land_water_map` (1 land, 2 water), `num_obs`. EMIT is surface reflectance only: no mineral identification product on that page, so it is a spectral cross-check, not a silica index. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). | 2026-11-14 | Claude (catalog page only) |
| MERRA-2 dust or MODIS aerosol | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| Permafrost probability map, GPM precipitation, SMAP soil moisture | NOT VERIFIED (leads) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| MODIS land cover (built-up, cropland gate) | `MODIS/061/MCD12Q1` (version 6.1; DOI 10.5067/MODIS/MCD12Q1.061); `MODIS/006/MCD12Q1` is no longer available | 2001-01-01 to 2025-01-01 on the page | 500 m, yearly | LP DAAC: no restrictions on subsequent use, sale or redistribution | Earth Engine; `LC_Type1` (IGBP: 12 cropland, 13 urban and built-up, 14 cropland/natural vegetation mosaic, 15 permanent snow and ice, 17 water bodies), `LW` land/water mask (1 water, 2 land), `QC`. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). | 2026-11-14 | Claude (catalog page only) |
| LOLA (Moon topography) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | PDS (planned) | not checked | not checked |
| Diviner (Moon temperature, silica proxy; not at the pole) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | PDS (planned) | not checked | not checked |
| MOLA (Mars topography) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | PDS (planned) | not checked | not checked |
| THEMIS, TES (Mars thermal inertia, albedo/dust, silica proxy) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | PDS (planned) | not checked | not checked |
| SWIM ice consistency map (also gives the Arcadia centre) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | swim.psi.edu (planned) | not checked | not checked |

### DEM notes (issue B02)

- **Vertical datum differs between the two DEMs.** NASADEM merged void-free files are relative to the EGM96 geoid (SRTM-only files to the WGS84 ellipsoid); Copernicus GLO-30 is relative to EGM2008. Slope and relief use differences of heights, so a constant datum offset does not matter, but a datum that varies in space could add a very small tilt at the 56S / 60N seam. Checked in issue B07.
- **Copernicus GLO-30 is a surface model**: it includes buildings, infrastructure and vegetation (water bodies are flattened). NASADEM is also a radar surface model, so canopy affects both.
- **Voids and water.** NASADEM reduces voids by improved phase unwrapping; the page does not say voids are gone. In `num`, value 0 marks corrected-water pixels and 251 marks quad edges averaged where two neighbouring quads disagreed.
- **Coverage.** The NASADEM catalog bounding box is -56 to 60 latitude. The page text does not state the latitude coverage, so the 56S / 60N switch rule must be tested (issue B07).

### Climate and land-surface notes (issue B03)

- **2015-2024 climatology is possible for MODIS.** Both MOD11A2 and MOD13A2 run from February 2000 to late 2026 on the catalog pages.
- **MOD11A2 does no QA filtering.** It averages all daily LST values and sets each QA value by majority vote of the daily QA values, so the export must mask on `QC_Day`/`QC_Night` itself.
- **ASTER GED is from 2000-2008 only** and from clear-sky pixels. The page lists a bounding box of -59 to 80 latitude but does not describe it as a coverage limit and states no no-data value. Treat high latitudes, persistent cloud, ice and water as gaps until measured (issue B20 reports the real NaN map).
- **EMIT** is reachable in Earth Engine as surface reflectance only. It is not used to build `silica_index`; at most a spectral cross-check.

## Software

Resolved by `uv lock` (see `uv.lock`, which is the source of truth). Python is pinned to 3.11 in `.python-version`.

| Package | Resolved version | Platform checked | Wheel only (no build from source) | Checked on |
|---|---|---|---|---|
| numpy | 2.4.6 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scipy | 1.17.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scikit-learn | 1.9.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| rasterio | 1.4.4 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |

Checked on Apple M2 (arm64), standing in for Shaurya's M1 by Yash's decision: same architecture, but not the same chip. Not yet checked: Windows 11 (Anay), who must run `uv sync --frozen` and the import line from issue A03 and paste the output into the issue.
