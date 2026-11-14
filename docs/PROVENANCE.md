# Provenance

Where every dataset, paper value and software version comes from. A dataset is only listed as verified after someone has opened the live catalog or archive page and filled in every column. Until then it is a **lead**: it stays `NOT VERIFIED` here and has a row in `docs/UNVERIFIED.md`. Never fill an id, version, date range or licence from memory.

## Datasets

Columns: dataset, id/version, date range, resolution, licence, how accessed, checked on (date), checked by (who).

| Dataset | Id / version | Date range | Resolution | Licence | How accessed | Checked on | Checked by |
|---|---|---|---|---|---|---|---|
| NASADEM (SRTM), Earth DEM 56S-60N | `NASA/NASADEM_HGT/001` (Image). Provider NASA / USGS / JPL-Caltech | 2000-02-11 to 2000-02-22 (the SRTM flight) | 30 m, all bands | NASA data policy: may be used for any purpose without prior permission (exceptions on the NASA Data & Information Policy page) | Earth Engine; bands `elevation` (m), `num` (source/scene count), `swb` (water bodies, 255 = water). Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). Re-confirm with `scripts/ee_probe.py` once logged in. | 2026-11-14 | Claude (catalog page only) |
| Copernicus DEM GLO-30, Earth DEM above 60N | `COPERNICUS/DEM/GLO30_2024_1` (ImageCollection, release 2024_1; each tile has `tile_version`). The older id `COPERNICUS/DEM/GLO30` is not mentioned on this page | 2010-12-01 to 2020-11-13 (TanDEM-X acquisition) | 30 m, all bands; a collection because tiles have several resolutions, so mosaic and set the default projection before terrain analysis | Free worldwide licence with required attribution notice (full terms in the Copernicus licence PDF linked from the page) | Earth Engine; bands `DEM` (m above EGM2008), `EDM`, `FLM`, `HEM` (height error, m), `WBM`. Catalog page read on 2026-11-14 through an automated page reader; `getInfo()` probe NOT yet run (no Earth Engine login). Re-confirm with `scripts/ee_probe.py` once logged in. | 2026-11-14 | Claude (catalog page only) |
| MODIS land-surface temperature (2015-2024 climatology) | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| MODIS NDVI | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| ASTER GED (EMIT where available) | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| MERRA-2 dust or MODIS aerosol | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| Permafrost probability map, GPM precipitation, SMAP soil moisture | NOT VERIFIED (leads) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| Land-cover product (built-up, cropland); water and permanent-ice product | NOT CHOSEN (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
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

## Software

Resolved by `uv lock` (see `uv.lock`, which is the source of truth). Python is pinned to 3.11 in `.python-version`.

| Package | Resolved version | Platform checked | Wheel only (no build from source) | Checked on |
|---|---|---|---|---|
| numpy | 2.4.6 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scipy | 1.17.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scikit-learn | 1.9.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| rasterio | 1.4.4 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |

Checked on Apple M2 (arm64), standing in for Shaurya's M1 by Yash's decision: same architecture, but not the same chip. Not yet checked: Windows 11 (Anay), who must run `uv sync --frozen` and the import line from issue A03 and paste the output into the issue.
