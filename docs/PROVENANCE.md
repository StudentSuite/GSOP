# Provenance

Where every dataset, paper value and software version comes from. A dataset is only listed as verified after someone has opened the live catalog or archive page and filled in every column. Until then it is a **lead**: it stays `NOT VERIFIED` here and has a row in `docs/UNVERIFIED.md`. Never fill an id, version, date range or licence from memory.

## Datasets

Columns: dataset, id/version, date range, resolution, licence, how accessed, checked on (date), checked by (who).

| Dataset | Id / version | Date range | Resolution | Licence | How accessed | Checked on | Checked by |
|---|---|---|---|---|---|---|---|
| NASADEM (SRTM), Earth DEM 56S-60N | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
| Copernicus DEM GLO-30, Earth DEM above 60N | NOT VERIFIED (lead) | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Earth Engine (planned) | not checked | not checked |
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

## Software

Resolved by `uv lock` (see `uv.lock`, which is the source of truth). Python is pinned to 3.11 in `.python-version`.

| Package | Resolved version | Platform checked | Wheel only (no build from source) | Checked on |
|---|---|---|---|---|
| numpy | 2.4.6 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scipy | 1.17.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| scikit-learn | 1.9.1 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |
| rasterio | 1.4.4 | macOS arm64 (Apple M2, Yash) | yes | 2026-11-14 |

Checked on Apple M2 (arm64), standing in for Shaurya's M1 by Yash's decision: same architecture, but not the same chip. Not yet checked: Windows 11 (Anay), who must run `uv sync --frozen` and the import line from issue A03 and paste the output into the issue.
