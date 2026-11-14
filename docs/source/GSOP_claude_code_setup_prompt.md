# GSOP: Claude Code setup prompt v2, 14 Nov (sim repo + main repo + brand-asset intake + handoff to teammate S)

Paste everything below the line into Claude Code on the machine of whoever builds the sim (currently Yash). Start Claude Code from any folder; it will find or create `lenovo/` itself (section 2).
Before pasting: save `GSOP_combined_-_plan__weights_and_evidence.md` and `GSOP_brand_assets_prompts.md` somewhere on the machine and tell Claude Code the path in your first message after the paste (optional; every number it needs is embedded below).

---

# TASK

You are setting up the GSOP simulation software and its main repo. GSOP (Geospatial Similarity Observation Platform) is a NASA Space Apps 2026 entry. It scores every 6 km land tile on Earth by how well it matches a Moon-base or Mars landing target, runs a 2,000-run Monte Carlo on the score, validates against known analog sites, and exports results for a website.

Machines:
- **The machine running this prompt** (person: **Yash**; OS and specs unknown, detect them with `gsop doctor` and report). It builds the repos.
- **Anay**: Windows 11, 8 cores, RAM unknown. Runs the **Moon** targets.
- **S** (Shaurya): macOS on Apple M1, 8 cores, RAM unknown. Runs the **Mars** targets.
All use the same Earth feature file. The sim code is sent to S as a zip.

**Who you talk to.** The person running you is Yash. Wherever this prompt says "ask Anay", ask Yash; if it is a decision (weights, thresholds, coordinates, which place wins), tell Yash it is for Anay and carry on with the documented default. The GitHub issue tracker uses phases A to F; the phases below are 1 to 5 and are a different numbering.

Work in phases. Commit at the end of each phase. After each phase print a short status (what works, what failed, what you assumed). If a data download or login fails twice, stop and ask Yash instead of working around it.

## 0. Ground rules (non-negotiable)

1. **Never invent numbers, coordinates, dataset IDs, URLs or paper values.** Verify dataset IDs live (for Earth Engine, query the catalog; for PDS, open the page). Anything unverified goes into the file `docs/UNVERIFIED.md` with where it is used. A feature you cannot build becomes `unavailable`; the "missing data" gate then handles it and the reason is logged. Never substitute silently.
2. **Do not push, create GitHub remotes, delete files, or change anything outside `lenovo/`.** Print the commands for Yash or Anay to run instead.
3. **Windows and Mac must both work.** Use `pathlib`, no bash-only scripts, a single CLI (`uv run gsop ...`), multiprocessing that is spawn-safe (Windows and macOS both spawn), and a `uv.lock` so both machines get identical package versions.
4. **Same operator on both sides.** Terrain features (slope, relief, roughness) must be computed with one documented operator on Earth, Moon and Mars. Write the operator once in `src/gsop/operators.py` (numpy). If Earth Engine has to compute the Earth side, implement the matching operator there and add `gsop check-operators` (see section 4) that compares the two on one small test window.
5. **Be blunt in reports.** If something is weak or unverified, say so in the status. Do not smooth it over.
6. **No design.** Anay designs the site himself. You build structure, data and code only: no layout, no component styling, no fonts, no images placed into pages, no animation. Pages stay plain, with default browser styling. The only design-related work in scope is (a) the locked colour tokens (Part B step 3) and (b) brand-asset intake (Part C).

## 1. Decisions already made by Anay (do not reopen)

- D1. Tile size is **6 km x 6 km**, on the equal-area grid EPSG:6933 (6000 m cells). Land only, with polar ice gated out.
- D2. **2,000 Monte Carlo runs** per configuration (`--runs` overridable). Master seed `20261114`; per-run seeds from `numpy.random.SeedSequence(master).spawn(n)` so results do not depend on worker count.
- D3. **Lunar thermal inertia weight.** Use the weight derived exactly by the rule (0.019). The published table shows 0.03, which is a rounding-up artifact so the column sums to 1. Keep both in config; the derived value drives the run.
- D4. **Illumination.** The primary run follows the published weights (illumination 0.18 in the Moon similarity score). Also run a sensitivity variant `no_illumination` (illumination weight set to 0, rest renormalised) for the Moon, and report whether the robust tiles survive. Earth cannot match polar lighting, so also report illumination in the fidelity output.
- D5. **Purpose profiles.** Moon has 3 (landing_mobility, power_night, ice_resources) plus `base`. Mars has 4 (the three plus geology_life) plus `base`. **No Moon life-detection.** Mark Mars landing_mobility and power_night results `confidence: low`.
- D6. **Mars target check gate** (target within +/-50 deg latitude, elevation below 2 km, thermal inertia >= 100): implement it as `warn` by default. If a target fails (Arcadia may), still run it, but label every output for that target `fails_own_gate: true` and put it in `validation.json`.
- D7. **Mars ice variables are one composite** (`water_index`). Do not count ice depth, ice consistency and thermal inertia separately.
- D8. Monte Carlo jitters **weights and data**. See section 3.
- D9. ML runs today as a second opinion only. It never produces the score.
- D10. Arcadia Planitia's centre is **not in our files**. Load it from the SWIM ice map if you can reach it; otherwise leave Arcadia as `unscored: no target data`. Do not guess coordinates.

## 2. Repos and folders

Find the folder named `lenovo` (search the user profile, Desktop, Documents, Downloads and drive roots). If several match, ask Yash which one. If none exists on this machine, ask Yash where to create it (suggest `~/lenovo`) and create it there. Inside it:

```
lenovo/
  sim/              <- NEW git repo (own history, MIT licence). The zip for S comes from here.
  nasa-project/     <- NEW (or existing) git repo = main repo, remote anaydhawan/GSOP
  ucip/             <- existing UCIP project, used only as the site-format reference (read, never edit). If it is not on this machine, use the public site instead (Part B step 2).
```

If `nasa-project` or `sim` already exist, inspect them and extend; never overwrite.

## PART A: `lenovo/sim`

### A1. Environment
- `uv` project, Python 3.11 (`.python-version`), `uv.lock` committed. If `uv` is missing, install it (Windows: `winget install --id=astral-sh.uv -e`, or `pip install uv`).
- Dependencies: numpy, pandas, pyarrow, scipy, scikit-learn, rasterio, pyproj, shapely, psutil, pyyaml, typer, rich, tqdm, requests, earthengine-api, pytest. No numba at first; add it only if profiling shows it is needed and it installs on both OSes.
- `gsop doctor`: prints OS, Python, cores, total and free RAM, free disk, package versions, whether `earthengine-api` is authenticated, and the memory-safe chunk size it will use (cell chunk sized to at most 40% of free RAM).

### A2. Layout
```
sim/
  README.md  HANDOFF_S.md  LICENSE(MIT)  pyproject.toml  uv.lock  .gitignore
  config/   weights.yaml  gates.yaml  purposes.yaml  targets.yaml  known_sites.yaml
            ledger.yaml  features.yaml  run.yaml
  src/gsop/ cli.py  grid.py  operators.py  similarity.py  gates.py  weights.py
            montecarlo.py  ml.py  validate.py  discover.py  export.py
            earth/ (gee_export.py, local_loaders.py)   targets/ (moon.py, mars.py)
  tests/    (see section 6)
  scripts/  make_zip.py   (cross-platform; zips tracked files only, no data/, results/, .venv/)
            make_results_zip.py  (zips results/<body>/ plus run_meta.json)
  data/     raw/ interim/ processed/ (gitignored, with .gitkeep and a MANIFEST.json writer)
  results/  (gitignored)
  docs/     SPEC.md  DECISIONS.md  UNVERIFIED.md  PROVENANCE.md  source/ (copy of the md file if found)
```
`docs/SPEC.md` is written as you build: every formula, every assumption, every deviation from the published plan. A person outside the team must be able to re-implement the score from it.

### A3. Config: write exactly these values

`config/weights.yaml` (the weight of a family is split equally among its columns unless stated):

```yaml
# derived = (paper_score / sum(paper_score)) * matchability, rescaled to sum 1.  Code computes it; do not hard-code.
moon:
  slope:           {paper_score: 16, matchability: 1.0, range: [0.21, 0.39], published: 0.30, columns: [slope_med, slope_p90]}
  illumination:    {paper_score: 19, matchability: 0.5, range: [0.10, 0.24], published: 0.18, columns: [sun_max_elev, sun_low_share]}
  cold_dry:        {paper_score: 26, matchability: 0.3, range: [0.10, 0.20], published: 0.15, columns: [water_index]}
  roughness:       {paper_score: 7,  matchability: 0.9, range: [0.07, 0.25], published: 0.12, columns: [rough_a, rough_b]}
  relief:          {paper_score: 6,  matchability: 1.0, range: [0.07, 0.18], published: 0.11, columns: [relief]}
  temperature:     {paper_score: 16, matchability: 0.2, range: [0.04, 0.09], published: 0.06, columns: [t_mean, t_coldest]}
  composition:     {paper_score: 7,  matchability: 0.4, range: [0.03, 0.09], published: 0.05, columns: [silica_index]}
  thermal_inertia: {paper_score: 2,  matchability: 0.5, range: [0.02, 0.04], published: 0.03, columns: [ti_proxy]}
mars:
  composition:     {paper_score: 26, matchability: 0.7,  range: [0.16, 0.30], published: 0.23, columns: [silica_index]}
  water:           {paper_score: 22, matchability: 0.7,  range: [0.12, 0.25], published: 0.19, columns: [water_index]}
  sunlight:        {paper_score: 17, matchability: 0.7,  range: [0.10, 0.20], published: 0.15, columns: [insol_annual, sun_max_elev]}
  slope:           {paper_score: 9,  matchability: 1.0,  range: [0.08, 0.16], published: 0.11, columns: [slope_med, slope_p90]}
  thermal_inertia: {paper_score: 12, matchability: 0.6,  range: [0.06, 0.12], published: 0.09, columns: [ti_proxy, dust_index]}
  roughness:       {paper_score: 7,  matchability: 0.9,  range: [0.05, 0.10], published: 0.08, columns: [rough_a, rough_b]}
  relief:          {paper_score: 6,  matchability: 1.0,  range: [0.05, 0.11], published: 0.08, columns: [relief]}
  elevation_density: {paper_score: 20, matchability: 0.15, range: [0.02, 0.05], published: 0.04, columns: [air_density]}
  temperature:     {paper_score: 5,  matchability: 0.5,  range: [0.02, 0.05], published: 0.03, columns: [t_mean, t_coldest]}
alt_weight_sets:        # run as checks; report top-tile overlap and Spearman rank correlation against the primary
  moon:
    liu2025:   {roughness: 0.39, slope: 0.26, cold_dry: 0.16, illumination: 0.11, composition: 0.08}
    no_illumination: {rule: "primary with illumination=0, renormalised"}
    equal:     {rule: "all families equal"}
  mars:
    mashal2025: {composition: 0.75, water: 0.125, sunlight: 0.125}
    equal:     {rule: "all families equal"}
```
Test: derived weights are Moon 0.3042, 0.1806, 0.1483, 0.1198, 0.1141, 0.0608, 0.0532, 0.0190 (in the order above) and Mars 0.2289, 0.1937, 0.1497, 0.1132, 0.0906, 0.0792, 0.0755, 0.0377, 0.0314. Each must be within 0.012 of `published`, and each set sums to 1.

`config/purposes.yaml`: multiply base weights, then renormalise to sum 1.
```yaml
landing_mobility: {multipliers: {slope: 2, roughness: 2, relief: 2},
                   extra_gate: {moon: "slope_med <= 8", mars: "slope_p90 <= 30"}}
power_night:      {multipliers: {illumination: 2.5, sunlight: 2.5, temperature: 2}}
ice_resources:    {multipliers: {cold_dry: 2, water: 2, thermal_inertia: 2, temperature: 2}}   # cold_dry is the Moon's water family
geology_life:     {bodies: [mars], multipliers: {composition: 2, water: 1.5}}
```

`config/gates.yaml` (a tile failing any gate is rejected; log the reason per tile):
- open water or permanent ice covers more than 50% of the tile
- median NDVI above 0.2 (tunable; keep the threshold in config)
- built-up or farmed land covers more than 20% of the tile
- landing_mobility extra gate (above)
- missing data for any feature the target's weights use
- Mars target check (D6)

`config/features.yaml`: feature columns and sources (shown below as notes; write valid YAML). Earth sources are **leads to verify, not facts**:
```yaml
slope_med, slope_p90: slope at one shared 500 m baseline, median and 90th percentile over the tile.   Earth: NASADEM (SRTM) between 56S and 60N; Copernicus DEM GLO-30 above 60N.  Moon: LOLA.  Mars: MOLA.
relief:               elevation range (max-min of 500 m cells) in the tile.   same sources
rough_a, rough_b:     RMS height after plane removal at two baselines (default ~100 m and ~1 km, TODO confirm with Vihaan).   same sources
sun_max_elev, sun_low_share, insol_annual:  computed from latitude, obliquity, slope/aspect. sun_low_share = share of daylight with the Sun under 5 deg.  Moon pole targets: Sun within about 1.5 deg of the horizon all year (so sun_max_elev ~ 1.5, sun_low_share ~ 1.0).
t_mean, t_coldest:    annual mean and coldest-month surface temperature.   Earth: MODIS LST.  Moon: Diviner (not at the pole; flag).  Mars: climate data.
ti_proxy:             day-night temperature swing in the dry season (three driest consecutive months; fall back to annual median and flag).  Earth: MODIS.  Moon: Diviner.  Mars: THEMIS.
dust_index (Mars):    shares the thermal_inertia weight.  Earth: MERRA-2 dust or MODIS aerosol (lead).  Mars: TES albedo/dust cover.
water_index:          0-1. Earth: mean of permafrost probability and a 0-1 aridity index (lead sources; GPM, SMAP).  Moon: PSR share of the tile.  Mars: SWIM ice consistency.
silica_index:         thermal-infrared emissivity band ratio.  Earth: ASTER GED (EMIT only where available).  Moon: Diviner Christiansen feature.  Mars: TES.
air_density (Mars family only): standard-atmosphere density from elevation on Earth, exponential pressure law on Mars (constants from config with a source; mark TODO if unsourced).
gate layers (Earth only): water/permanent ice fraction, median NDVI (MODIS NDVI), built-up/cropland fraction (land-cover product).
```
`water_index` and `silica_index` are cross-instrument proxies. Document in SPEC.md that these two are the weakest comparisons.

`config/targets.yaml`:
```yaml
moon:
  connecting_ridge_c1: {lat: -89.45, lon_e: 222.84, source: "Glaser 2018", lit_surface_pct: 76.0, lit_2m_pct: 88.0, max_shadow_h_surface: 268, max_shadow_h_2m: 112, nearest_psr_m: 100, psr_slope_deg: 8.4}
  shackleton_rim:      {lat: -89.78, lon_e: 203.97, source: "Glaser 2018", lit_surface_pct: 81.0, lit_2m_pct: 85.5, max_shadow_h_surface: 221, max_shadow_h_2m: 66, nearest_psr_m: 370}
mars:
  jezero:  {lat: 18.4386, lon_e: 77.5031, source: "Grant 2018"}
  arcadia: {lat: null, lon_e: null, source: "centre to be taken from the SWIM ice map", status: unscored_no_target_data}
target_window_km: 15      # features computed on 6 km tiles (stride 3 km) inside the window; value = median, spread = IQR/1.349
```
Receiver height for illumination is a parameter, default 2 m. "Shackleton ridge" means Connecting Ridge C1. Use that name everywhere. (Do not use C2.)

`config/ledger.yaml`: the limits ledger. Reported beside every score, **never multiplied into it**. Rows (property, target value, best Earth can do, status, source):
- Gravity | 1.62 m/s2 Moon, 3.71 Mars | 9.81 everywhere | Broken | Fudali 1980; Foucher 2021
- Air pressure | vacuum Moon; under 1% of Earth on Mars | about half sea level on the highest roads | Broken | Pandey 2020
- Cold traps | below 110 K, down to 38 K | about 184 K air-temperature record; satellite surface minimum is lower, check before quoting | Broken | Hayne; Paige
- Polar lighting | Sun within about 1.5 deg of the horizon all year, 18.6-year cycle | low Sun in polar spring and autumn only | Partly | Mazarico; Glaser
- Radiation and UV | no shielding | UV-A 10x sea level in Ladakh; no particle radiation | Partly | Pandey; Harris
- Regolith | unweathered, glass-welded, dry | weathered, oxidised, damp ash and basalt | Partly | McKay 1993; Foucher
- Rock chemistry | iron-rich Martian basalt; lunar anorthosite | basalt 45-52% silica; anorthosite rare | Partly | Foucher 2021
- Oxidants | 0.5-1 wt% perchlorate on Mars | up to 0.6 wt% in the Atacama | Partly | Harris 2022
- Sterility | no known life | 1.7-13.6 million cells per gram in desert cores | Broken | Stoker; Harris; Pandey
- Line of sight to Earth, GPS | Moon site criterion; no GPS anywhere | always available | Broken | Artemis III; Young 2013
- Electrostatic dust | status Broken, provisional | no global Earth data | source: Shiven's report (unverified)
- Regolith mechanics (grain size, density, abrasiveness) | status Partly, provisional | no global Earth data | source: Shiven's report (unverified)

**Fidelity** (separate output, 0-100, per target x purpose): `fidelity = 100 * mean(status_value over the ledger rows relevant to that purpose)` with Broken 0, Partly 0.5, Matchable 1. Draft the relevance matrix (which rows matter for which purpose) in `ledger.yaml` with `provisional: true` and a one-line reason each; Anay and Vihaan will review it. Also output, per tile, an `illumination_match` (the illumination-family similarity alone) so the site can show it.

`config/known_sites.yaml`: **all coordinates below are approximate, from Claude's memory, and unconfirmed.** Shaurya must confirm them before any result is published. Write them with `confirmed: false`, make `gsop` print a warning listing unconfirmed sites at the start of every validation run, and stamp `provisional: true` into `validation.json`. Validation uses the best-scoring tile within `site_match_radius_km: 20` of each point so small coordinate errors do not matter.
```yaml
primary:      # the 10 Nov validation set
  lonar:            {lat: 19.98,  lon: 76.51,   bodies: [mars, moon]}
  haughton:         {lat: 75.37,  lon: -89.68,  bodies: [mars, moon]}
  mcmurdo_dry_valleys: {lat: -77.5, lon: 162.0, bodies: [mars]}
  atacama_salars:   {lat: -23.5,  lon: -68.25,  bodies: [mars]}      # salars, not Yungay
  ladakh_puga:      {lat: 33.22,  lon: 78.32,   bodies: [mars]}
  mdrs_utah:        {lat: 38.41,  lon: -110.79, bodies: [mars]}
secondary:    # in the novelty exclusion list and sensitivity checks, not in the pass criterion
  sf_volcanic_field: {lat: 35.35, lon: -111.5,  bodies: [moon], radius_km: 30}
  gosses_bluff:     {lat: -23.8,  lon: 132.3,   bodies: [moon, mars]}   # the file flags this as uncertain
  mauna_kea:        {lat: 19.82,  lon: -155.47, bodies: [moon]}
  askja:            {lat: 65.03,  lon: -16.75,  bodies: [moon, mars]}
gate_tests:   # a correct pipeline must reject all three
  manicouagan:      {lat: 51.1,  lon: -68.7,   expect: rejected}       # coordinates as given in our files
  tatacoa:          {lat: 3.23,  lon: -75.17,  expect: rejected}       # as given in our files
  samoylov:         {lat: 72.37, lon: 126.48,  expect: rejected}       # approximate
negative_controls: # must score low
  amazon_basin:     {lat: -3.0, lon: -60.0}
  congo_basin:      {lat: -0.5, lon: 23.0}
```
Pre-registered pass criteria (write into `validation.json` before computing anything): median rank percentile of the primary sites for their intended body >= 90th; negative controls below the 25th percentile; all three gate tests rejected. Report pass or fail as it comes. Do not tune weights to pass.

### A4. Pipeline commands (file names as in our plan)
```
gsop doctor
gsop s1 tiles                        grid + land mask -> data/interim/tiles.parquet
gsop s2 earth   [--source gee|local] Earth features -> data/processed/earth_features_6km.parquet (+ grid.json, MANIFEST sha256)
gsop s2 targets --body moon|mars     target profiles -> data/processed/targets_{body}.parquet (value, spread, source, evidence grade)
gsop s3 gates   --body moon|mars     passed/rejected tables with reasons
gsop s4 score   --body moon|mars     base similarity 0-100 per target x profile
gsop s5 simulate --body ... --runs 2000
gsop s6 ml      --body ...
gsop s7 validate --body ...
gsop s8 export  --body ...           minimal: parquet/JSON per view, schema in docs/RESULTS_SCHEMA.md
gsop s9 discover --body ...          minimal: robust + novel candidate list only (cards are written by people)
gsop run --body moon|mars [--runs N] runs s3..s7 (+s8, s9) in order, resumable
gsop golden                          runs a tiny fixed slice end to end (see section 5) and writes golden.json
gsop verify-data                     checks sha256 of data/processed files against MANIFEST.json
gsop compare DIR_A DIR_B             compares two result folders with tolerances, writes a report
gsop merge  RESULTS_MOON RESULTS_MARS  -> results/all/ for the site
gsop check-operators                 Earth Engine vs numpy operators on one test window
```
`gsop run` is resumable (skips finished stages whose input hashes and config hash are unchanged) and writes `run_meta.json` (git commit, config hash, package versions, machine info, seeds, timings).

### A5. Scoring and Monte Carlo (section 3 spells out the maths)

## 3. Scoring maths and Monte Carlo

**Similarity per column.** `s_k = exp(-0.5 * ((x_tile - x_target) / sigma_k)^2)`, in [0, 1]. A family's score is the mean of its columns' similarities. Tile score = `100 * sum_f w_f * S_f`, with `w` the (purpose-adjusted) family weights.

**Sigma rule** (document in SPEC.md): `sigma_k = max(target_spread_k, measurement_uncertainty_k, 0.25 * IQR_of_k_across_all_gated_Earth_tiles)`. The last term stops the score collapsing to zero everywhere for variables Earth cannot match (for example polar lighting). Put the measurement-uncertainty floors in `features.yaml` with a source, for example Barker 2021: Moon slope error 1.5 to 2.5 deg, height error 0.3 to 0.5 m. Where there is no source, mark it `TODO` and use the IQR term only.

**Monte Carlo, per run** (2,000 runs per target x profile):
1. Draw each family weight uniformly within its `range` in `weights.yaml`, then renormalise to sum 1 (apply the purpose multipliers after drawing, then renormalise again).
2. Jitter the data: target value of each column ~ N(value, measurement_uncertainty); a systematic Earth bias per column ~ N(0, Earth_uncertainty_k); optional small per-tile noise only for candidate tiles. Document exactly what you did.
3. Score all tiles. Do not keep an N x 2000 matrix (about 4 million tiles; it will not fit). Process runs in batches, accumulate online statistics, and use float32.
4. Per run, find the top 1%, 5% and 10% tiles with `argpartition` and count membership per tile (uint16 counters).
5. Pass 2, only for candidate tiles (top 3% by mean score or membership frequency): recompute their per-run scores with the same seeds and get exact median, 5th and 95th percentile of score, plus the rank-percentile range among all tiles (sort each run's scores once and use `searchsorted`).
6. Outputs per target x profile: `mc_median`, `mc_p05`, `mc_p95`, `rank_pct_p05/p50/p95`, `freq_top1`, `freq_top5`, `freq_top10`, `robust` (top 5% in at least 90% of runs; thresholds in `run.yaml`), plus `convergence.json` (top-1% set overlap at 500, 1,000, 1,500 and 2,000 runs).
7. Alternative weight sets and the `no_illumination` variant: score once each (no Monte Carlo) and report top-1% overlap and Spearman against the primary base score.

Parallelise runs across all cores with `concurrent.futures.ProcessPoolExecutor`; share the feature array via `multiprocessing.shared_memory` or `np.memmap`, never by pickling. Budget: one target x profile at 2,000 runs should take under about 30 minutes on 8 cores. If it does not, profile first, optimise second, and tell Anay the real number.

**ML (second opinion, D9).** (a) k-means or GMM terrain archetypes on the gated Earth tiles, choose k by silhouette on a sample, report which archetype each primary site and each robust tile falls in. (b) A classifier (logistic regression and a small random forest) trained on the primary known sites against random gated tiles, leave-one-site-out, with the spatial-block rule: tiles within 100 km of a held-out site are excluded from training. Report AUC per held-out site, feature importances, and Spearman between classifier probability and the index. Say plainly that there are only 6 positives. Ship an ML result only if it adds information; otherwise report "no added signal".

**Discovery (s9, minimal).** Candidates = `robust` tiles more than 100 km (haversine) from every site in `known_sites.yaml` (primary and secondary). Output top candidates with tile coordinates, score band, which features drive the match, which gates were close, and an empty `process_tag` (impact, volcanic, aeolian, periglacial, evaporite) for a person to fill. Zoom to a 2 km window and the discovery write-ups are done later by people.

## 4. Data plan

**Think before downloading.** A global 30 m DEM is hundreds of GB; do not download it. The route:

1. **Earth features via Google Earth Engine (primary route).** Compute every Earth layer server-side and export **6 km rasters** (`crs='EPSG:6933', scale=6000`), one band per feature, in chunks if a single export times out. Roughly 12 bands at 6 km global is well under 1 GB in total. Slope for the tile: resample the DEM to a 500 m baseline first (`reduceResolution`, mean), then slope, then reduce to the 6 km tile with percentile 50 and 90. Use NASADEM between 56S and 60N and Copernicus GLO-30 above 60N. Verify every collection ID in the live catalog before using it, and record id, version, date range, resolution and licence in `docs/PROVENANCE.md`. Use 2015-2024 climatology for MODIS. Gate layers: water and permanent ice fraction, median NDVI, built-up and cropland fraction.
   - Whoever's Earth Engine account is used must authenticate (`earthengine authenticate`) and give you the Cloud project ID. Ask Yash whose account that is (his or Anay's) and let that person log in; never try to bypass login. Test on one small region first (for example a 600 km box around Lonar) before the global export.
   - Only **one** machine builds the Earth file. The build machine (ask Yash: this one or Anay's) builds it and writes `grid.json` and `data/MANIFEST.json` (sha256 of the parquet). S downloads it from a shared Drive link and verifies the hash.
2. **Fallback if Earth Engine is unavailable:** direct open downloads at coarser resolution (for example a global ~1.8 km relief grid for terrain) plus MODIS and ASTER products from NASA Earthdata. Say plainly in SPEC.md that terrain features are then coarser, and that slope at a 500 m baseline cannot be computed from them. Features that cannot be built become `unavailable`.
3. **Target profiles.** Moon (Anay's machine): LOLA south-polar DEM windows around the two targets, Diviner for temperature where it exists (not at the pole; flag). Mars (S's machine): MOLA elevation, THEMIS and TES thermal inertia and albedo, SWIM ice consistency (swim.psi.edu), around Jezero and Arcadia. Use windowed raster reads (`rasterio`), never a full-planet download. Compute with `operators.py`. Every target value gets `source`, `evidence_grade` (A measured, B nearby or same class, C inference) and `window`. Malapert has no data; leave it out.
4. **`gsop check-operators`**: run the Earth Engine path and the numpy path on one small Earth window and compare slope_med, relief and rough_a. Report the difference. If the two disagree by more than 10% on any, fix it before any global run.

## 5. Tests (pytest; must pass on both OSes)

1. Weights: derived values match section A3, sums are 1, within 0.012 of published.
2. Similarity: exact target match gives 1; far values give about 0; sigma floor works; no NaN or underflow ties on a synthetic grid.
3. Gates: synthetic tiles each fail the intended gate with the right reason; missing data is rejected.
4. **Planted analog test:** a synthetic Earth grid with one tile equal to the target must rank first with `robust: true`; a tile that matches on only one family must not.
5. **Determinism:** same seed gives identical output twice; worker count 1 vs 4 gives identical output.
6. Memory: a synthetic 4,000,000-tile run stays under the chunk budget that `doctor` reports.
7. `gsop golden`: fixed slice of about 5,000 tiles, 50 runs, both bodies, checked into the repo as `tests/golden/` inputs with expected `golden.json` (scores to 1e-4, top-1% sets with Jaccard at least 0.99). The two machines use different BLAS libraries, so **do not require bit-identical outputs**; hash the inputs exactly and compare outputs with tolerance.

## 6. Handoff to S (write `HANDOFF_S.md`, step by step, macOS)

1. Unzip the sim zip. Install `uv` (`brew install uv` or the astral.sh install script). In the folder: `uv sync`, then `uv run gsop doctor`; send Yash and Anay the output.
2. Put `earth_features_6km.parquet`, `grid.json`, `MANIFEST.json` into `data/processed/`. Run `uv run gsop verify-data` (checks sha256).
3. `uv run pytest -q` then `uv run gsop golden`; send `golden.json` to Yash and Anay. Anay or Yash runs `gsop compare` on both.
4. `uv run gsop s2 targets --body mars`, then `uv run gsop run --body mars --runs 2000`.
5. Zip `results/mars/` (use `uv run python scripts/make_results_zip.py mars`) and send it back, with `run_meta.json`.
6. If anything fails, send the full error text and `doctor` output; do not edit code.
Also write `scripts/make_zip.py` (tracked files only, plus `uv.lock`) and print the exact command to make the zip. Anay (or Yash) then runs `gsop merge` on `results/moon` and S's `results/mars`.

## PART B: `lenovo/nasa-project` (main repo, remote `anaydhawan/GSOP`)

Initialise only. **No design** (ground rule 6): do not build any page beyond what is listed here.

1. `git init` if needed. Check the remote: `gh repo view anaydhawan/GSOP`. If it exists and is reachable, `git remote add origin https://github.com/anaydhawan/GSOP.git` (do not push). If it does not exist or is not reachable, print the `gh repo create anaydhawan/GSOP --public` command for Anay to run and stop there. Add MIT `LICENSE`.
2. Read `lenovo/ucip` (or, if missing, fetch https://uciplatform.vercel.app) and identify its stack from `package.json`, the config files and the folder layout. Scaffold `site/` with **the same stack and conventions** (framework, language, styling approach, lint and format setup). Do not copy UCIP content or branding. Install and run it once to confirm it builds.
3. **Colour theme (locked).** Define these as variables only, in one tokens file in the site's styling layer (CSS variables or the equivalent for that stack): Void `#06080A`, panel `#0D1114`, hairline `#2C3238`, Moon `#ECEDE8`, Titanium `#9AA0A6`, Earth `#1E8C6A`, Earth Light `#46D9A6`, Mars `#A8372A`, Mars Light `#FF6F5B`, Mars Deep `#5A1C16`. **Define, do not apply**: no page background, no text colour, no component styling uses them yet; Anay does that. The rule from now on: any colour that ever appears in code comes from these tokens, no raw hex anywhere else, and no orange or saffron outside the Mars planet image. Dark is the main theme; a white day mode is planned: leave a `TODO day mode (colours not decided)` in the tokens file and do not invent colours for it. Add `scripts/check_colors.py` (plain Python, no dependencies): it scans the site source for hex, `rgb()`/`hsl()` literals and the words `orange` and `saffron` outside the tokens file, flags any hex that is not one of the ten above, prints file:line for each, and exits 1 if it found any.
4. `docs/`: `README` pointing to the plan, `DECISIONS.md` (copy of section 1 above with today's date), `AI_USE_LOG.md` (stub: date, tool, what it was used for, who checked it), `BUILD_LOG.md` (dated entries, UTC+05:30, including this setup session), `RESULTS_SCHEMA.md` (what the site reads from `results/all/`: file names, columns, types; mirror the sim's export).
5. `data/` or `public/results/` placeholder where merged results will be copied, with a note that files must stay under 3 MB each.
6. `.gitignore`, a root `README.md` with the project's one-paragraph description (GSOP searches the whole Earth for places that can stand in for Moon-base and Mars landing sites; its product is two or three newly identified places, each measured and written up) and how the sim repo relates to this one.
7. First commit on a `main` branch. Commit message ends with the attribution lines Claude Code is configured to add.

## PART C: brand-asset intake (`lenovo/nasa-project`)

Yash is generating the brand images with the prompts in `GSOP_brand_assets_prompts.md` (STYLE LOCK, Board A, Board B, P1 to P5) in an image tool. **You cannot generate images; never draw, fake or placeholder-generate one.** Your job is the folders, the checklist, the checks and the plain wiring. Still no design (ground rule 6).

**C1. Folders.**
```
nasa-project/
  brand/
    incoming/     <- Yash drops raw files here (git-ignored)
    reference/    <- accepted boards A and B (visual reference only; committed)
    originals/    <- full-resolution accepted files (git-ignored)
  site/public/brand/
    logo/  hero/  og/  footer/  cards/     <- web copies that ship
```

**C2. Expected files and names** (names for boards, P1 and P2 are from the prompts file; P3 to P5 names are chosen here):

| Asset | Spec (from the prompts) | Source file name | Shipped file |
|---|---|---|---|
| Board A, identity | 3:2, 1536x1024, reference only | `gsop_boardA_v1.png` (v2, v3 for more tries) | none (reference) |
| Board B, system | 3:2, 1536x1024, reference only | `gsop_boardB_v1.png` | none (reference) |
| P1 logo | transparent background, 1536x1024 | `gsop_logo_P1.png` | `site/public/brand/logo/` |
| P2 hero | 3:2 | `gsop_hero_P2.png` | `site/public/brand/hero/` |
| P3 OG background | 3:2, crop to 1200x630 | `gsop_og_P3.png` | `site/public/brand/og/gsop_og_1200x630.png` |
| P4 footer horizon | 3:2, "crop wide" (size not decided: TODO ask Anay, do not guess) | `gsop_footer_P4.png` | `site/public/brand/footer/` |
| P5 target cards | three images, 1024x1536 each | `gsop_card_moon_P5.png`, `gsop_card_mars_P5.png`, `gsop_card_earth_P5.png` | `site/public/brand/cards/` |

That is 9 source files in total.

**C3. `docs/BRAND_ASSETS.md`.** The table above plus a status column (`missing`, `received`, `accepted`, `rejected: reason`) and an owner column. **Only Anay marks `accepted`; you never do.** Copy these known problems below the table: image tools garble text and hex codes (boards are visual reference, real text is always HTML text); photoreal discs turn to mush at 16 to 32 px (favicon choice belongs to Anay: if flat discs still muddy, the favicon is one Earth disc); Earth is blue in the logo and green in the UI accent (pick one for charts, tokens only); ribbons keep the S-curve; there is no day-mode prompt and no day-mode colours yet.

**C4. `scripts/check_brand_assets.py`** (run with `uv run --with pillow python scripts/check_brand_assets.py`). For every expected file, in `brand/incoming/` and in `site/public/brand/`: present or missing; opens as PNG or WebP; pixel size against the spec (P1 1536x1024, P5 1024x1536, 3:2 ratio for P2, P3 and P4 sources, 1200x630 for the shipped OG); P1 has an alpha channel with fully transparent corners; file size. Print one table. Informational: exit 1 only for a corrupt file. It never edits `docs/BRAND_ASSETS.md`.

**C5. `scripts/prepare_web_assets.py`.** From files in `brand/originals/` write web copies into `site/public/brand/`: the logo stays PNG with transparency; the others as the smallest file that keeps edges crisp (PNG or WebP; report sizes; hard limit 3 MB each, same rule as the results files). OG: scale to 1200 px wide, then crop 630 px tall anchored to the **bottom** (the P3 prompt puts the ribbons in the lower 30% and the discs bottom-right); say in the report that Anay must eyeball the crop. Footer: skip until the crop size is decided. Never upscale, never overwrite an original.

**C6. Plain wiring only.**
- A dev-only page `/brand-check`, unstyled, not linked from any navigation: for each shipped asset an `<img>`, its file name, pixel size and file size; missing assets listed as `missing`. Note in `STATUS.md` that it must be removed or set to `noindex` before the final freeze.
- In the page `<head>`: `og:image` pointing at the shipped OG file and a favicon link, each added only if the file exists at build time, so nothing is ever a broken link. Do not create a favicon.
- All real text (wordmark, descriptor, labels) is HTML text. Never rely on text inside an image.

**C7. Report** in the Phase 4 status: a table of the 9 source files (received or missing, size, checks passed) and the output of `scripts/check_colors.py`.

## 7. Order of work and reporting

- **Phase 1** (do first, about an hour): Part A scaffold, config, weights test, similarity, gates, Monte Carlo engine, ML, validation, discovery, compare, merge, golden test, all passing on **synthetic data**. Commit. Print status.
- **Phase 2**: Earth Engine route: ask Yash who authenticates, run the small-region test, `check-operators`, then the global export. Report time and size.
- **Phase 3**: target profiles for the Moon, then a real `gsop run --body moon --runs 50` smoke test, then 2,000 runs.
- **Phase 4**: Part B, then Part C (brand-asset intake), then `HANDOFF_S.md`, `scripts/make_zip.py`, and print the exact steps for Yash or Anay to send S the zip and the Earth feature file.
- **Phase 5**: write `STATUS.md` in both repos: what is done, what is `unavailable`, what is `UNVERIFIED`, measured run times, and the list of things a human must confirm (known-site coordinates, Arcadia centre, ledger relevance matrix, sigma floors, tolerance values, `rough_a` and `rough_b` baselines, the P4 footer crop size, day-mode colours, the favicon choice). The main repo's `STATUS.md` also carries the brand-asset table from C7.

Start with Phase 1 now. First locate (or create) `lenovo/`, then report its exact path, this machine's OS and what is already in the folder.