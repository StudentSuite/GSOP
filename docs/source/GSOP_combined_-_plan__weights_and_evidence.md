# GSOP: plan, weights and evidence (documents 1, 2 and 5 combined)

Geospatial Similarity Observation Platform · NASA Space Apps 2026 · 11 Nov 2026

This file joins three of the five GSOP documents: what we are doing (document 1), variables and weights (document 2), and paper evidence and research review (document 5). The build prompt (document 3) and the brand and interface design (document 4) are separate.

Tags used below: [FULL] full text or tables were read; [ABSTRACT] abstract only; [SECONDARY] numbers come from another source that quotes the paper; [ADDED] a paper added beyond the team's list; [CORRECT], [PARTLY], [WRONG], [UNVERIFIED] are verdicts on the team's research notes.

# Part 1: What we are doing

**GSOP searches the whole Earth for places that can stand in for Moon-base and Mars landing sites, and its submission is two or three newly identified places, each measured and written up.** Well-known analog sites are the test of the method; new places are the result. A scored similarity index does the ranking, a Monte Carlo puts error bars on it, and two small ML models act as a second opinion.

**Rule check.** Space Apps prohibits project work before the hackathon starts on 14 Nov. These documents are reading and design. Ask your Local Lead whether the pre-made logo and mockup may be used, and paste prompts into Claude Code only after the start.

#### The five documents

| Doc | Title                                   | Open it when                                                                 |
|-----|-----------------------------------------|------------------------------------------------------------------------------|
| 1   | What we are doing                       | You need the plan, the roles, the timeline, or the answer to 'what is GSOP?' |
| 2   | Variables and weights                   | You need a number: a weight, a range, a gate, a ledger row                   |
| 3   | Build prompt: model, simulations and ML | You are at the keyboard with Claude Code (also as a .md file to paste)       |
| 4   | Brand and interface design              | You are building the screen, the site or the video                           |
| 5   | Paper evidence and research review      | A judge or teammate asks 'which paper says that?'                            |

### What GSOP is, against the challenge brief

**GSOP is a method for systematically searching Earth for analogs, and its product is a short list of newly identified places, each characterised.** The brief allows exactly this ("develop a method for systematically searching Earth for previously unrecognized analogs") and still requires the other half ("identify and characterize one or more locations"). So the method alone is not the submission: two or three named discoveries, each with what matches, where it breaks and what it helps rehearse, are the headline.

| Role in GSOP                                                                                                                           |
|----------------------------------------------------------------------------------------------------------------------------------------|
| The test. If the index cannot find Lonar, the Dry Valleys or the San Francisco volcanic field, nobody should trust what else it finds. |
| The result. Robust high-scoring tiles more than 100 km from any known site, zoomed to a specific feature and written up.               |
| The engine. Expert-weighted, with Monte Carlo uncertainty. Not ML.                                                                     |
| The second opinion: clustering, and a classifier trained on the old sites. They do not produce the score.                              |

#### How each line of the brief is covered

| Brief asks for                           | Covered by                                                                                         | Status                                  |
|------------------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------|
| Beyond visual similarity; use data       | Nine measured feature families; no imagery matching                                                | Covered                                 |
| Terrain, slope, roughness, craters       | Slope, relief, roughness features                                                                  | Covered                                 |
| Temperature, illumination, extremes      | Temperature and illumination features; ledger for what cannot match                                | Covered                                 |
| Water, ice, permafrost                   | Water feature                                                                                      | Covered                                 |
| Geology, mineralogy                      | Silica index; mineral flags where maps exist                                                       | Partly: weakest Earth data              |
| Dust, wind, atmospheric processes        | Added now: a dust column beside thermal inertia in the Mars profile; process tag on each discovery | Partly                                  |
| Biology, habitability                    | Only as a gate (vegetation) and a ledger row (sterility)                                           | Stated limit                            |
| Caves, lava tubes, subsurface            | Outside what surface rasters can score                                                             | Stated limit; say so on the limits page |
| Say what matches, how much, what differs | Per-feature match bars and the ledger                                                              | Covered                                 |
| Processes as well as places              | Process tag per discovery (impact, volcanic, aeolian, periglacial, evaporite)                      | Added now                               |
| Look beyond well-known sites             | Novelty filter: 100 km from any known site                                                         | Added now                               |
| Consider scale                           | Zoom step from a 10 km tile to a 2 km window                                                       | Added now                               |
| Multiple datasets                        | Every tile combines terrain, thermal, climate and composition layers                               | Covered                                 |
| Connect to exploration                   | Four purpose profiles; 'what you could rehearse here' on each discovery card                       | Covered                                 |
| Open NASA Earth data                     | Earth sources switched to NASA products first (table in document 2)                                | Changed now                             |
| Remember Earth's limits                  | Limits ledger                                                                                      | Covered                                 |
| Written format for the public            | discoveries.md and the method pages                                                                | Added now                               |
| Surprise us                              | Depends on the result; lead with the least expected robust discovery                               | Open                                    |

The brief says to use open NASA Earth data. The earlier plan leaned on a European elevation model and a non-NASA climate set, so the Earth side now names NASA products first. I have not checked coverage or access for these; treat them as leads to confirm on 13 Nov when the Resources tab opens.

### How it works

One pipeline, nine stages, each a single command. Everything the screen shows is computed beforehand and read from files.

| Stage      | What it does                                                                                                 | Output                                      |
|------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| 1 Tiles    | Cuts Earth's land and each target region into 10 km cells                                                    | Tile table                                  |
| 2 Features | Measures slope, relief, roughness, light, temperature, thermal inertia, water and composition for every cell | One row per cell                            |
| 3 Gates    | Removes cells that are water, ice, vegetated or built-up, and keeps the reason                               | Passed and rejected tables                  |
| 4 Score    | Compares each Earth cell with the target using the weights in document 2                                     | Similarity 0–100                            |
| 5 Simulate | Repeats the score 2,000 times with weights and data jittered                                                 | Median, band, rank range, robust or fragile |
| 6 ML       | Clusters terrain; trains a classifier on known analog sites                                                  | Second opinion and feature importances      |
| 7 Validate | Checks where the known sites rank, and lists the ones that fail                                              | Recovery counts and failure list            |
| 8 Export   | Writes small files for the globe                                                                             | One file per view                           |
| 9 Discover | Keeps robust cells far from any known site, zooms in, drafts a card                                          | Two or three named places                   |

### Timeline for 14–15 Nov

| When    | Step                                      | Command               | Done when                                                                     | Owner                                           |
|---------|-------------------------------------------|-----------------------|-------------------------------------------------------------------------------|-------------------------------------------------|
| h0–h2   | Repo, SPEC.md, config files, brand tokens | —                     | Weights and gates in YAML; tests folder exists                                | Yash                                            |
| h0–h10  | Download and tile                         | `gsop s1`, `gsop s2`  | features.parquet with coverage above 90% for terrain columns                  | Shiven (Moon), Vihaan (Mars), data lead (Earth) |
| h8–h12  | Gates and first scores                    | `gsop s3`, `gsop s4`  | Thin slice ranks tiles for one target                                         | Model lead                                      |
| h10–h30 | Interface on mock data, then real         | `npm run dev`         | Globe, panels and table read results files                                    | Front end                                       |
| h12–h20 | Monte Carlo                               | `gsop s5 --runs 2000` | 20 result files plus convergence.json                                         | Model lead                                      |
| h18–h26 | ML and validation                         | `gsop s6`, `gsop s7`  | validation.json with recovered counts and failure list                        | Shaurya, model lead                             |
| h26–h34 | Discoveries: novelty filter, zoom, cards  | `gsop s9`             | Two or three named places, each with match, break and use written by a person | Shaurya, writer                                 |
| h12–h40 | Method, validation, limits and data pages | Markdown in /site     | Every on-screen number links to a paragraph                                   | Writer                                          |
| h28     | Outside reviewer reads method and limits  | —                     | One correction logged                                                         | Yash                                            |
| h30–h36 | Export and polish                         | `gsop s8`             | Files under 3 MB each; cold load under 3 s                                    | Front end                                       |
| h40     | Feature freeze                            | —                     | Nothing new after this                                                        | All                                             |
| h40–h48 | Video and submission                      | —                     | Two-minute video, AI-use log linked                                           | Yash                                            |

**Data shortcut.** Computing slope for all land from a 30 m DEM is the slowest step. Two workable routes: Google Earth Engine, exporting tile statistics directly (an account is needed; creating one is setup, not project work), or the 90 m Copernicus DEM for a pre-agreed list of dry and cold regions. Decide which on paper before 14 Nov.

### Before 14 Nov

| Person  | Task                                                                                                                                                                                                                            |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Shiven  | Re-check every number marked UNVERIFIED in document 5 against the paper itself, and drop any claim whose only source is a blog. Liu 2025 Figure 7 and Jia 2022 Table 2 are now in.                                              |
| Vihaan  | Mashal 2025 is now in. Re-read the Morgan 2021 landing-requirements sentence and record it exactly; copy the Grant 2018 engineering table.                                                                                      |
| Shaurya | Fix three citations (Pandey is 19(1):78–98; the Hagerty and PISCES titles). Confirm coordinates for the known-site list, now including McMurdo Dry Valleys and Gosses Bluff. Draft the twelve ledger rows with one source each. |
| Yash    | Ask the Local Lead about pre-made design assets. Decide the Earth-data route. Send the two weight tables to the outside reviewer.                                                                                               |

### Words to use

#### What to say when a judge asks "where do the weights come from?"

"From 34 published site-selection and analog studies. We scored how each paper uses each variable, as a gate, a ranking factor or a mention, then discounted each variable by how well Earth can reproduce it. The weights are expert-derived, so we never trust one set: every result is the outcome of 2,000 runs across the plausible range. We then checked the ranking against sites experts already use, and against a separate model trained only on those sites."

#### Never cut

Validation with its failure list, the limits ledger, uncertainty bands, the method pages, the discovery write-ups and the video. Cut first: the second target body, the compare view, sensitivity bars.


---

# Part 2: Variables and weights

**These are the final weights: lunar version 2 and Mars version 2, each from sixteen scored sources.** They are expert-derived by a stated rule, so no single set is trusted: the Monte Carlo draws from the range column, and two published weight sets are run as a check. Shaurya's paper analysis was reviewed on 11 Nov and changes none of these numbers.

### How the weights were derived

1.  **Score each paper's use of each variable.** 3 if the variable is a hard gate or the paper's main driver; 2 if it is used to rank or characterise sites; 1 if it is only mentioned. Sum over papers to get a paper score.
2.  **Convert to importance.** Divide each score by the total.
3.  **Discount by match-ability.** Multiply by a 0–1 value for how well Earth data can represent the variable, taken from question 5. A variable Earth cannot match (0) leaves the similarity score and goes to the limits ledger.
4.  **Normalise to 1.**

<!-- -->

    weight_k = (paper_score_k / Σ paper_score) × matchability_k, then rescaled so Σ weight = 1

The range column is what the Monte Carlo draws from. It spans the result under two scoring schemes (the 3-2-1 roles and a plain count of papers) widened by 30% each way. The match-ability values are my judgment and are the weakest link; your outside reviewer should look at them first.

### Weights

Rule, unchanged: weight = (paper score ÷ total) × match-ability, rescaled to 1. Paper score is 3 for a gate or main driver, 2 for a ranking variable, 1 for a mention. Match-ability is my 0–1 judgment of how well Earth data can represent the variable. Lunar weights are version 2: sixteen sources, up from fourteen.

#### Lunar polar targets (version 2)

| Feature                                                        | Weight   | Monte Carlo range | Paper score | Match-ability |
|----------------------------------------------------------------|----------|-------------------|-------------|---------------|
| Slope (median, 90th percentile)                                | **0.30** | 0.21–0.39         | 16          | 1.0           |
| Illumination regime (highest Sun elevation, share of time lit) | **0.18** | 0.10–0.24         | 19          | 0.5           |
| Cold, dry ground (permafrost, aridity) as cold-trap stand-in   | **0.15** | 0.10–0.20         | 26          | 0.3           |
| Roughness (RMS height at two baselines)                        | **0.12** | 0.07–0.25         | 7           | 0.9           |
| Local relief                                                   | **0.11** | 0.07–0.18         | 6           | 1.0           |
| Temperature regime                                             | **0.06** | 0.04–0.09         | 16          | 0.2           |
| Composition (silica proxy)                                     | **0.05** | 0.03–0.09         | 7           | 0.4           |
| Thermal inertia proxy                                          | **0.03** | 0.02–0.04         | 2           | 0.5           |

Change from version 1: slope 0.27 to 0.30, roughness 0.05 to 0.12, relief 0.16 to 0.11, illumination 0.20 to 0.18. Terrain now carries 0.53 of the lunar score.

#### Mars targets (version 2, sixteen sources)

| Feature                                                                   | Weight   | Monte Carlo range | Paper score | Match-ability |
|---------------------------------------------------------------------------|----------|-------------------|-------------|---------------|
| Composition and alteration minerals (basalt, clays, sulfates, carbonates) | **0.23** | 0.16–0.30         | 26          | 0.7           |
| Water: ground ice or permafrost, and aridity                              | **0.19** | 0.12–0.25         | 22          | 0.7           |
| Sunlight and thermal latitude (annual insolation, day-night swing)        | **0.15** | 0.10–0.20         | 17          | 0.7           |
| Slope (median and 90th percentile)                                        | **0.11** | 0.08–0.16         | 9           | 1.0           |
| Thermal inertia (dust against sand against rock)                          | **0.09** | 0.06–0.12         | 12          | 0.6           |
| Roughness and rock abundance                                              | **0.08** | 0.05–0.10         | 7           | 0.9           |
| Local relief                                                              | **0.08** | 0.05–0.11         | 6           | 1.0           |
| Elevation as air density                                                  | **0.04** | 0.02–0.05         | 20          | 0.15          |
| Temperature regime                                                        | **0.03** | 0.02–0.05         | 5           | 0.5           |

Change from version 1, after reading Mashal 2025 in full: composition 0.22 to 0.23, water 0.20 to 0.19, sunlight 0.14 to 0.15, slope 0.13 to 0.11, thermal inertia 0.08 to 0.09.

#### Published numbers to test against

Two papers now give numbers of their own. Neither is a weight for Earth-analog similarity, so they are used as alternative weight sets: if the top tiles survive them, the ranking does not depend on our rule.

| Paper                          | What it publishes                                                                                                                                                                                                                                                            | Alternative weight set in GSOP features                                                                                                                                |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Liu 2025, Figure 7 (Moon)      | Random-forest importances for south-pole landing suitability: Earth visibility 0.232, hydrogen 0.137, kilometre roughness 0.128, slope 0.115, Sun visibility 0.098, distance to water ice 0.097, hectometre roughness 0.060, geology 0.058, gravity anomaly 0.048, FeO 0.026 | After removing Earth visibility and gravity (ledger) and applying match-ability: roughness 0.39, slope 0.26, cold-dry ground 0.16, illumination 0.11, composition 0.08 |
| Jia 2022, Table 2 (Moon)       | Order only: PSR, slope, rock abundance, illumination, temperature                                                                                                                                                                                                            | None; it agrees with the order of the paper scores                                                                                                                     |
| Mashal 2025, Equation 2 (Mars) | Suitability = mask × plain mean of eight criteria. Six criteria are geology and mineralogy, one is water, one is low-latitude light                                                                                                                                          | Composition 0.75, water 0.125, sunlight 0.125                                                                                                                          |

Liu's largest factor, Earth visibility at 0.23, cannot be reproduced anywhere on Earth. Put that sentence on the limits page.

#### Purpose factors

| Profile                    | Multiply base weights by                      | Extra gate                                                                  |
|----------------------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| Landing and mobility       | Slope ×2, roughness ×2, relief ×2             | Moon: tile median slope at most 8°. Mars: 90th-percentile slope at most 30° |
| Power and night survival   | Illumination or sunlight ×2.5, temperature ×2 | None                                                                        |
| Ice and resources          | Water ×2, thermal inertia ×2, temperature ×2  | None                                                                        |
| Geology and life detection | Composition ×2, water ×1.5                    | None                                                                        |

#### Feature columns

| Column                      | Meaning                                                                                                       | Earth source                                                         | Target source                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------|
| slope_med, slope_p90        | Slope at one shared baseline (500 m)                                                                          | NASADEM (SRTM); Copernicus DEM above 60°N, where SRTM stops          | LOLA; MOLA                                      |
| relief                      | Elevation range in the tile                                                                                   | Same                                                                 | LOLA; MOLA                                      |
| rough_a, rough_b            | RMS height after plane removal at two baselines                                                               | Same                                                                 | LOLA; MOLA                                      |
| sun_max_elev, sun_low_share | Highest Sun elevation; share of daylight with Sun under 5°                                                    | Computed                                                             | Computed; Gläser values for the two lunar sites |
| t_mean, t_coldest           | Annual mean and coldest-month surface temperature                                                             | MODIS land-surface temperature                                       | Diviner (not at the pole); Mars climate data    |
| ti_proxy                    | Day-night temperature swing, dry season                                                                       | MODIS                                                                | Diviner; THEMIS                                 |
| dust_index (Mars profile)   | How often and how thickly dust is in the air; second column of the thermal-inertia family, sharing its weight | MERRA-2 dust or MODIS aerosol (lead)                                 | TES albedo and dust cover                       |
| water_index                 | Permafrost probability plus aridity                                                                           | Permafrost map (lead); GPM precipitation; SMAP soil moisture (leads) | PSR share of tile; SWIM or Piqueux              |
| silica_index                | Thermal-infrared emissivity band ratio                                                                        | ASTER GED; EMIT mineral maps where available (lead)                  | Diviner; TES                                    |
| air_density                 | From elevation (Mars profile only)                                                                            | NASADEM                                                              | MOLA                                            |

### Gates

| Gate                        | Threshold                                                                      | Basis                                        |
|-----------------------------|--------------------------------------------------------------------------------|----------------------------------------------|
| Open water or permanent ice | More than half the tile (set by you)                                           | No terrain analogy possible                  |
| Vegetation                  | Median NDVI above 0.2 (set by you; tune)                                       | Every analog paper treats life as a confound |
| Built-up or farmed land     | More than a fifth of the tile (set by you)                                     | Harris on Lonar; Pandey on Chumathang        |
| Slope, landing profile      | Moon: tile median above 8°. Mars: 90th percentile above 30°                    | Jia 2022; Golombek 2012                      |
| Missing data                | Any feature the target needs                                                   | Honesty of the score                         |
| Mars target check           | Target must lie within ±50° latitude, below 2 km, thermal inertia at least 100 | Mashal 2025                                  |

### Limits ledger

Reported beside every score and never multiplied into it. Two more rows come from Shiven's report: electrostatic dust, and regolith mechanics (grain size, density, abrasiveness), both marked broken or partly with no global Earth data.

| Property                    | Target value                                                   | Best Earth can do                                                                          | Status | Source                    |
|-----------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------|---------------------------|
| Gravity                     | 1.62 m/s² Moon, 3.71 Mars                                      | 9.81 everywhere; crater shape itself depends on gravity                                    | Broken | Fudali 1980; Foucher 2021 |
| Air pressure                | Vacuum on the Moon; under 1% of Earth on Mars                  | About half sea level on the highest roads (0.64 kg/m³ at 5,300 m)                          | Broken | Pandey 2020               |
| Cold traps                  | Below 110 K, down to 38 K                                      | About 184 K record low                                                                     | Broken | Hayne; Paige              |
| Polar lighting              | Sun within about 1.5° of the horizon all year, 18.6-year cycle | Low Sun in polar spring and autumn only                                                    | Partly | Mazarico; Gläser          |
| Radiation and UV            | No shielding                                                   | UV-A 10× sea level in Ladakh, which overshoots present Mars; no particle radiation         | Partly | Pandey; Harris            |
| Regolith                    | Unweathered, glass-welded, dry                                 | Ash and basalt that are weathered, oxidised, damp                                          | Partly | McKay 1993; Foucher       |
| Rock chemistry              | Iron-rich Martian basalt; lunar anorthosite                    | Basalt with 45–52% silica; anorthosite is rare                                             | Partly | Foucher 2021              |
| Oxidants                    | 0.5–1 wt% perchlorate on Mars                                  | Up to 0.6 wt% in the Atacama, far less elsewhere                                           | Partly | Harris 2022               |
| Sterility                   | No known life                                                  | 1.7–13.6 million cells per gram even in desert cores; farms at Lonar; litter at Chumathang | Broken | Stoker; Harris; Pandey    |
| Line of sight to Earth, GPS | Site criterion on the Moon; no GPS anywhere                    | Always available                                                                           | Broken | Artemis III; Young 2013   |

### Targets

| Target              | Body | Coordinates                              | Source                    |
|---------------------|------|------------------------------------------|---------------------------|
| Connecting Ridge C1 | Moon | −89.45°, 222.84°E                        | Gläser 2018               |
| Shackleton rim      | Moon | −89.78°, 203.97°E                        | Gläser 2018               |
| Jezero crater       | Mars | 18.4386°N, 77.5031°E                     | Grant 2018                |
| Arcadia Planitia    | Mars | Centre to be taken from the SWIM ice map | Morgan 2021; Piqueux 2019 |


---

# Part 3: Paper evidence and research review

**This is the evidence behind every number in GSOP: 32 paper cards answering the same five questions, plus three sources read later (Jia 2022, Liu 2025 and the full Mashal 2025), and a claim-by-claim review of Shiven's, Vihaan's and Shaurya's research.** No paper gives weights for Earth-analog similarity; the weights in document 2 are derived from how these papers use each variable.

### What I read and how far in

Many publishers block automated access, so depth varies. Every card carries one of three tags, and you should treat them differently.

| Tag           | Meaning                                                                          | Count | How to use it                                                          |
|---------------|----------------------------------------------------------------------------------|-------|------------------------------------------------------------------------|
| \[FULL\]      | Full text or the paper's tables were opened                                      | 10    | Cite numbers directly                                                  |
| \[ABSTRACT\]  | Abstract or repository record only                                               | 19    | Numbers are right but thin; methods and tables still need a human read |
| \[SECONDARY\] | The paper itself was blocked; numbers come from another source that reports them | 3     | Check against the original before they go on the methodology site      |

#### Verdicts on the links you flagged

| Item                              | Verdict                                                                                                                                                                                                                                |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mashal 2025                       | **Real, and now read in full** from the PDF Yash supplied (Planetary Science Journal 6(7):166). It uses equal weights by design, so it does not supply Mars weights; it supplies eight criteria, three masks and a validation pattern. |
| Hayne 2015                        | Citation confirmed (Icarus 255:58–69). Abstract not readable from any free copy; the numbers on its card come from Sefton-Nash 2019.                                                                                                   |
| Speyerer & Robinson 2013          | Abstract not readable. Site values come from Gläser's comparison table, which is the better source anyway because it lines up all three studies.                                                                                       |
| Stoker 2011                       | The Cambridge link is too long for my tool. I read the same team's LPSC 2011 abstract (NTRS 20110014304), which has the soil tables.                                                                                                   |
| PISCES (NTRS 20070037451)         | Opened. It is a programme proposal with no measurements. Replace it with a Mauna Kea paper that has data.                                                                                                                              |
| Desert RATS (2013AcAau..90..332Y) | Opened in full via NTRS. It is about tools and operations and says nothing on why the site was chosen.                                                                                                                                 |
| Grant 2018                        | Abstract only. The engineering table (slopes, rocks, thermal inertia) is in the full paper; one secondary source prints the elevation cap with a sign I doubt.                                                                         |
| Morgan 2021                       | Abstract only (paywalled). Dataset list and scoring method need the full text or swim.psi.edu.                                                                                                                                         |

### The five questions, answered across all papers

#### 1. Which variables describe or choose a site?

| Family         | Moon papers                                              | Mars papers                                                                 | Analog papers (India, America)                                          |
|----------------|----------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Terrain        | Slope, relief, horizon elevation, height above ground    | Slope at 2–5 m and 84 m, relief over 1–1000 m, rock abundance, ellipse size | Crater diameter, depth, rim height, ejecta extent; dune slip-face angle |
| Light and heat | Share of time lit, longest darkness, maximum temperature | Latitude (as power and thermal proxy), thermal inertia                      | Altitude, UV dose, air and water temperature                            |
| Water          | Distance to PSR, ice detections, hydrogen                | Depth to ice, ice thickness, dataset agreement on ice                       | Precipitation, water activity, permafrost, brines                       |
| Material       | (almost absent)                                          | Clay, sulfate, carbonate minerals; dust cover; load-bearing surface         | Basalt target rock, clays, sulfates, jarosite, salts, grain size        |
| Operations     | Earth visibility, launch windows                         | Elevation (air to brake), landing accuracy, traverse radius                 | Access, services, contamination                                         |

#### 2. What numbers do they give for your targets?

| Target                                                    | Numbers                                                                                                                                     | Source                           |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| Connecting Ridge C1, lunar south pole (−89.45°, 222.84°E) | 76.0% lit at surface, 88.0% at 2 m; longest shadow 268 h at surface, 112 h at 2 m; nearest PSR 100 m away, slope 8.4° average               | Gläser 2018                      |
| Shackleton rim (−89.78°, 203.97°E)                        | 81.0% lit at surface, 85.5% at 2 m; longest shadow 221 h, 66 h at 2 m; PSR 370 m away                                                       | Gläser 2018                      |
| Artemis III regions                                       | 13 regions of about 15 × 15 km within 6° of the pole; 6.5 days of continuous light                                                          | NASA 2022                        |
| Lunar cold traps                                          | Ice where maximum temperature stays below 110 K; 38 K at depth in Cabeus; only 3.5% of cold traps show surface ice                          | Hayne, Li, Paige                 |
| South pole landing DEMs                                   | 5 m per pixel; slope error 1.5–2.5°; height error 0.3–0.5 m                                                                                 | Barker 2021                      |
| Jezero crater, Mars                                       | 18.4386°N, 77.5031°E; about 45 km across                                                                                                    | Grant 2018, Harris 2022          |
| Arcadia Planitia, Mars                                    | Ice under 30 cm in places, as shallow as 2.5 cm; top SWIM score                                                                             | Piqueux 2019, Morgan 2021        |
| Mid-latitude scarps, Mars                                 | Ice from 1–2 m depth, more than 100 m thick                                                                                                 | Dundas 2018                      |
| Mars landing gates                                        | Latitude ±30° (rovers), ±50° (humans); elevation below −1 km (MSL), −2.5 km (InSight), +2 km (humans); slopes under 15–30°; rocks under 10% | Golombek 2012, 2017; Bussey 2016 |
| Lonar crater                                              | 1.88 km across, 230–245 m deep, 40 m rim, ejecta to 1.41 km; clays formed at 130–200 °C; lake pH 9.5–10.5                                   | Maloof, Fudali, Hagerty, Harris  |
| Ladakh                                                    | 3,000–5,700 m; air density 0.64 kg/m³ at 5,300 m; UV-A 10× sea level; springs 50–85 °C                                                      | Pandey 2020, Ansari 2025         |
| MDRS Utah                                                 | Sulfate to 22,707 ppm; organics 0–0.6%; 1.7–13.6 million cells per gram                                                                     | Stoker 2011 (LPSC)               |

#### 3. Which variable matters more, and for what?

| Purpose                       | What the papers put first                             | Who says so                                            |
|-------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| Landing and driving safely    | Slope, then rocks and relief; on Mars also elevation  | Artemis III; Golombek 2012 and 2017; Bussey 2016       |
| Power and surviving the night | Illumination on the Moon; latitude on Mars            | Mazarico, Speyerer, Gläser; Golombek 2012              |
| Finding and using water       | Maximum temperature on the Moon; depth to ice on Mars | Hayne, Fisher, Li; Dundas, Piqueux, Morgan             |
| Science and life detection    | Mineralogy and water history                          | Grant 2018; Harris 2022; Hagerty; Bhattacharya; Pandey |
| Choosing an analog at all     | The single property needed for the stated use         | Foucher 2021; Lemelin 2021 ranks by goal               |

#### 4. Which dataset measured each variable?

| Variable                            | Moon                                                    | Mars                                                               | Earth equivalent for your tiles                                           |
|-------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------|
| Elevation, slope, relief, roughness | LOLA (240 m, 20 m, 5 m products)                        | MOLA; HRSC, CTX and HiRISE stereo locally                          | Copernicus DEM GLO-30                                                     |
| Illumination                        | LOLA horizon models; LROC WAC time series (about 100 m) | Computed from latitude                                             | Computed from latitude and DEM horizon                                    |
| Temperature                         | Diviner                                                 | Mars Climate Sounder, TES, THEMIS                                  | MODIS land-surface temperature; WorldClim                                 |
| Thermal inertia, rocks              | Diviner rock abundance                                  | TES and THEMIS thermal inertia                                     | Day-night temperature swing from MODIS or ECOSTRESS (a lead, not checked) |
| Ice                                 | LAMP (UV), M3 (near-IR), LOLA reflectance, neutron data | Odyssey neutron and gamma-ray, SHARAD radar, MCS and THEMIS trends | Permafrost maps (a lead, not checked); WorldClim aridity                  |
| Minerals                            | Diviner Christiansen feature, M3                        | CRISM, OMEGA, TES                                                  | ASTER Global Emissivity; EMIT mineral maps (a lead, not checked)          |

#### 5. What does Earth fail to match?

Almost no site-selection paper raises this, because that is not their subject. The analog papers do, and this is the content of your limits ledger.

| Property                    | Target value                                                   | Best Earth can do                                                                          | Status | Source                    |
|-----------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------|---------------------------|
| Gravity                     | 1.62 m/s² Moon, 3.71 Mars                                      | 9.81 everywhere; crater shape itself depends on gravity                                    | Broken | Fudali 1980; Foucher 2021 |
| Air pressure                | Vacuum on the Moon; under 1% of Earth on Mars                  | About half sea level on the highest roads (0.64 kg/m³ at 5,300 m)                          | Broken | Pandey 2020               |
| Cold traps                  | Below 110 K, down to 38 K                                      | About 184 K record low                                                                     | Broken | Hayne; Paige              |
| Polar lighting              | Sun within about 1.5° of the horizon all year, 18.6-year cycle | Low Sun in polar spring and autumn only                                                    | Partly | Mazarico; Gläser          |
| Radiation and UV            | No shielding                                                   | UV-A 10× sea level in Ladakh, which overshoots present Mars; no particle radiation         | Partly | Pandey; Harris            |
| Regolith                    | Unweathered, glass-welded, dry                                 | Ash and basalt that are weathered, oxidised, damp                                          | Partly | McKay 1993; Foucher       |
| Rock chemistry              | Iron-rich Martian basalt; lunar anorthosite                    | Basalt with 45–52% silica; anorthosite is rare                                             | Partly | Foucher 2021              |
| Oxidants                    | 0.5–1 wt% perchlorate on Mars                                  | Up to 0.6 wt% in the Atacama, far less elsewhere                                           | Partly | Harris 2022               |
| Sterility                   | No known life                                                  | 1.7–13.6 million cells per gram even in desert cores; farms at Lonar; litter at Chumathang | Broken | Stoker; Harris; Pandey    |
| Line of sight to Earth, GPS | Site criterion on the Moon; no GPS anywhere                    | Always available                                                                           | Broken | Artemis III; Young 2013   |

### Paper cards: Moon (Shiven)

#### Mazarico et al. 2011, Icarus 211(2). Illumination conditions of the lunar polar regions using LOLA topography \[ABSTRACT\]

- **1 Variables:** Average illumination fraction; longest continuous sunlight and darkness; horizon elevation; height gain above ground; permanently shadowed (PSR) area.
- **2 Numbers:** PSR area 12,866 km² north, 16,055 km² south. A point near Shackleton rim is sunlit about 240 days a year, longest darkness about 1.5 days. 240 m polar DEMs to 75° latitude, illumination computed to 80°, 6 h steps over several 18.6-year cycles. Gläser's table quotes its site values as 78.1–89.0% average illumination.
- **3 What matters:** Illumination is the single driver, for solar power. A height gain of roughly 10 m above the surface improves it substantially.
- **4 Datasets:** LOLA altimetry on LRO for topography; LRO Wide-Angle Camera images to validate.
- **5 Earth cannot:** Not discussed. Implied: an 18.6-year lighting cycle with the Sun never more than about 1.5° above the horizon has no Earth equivalent.

#### Speyerer & Robinson 2013, Icarus 222(1):122–136. Persistently illuminated regions at the lunar poles \[SECONDARY\]

- **1 Variables:** Percent of time illuminated; longest eclipse; combined illumination of nearby stations.
- **2 Numbers:** From Gläser's comparison table: Hinshelwood 37.0%, Peary 44.7%, Whipple 63.3%, Connecting Ridge 64.2%, Shackleton 71.6% surface illumination; longest shadow 145–650 h. Via Wikipedia citing this paper: some Shackleton-rim points lit 94% of a lunar year; three linked outposts eclipsed together for at most 43 h. LROC processed 3,548 north and 3,807 south WAC images.
- **3 What matters:** Illumination, for power and thermal stability. Linking several nearby sites matters more than one best point.
- **4 Datasets:** LROC Wide-Angle Camera image time series (maps at about 100 m per pixel).
- **5 Earth cannot:** Abstract not opened (Elsevier blocked). Nothing recorded on analog limits.

#### Gläser et al. 2018, Planet. Space Sci. 162. Illumination conditions at the lunar poles: implications for future exploration \[FULL\]

- **1 Variables:** Average illumination at surface and at 2 m; solar visibility; maximum time in shadow; area of the lit cluster; distance, area and slope of the nearest PSR.
- **2 Numbers:** Six sites. Surface illumination 69.5% (Peary) to 82.9% (Whipple); at 2 m, 77.1% to 88.0% (Connecting Ridge C1, −89.45°, 222.84°E). Max shadow 101 h (Whipple) to 392 h (Hinshelwood) at surface; 66 h at Shackleton at 2 m. Nearest PSR 40–680 m away with slopes of 4–12° average, all below the 21.8° rover limit the paper cites. 20 m DTM, hourly steps 2017–2037.
- **3 What matters:** Illumination first (power), PSR proximity second (science). States that average illumination shows intensity, not duration, so max shadow time must be read alongside it.
- **4 Datasets:** LOLA only: 400 × 400 km DTMs at 20 m per pixel.
- **5 Earth cannot:** Not discussed. Limits of its own data: lit areas are only 20–40 m wide; 23–33% of pixels are interpolated; values sit between Mazarico (240 m, smoothed, higher) and Speyerer (100 m images, lower).

#### Barker et al. 2021, Planet. Space Sci. 203:105119. Improved LOLA elevation maps for south pole landing sites \[ABSTRACT\]

- **1 Variables:** Elevation, slope, and their uncertainty; horizon elevation; predicted illumination.
- **2 Numbers:** Four sites, each 16 × 16 km, at 5 m per pixel. Height uncertainty about 0.30–0.50 m RMS; slope uncertainty about 1.5–2.5° RMS; track geolocation error cut more than tenfold to 10–20 cm horizontal, 2–4 cm vertical.
- **3 What matters:** Slope and height error matter because they propagate into illumination predictions and landing safety.
- **4 Datasets:** LOLA tracks adjusted to a LOLA-based DEM. Imagery-based models are noted as less well controlled and hindered by shadow.
- **5 Earth cannot:** Not discussed. Useful for you as the published error to feed the Monte Carlo (slope ±1.5–2.5°).

#### NASA, Aug 2022. Artemis III: 13 candidate landing regions \[FULL\]

- **1 Variables:** Terrain slope (safe landing); lighting; ease of communication with Earth; launch-window access; distance to a PSR within moonwalk range.
- **2 Numbers:** 13 regions, each about 15 × 15 km, all within 6° of latitude of the south pole; landing sites about 100 m radius; every region has sites with continuous sunlight for a 6.5-day surface stay.
- **3 What matters:** No ranking given. Slope is framed as the safety criterion, lighting as power and thermal, PSR proximity as the science goal.
- **4 Datasets:** Lunar Reconnaissance Orbiter data plus published lunar science.
- **5 Earth cannot:** Not discussed. Direct line of sight to Earth for communications is a criterion with no Earth analog.

#### Hayne et al. 2015, Icarus 255:58–69. Evidence for exposed water ice in the Moon's south polar regions \[SECONDARY\]

- **1 Variables:** Maximum surface temperature; far-UV albedo and band ratio.
- **2 Numbers:** As described by Sefton-Nash et al. 2019: ice signature where temperature never exceeds 110 K; off-band/on-band ratio above 1.2 with Lyman-alpha albedo below 0.03; possible water content 0.1–2.0% by mass.
- **3 What matters:** Maximum temperature is the control: below about 110 K surface ice can persist.
- **4 Datasets:** LAMP far-UV spectrograph and Diviner radiometer, both on LRO.
- **5 Earth cannot:** Abstract not opened. Earth's record low is about 184 K, far above 110 K, so cold-trap conditions cannot be matched.

#### Fisher et al. 2017, Icarus 292:74–85. Evidence for surface water ice in the lunar polar regions (LOLA reflectance + Diviner) \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Maximum temperature; 1064 nm reflectance; slope used as a filter.
- **2 Numbers:** Reflectance rises sharply as temperature drops through about 110 K, only within 5° of the south pole; result holds when limited to slopes under 10°; other steps near 200 K and possibly 300 K.
- **3 What matters:** Temperature again. Slope is used to exclude false positives from mass wasting.
- **4 Datasets:** LOLA 1064 nm reflectance; Diviner temperature.
- **5 Earth cannot:** Not discussed.

#### Li et al. 2018, PNAS 115(36):8907–8912. Direct evidence of surface exposed water ice in the lunar polar regions \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Near-infrared ice absorptions; annual maximum temperature; latitude.
- **2 Numbers:** Ice within 20° of both poles; most ice-bearing spots have annual maximum temperature below 110 K; only about 3.5% of cold traps show ice; some pixels may hold about 30 wt% ice.
- **3 What matters:** Temperature is necessary but not sufficient: most cold traps show no surface ice.
- **4 Datasets:** Moon Mineralogy Mapper (M3) on Chandrayaan-1, cross-checked with LOLA reflectance and LAMP UV.
- **5 Earth cannot:** Detects only the top few millimetres. Nothing on analogs.

#### Paige et al. 2010, Science 330:479–482. Diviner observations of cold traps in the Moon's south polar region \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Surface and subsurface temperature.
- **2 Numbers:** Subsurface temperature at the LCROSS impact site estimated at 38 K; cryogenic regions extend beyond permanent shadow.
- **3 What matters:** Temperature defines where volatiles can be trapped.
- **4 Datasets:** Diviner radiometer; LCROSS ejecta plume composition.
- **5 Earth cannot:** Not discussed. 38 K is unreachable in any natural Earth setting.

#### Lemelin et al. 2021, Planet. Sci. J. 2(3). Framework for coordinated exploration of volatiles in the south polar region \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Temperature regime; slope; hydrogen content; number of ice detections; depth stability of water ice and dry ice; mobility (traverse distance).
- **2 Numbers:** 169 ice-bearing PSRs, 37 sites of interest, 11 meeting more than one goal, all within 6° of the pole; case traverses of 20–50 km, about 100 km and about 300 km.
- **3 What matters:** Ranks by purpose: highest volatile abundance, mapping volatile distribution, or fastest useful sample. Different purposes give different rankings. Thresholds are in the full paper, which I did not open.
- **4 Datasets:** M3 ice detections plus temperature, slope and hydrogen maps.
- **5 Earth cannot:** Not discussed.

### Paper cards: Mars (Vihaan)

#### Morgan et al. 2021, Nature Astronomy 5:230–236. Availability of subsurface water-ice resources in the northern mid-latitudes of Mars (SWIM) \[ABSTRACT\]

- **1 Variables:** Consistency of several independent datasets with buried ice: neutron data, thermal behaviour, radar surface and subsurface returns, geomorphology.
- **2 Numbers:** Abstract gives no thresholds. Best regions: Arcadia Planitia and the Deuteronilus Mensae glacial networks, both equatorward of today's ice-stability zone.
- **3 What matters:** Agreement between datasets is the score. Mashal 2025 describes the SWIM consistency index as running from −0.5 to 1. How the datasets are combined still needs the Methods.
- **4 Datasets:** Mars Odyssey neutron spectrometer, thermal inertia from TES/THEMIS, SHARAD radar, image-based geomorphology. Maps at swim.psi.edu.
- **5 Earth cannot:** Not discussed.

#### Dundas et al. 2018, Science 359(6372):199–201. Exposed subsurface ice sheets in the Martian mid-latitudes \[ABSTRACT\]

- **1 Variables:** Depth to the top of ice; ice thickness; scarp exposure.
- **2 Numbers:** 8 scarps; ice starts as shallow as 1–2 m below the surface and is more than 100 m thick.
- **3 What matters:** Depth of ice matters most, for access by a mission.
- **4 Datasets:** Not named in the abstract (HiRISE imaging and CRISM spectra in the paper, from memory).
- **5 Earth cannot:** Not discussed. Earth's ground ice sits in wet, vegetated permafrost; Mars ice sublimates directly to vapour.

#### Piqueux et al. 2019, Geophys. Res. Lett. Widespread shallow water ice on Mars at high latitudes and midlatitudes \[ABSTRACT\]

- **1 Variables:** Depth to the ice table; thermal inertia of the dry layer above it; latitude; elevation.
- **2 Numbers:** Ice as low as 35°N and 45°S, sometimes a few centimetres down (NASA release: as little as 2.5 cm; under 30 cm across Arcadia Planitia). Maps at about 3 pixels per degree; released grid is 2 per degree.
- **3 What matters:** Ice depth for ISRU. Northern mid-latitudes are preferred because lower elevation means more air for landing and more sun than the poles. Very fine dust is flagged as a place a lander could sink.
- **4 Datasets:** Mars Climate Sounder (MRO) and THEMIS (Odyssey) seasonal temperatures; gamma-ray spectrometer, radar and fresh craters as checks.
- **5 Earth cannot:** Not discussed.

#### Grant et al. 2018, Planet. Space Sci. 164:106–126. The science process for selecting the landing site for the 2020 Mars rover \[ABSTRACT\]

- **1 Variables:** Elevation and latitude (engineering); habitability, biosignature preservation and sample value (science).
- **2 Numbers:** More than 30 sites narrowed to Jezero (18.4386°N, 77.5031°E), NE Syrtis (17.8899°N, 77.1599°E), Columbia Hills (14.5478°S, 175.6255°E). A secondary source (Chojnacki 2018) gives the constraints as within ±30° latitude, thermal inertia above 100–150, ellipse about 12 × 10 km, and prints the elevation cap as +0.5 km; I recall −0.5 km, so check the sign in the full paper.
- **3 What matters:** Engineering constraints act as gates. Once passed, sites were ranked mainly on science merit, because new landing technology relaxed the constraints.
- **4 Datasets:** MOLA elevation; orbital imaging and spectroscopy for science.
- **5 Earth cannot:** Not discussed.

#### Harris et al. 2022, Front. Astron. Space Sci. 9:849078. Remote and in-situ characterization of Mars analogs \[FULL\]

- **1 Variables:** Mean annual temperature and precipitation; pH; salinity; redox (Eh); water activity; mineralogy; size of features against instrument resolution.
- **2 Numbers:** Five analogs. Lake Bonney −17.1 °C; Haughton −16.4 °C, under 13 mm rain, 23 km crater; Rio Tinto pH 2–2.5; Lonar 26 °C, 752 mm, pH 9.5–10.5; Atacama salars 1–14 °C, perchlorate up to 0.6 wt% against 0.5–1 wt% on Mars. Water activity limit for life 0.6.
- **3 What matters:** Water availability controls habitability; mineralogy (clays, salts, iron oxides) controls biomarker preservation and what orbit can see.
- **4 Datasets:** CRISM, OMEGA, MRO, Odyssey; Curiosity and Perseverance instruments; field and lab spectra.
- **5 Earth cannot:** No analog is perfect. Mars has no plate tectonics, stronger UV and perchlorate oxidants. Lonar is confounded by farming and people; Lake Bonney is the only site free of plants and human influence.

#### Mashal et al. 2025, Planet. Sci. J. 6(7):166. Multi-criteria overlay analysis for identifying preferred exploration zones on Mars \[FULL\]

- **1 Variables:** Eight criteria from the 2015 human landing-site workshop: C1 biosignature-preserving deposits, C2 Noachian or Hesperian rocks, C3 two regional units for dating, C4 outcrops showing water activity, C5 stratigraphic contacts, C6 water feedstock, C7 metal and silicon feedstock, C8 low latitude for steady light. Three masks: latitude, elevation, thermal inertia.
- **2 Numbers:** Masks: beyond ±50° latitude, above 2 km elevation, thermal inertia under 100. All layers resampled to about 3 km and scaled 0–100; vector layers buffered by 100 km. 23 regions exceed a z-score of 3 (above 78% suitability): 14 at 80–90%, 9 above 90%. Of 47 sites proposed at the workshop, 43 score above 50%, 17 above 80%, 4 above 90%.
- **3 What matters:** No criterion is weighted above another: the paper states layers were 'treated equally without preassigned weights' and the score is the plain mean of the eight, times the mask. In effect six of eight criteria are geology and mineralogy, one is water, one is sunlight; engineering acts only as a mask.
- **4 Datasets:** Mineral maps from CRISM and OMEGA; global geologic map; olivine and pyroxene maps; SWIM ice consistency at 0–1 m, 1–5 m and over 5 m; TES thermal inertia (3 km, 25–600 units); MOLA-HRSC elevation (200 m).
- **5 Earth cannot:** Not discussed; no Earth analogs in the paper. Own limits: results favour well-mapped regions, global mineral maps risk false positives, and sparse data in the northern lowlands lowers scores there.

#### Golombek et al. 2012, Space Sci. Rev. 170:641–737. Selection of the Mars Science Laboratory landing site \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Latitude; elevation; relief; slope; rock abundance; radar reflectivity; load-bearing, dust-free surface; science diversity.
- **2 Numbers:** Latitude within ±30°; elevation below −1 km; relief under 100–130 m over 1–1000 m baselines; slopes under 30° at 2–5 m; more than 50 sites of 25 × 20 km narrowed to four.
- **3 What matters:** Each constraint is tied to a purpose: latitude for thermal control, elevation for atmosphere to brake, relief for descent fuel, slope for touchdown stability, rocks for the belly pan. Science chose among survivors.
- **4 Datasets:** All active orbiters, mainly Mars Reconnaissance Orbiter.
- **5 Earth cannot:** Not discussed.

#### Golombek et al. 2017, Space Sci. Rev. 211:5–95. Selection of the InSight landing site \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Elevation; latitude; ellipse size; slope; rock abundance; regolith thickness; dust.
- **2 Numbers:** Elevation at or below −2.5 km; latitude 15°S–5°N, later 3–5°N; ellipse 130 × 27 km; slopes under 15° at 84 m and 2 m; rock abundance under 10%; about 5 m of broken-up regolith.
- **3 What matters:** Engineering only. Science did not influence the choice, unlike other landers.
- **4 Datasets:** CTX, HiRISE, THEMIS, HRSC, radar.
- **5 Earth cannot:** Not discussed.

#### Bussey & Hoffman 2016, IEEE Aerospace. Human Mars landing site and impacts on Mars surface operations \[FULL\] \[ADDED\]

- **1 Variables:** Latitude; elevation; exploration-zone radius; slope and hazard size; access to water ice; science regions of interest.
- **2 Numbers:** Latitude within ±50°; elevation below +2 km; zone radius about 100 km; land within 100 m; avoid 5° slopes and 30 cm hazards; stays of 300–500 sols.
- **3 What matters:** States a trade: low latitude is easier for landing and ascent, high latitude gives water ice.
- **4 Datasets:** MOLA; says new orbital data are needed.
- **5 Earth cannot:** Not discussed.

### Paper cards: India (Shaurya)

#### Maloof et al. 2010, GSA Bulletin 122(1–2):109–126. Geology of Lonar Crater, India \[ABSTRACT\]

- **1 Variables:** Crater diameter; rim structure; ejecta blanket extent and profile; target rock.
- **2 Numbers:** 1.88 km simple crater in Deccan basalt; rim-fold hinge preserved around 10–15% of the rim; mapped on a metre-scale DEM. Shaurya's full-text read adds (not checked by me): about 135 m from rim to lake, ejecta up to about 7 m thick, DEM built from 64,000 GPS points.
- **3 What matters:** Target rock (basalt) and ejecta form make it an analog for small craters on basaltic surfaces of the Moon and for fluidized ejecta on Mars.
- **4 Datasets:** Metre-scale DEM, field mapping, radiocarbon dating, paleomagnetism.
- **5 Earth cannot:** Abstract does not say. Lonar holds a lake and soils, which no lunar or present Martian crater does.

#### Hagerty & Newsom 2003, Meteoritics & Planet. Sci. 38(3):365–381. Hydrothermal alteration at the Lonar Lake impact structure \[ABSTRACT\]

- **1 Variables:** Alteration minerals; formation temperature; crater size; target rock.
- **2 Numbers:** About 1.8 km crater, about 50,000 years old, in about 65 Ma basalt; saponite with minor celadonite and carbonate; clays formed at 130–200 °C.
- **3 What matters:** Mineralogy matters, as evidence of impact-driven hydrothermal systems; Lonar marks a lower size limit for them.
- **4 Datasets:** Drill cores; electron microprobe; X-ray diffraction; thermodynamic modelling.
- **5 Earth cannot:** Depends on liquid water being present, which is an assumption for Mars. One of only two known Earth craters in basalt.

#### Pandey et al. 2020, Int. J. Astrobiology 19(1):78–98. Ladakh: diverse, high-altitude extreme environments for off-Earth analogue research \[FULL\]

- **1 Variables:** Elevation; air density; UV dose; temperature; pH; landform and regolith type; permafrost; salts and minerals.
- **2 Numbers:** Sites at about 3,000–5,700 m: Khardung La 5,359 m, Taglang La 5,328 m, Puga 4,414 m. Air density at 5,300 m about 0.64 kg/m³ against 1.2 at sea level and at least 0.4 estimated for Hesperian Mars. UV-A about 10× sea level. Hot springs 60–75 °C; dune slip faces up to 35°. Shaurya adds from Table 1 (not visible to me): Tso Kar 4,535 m, Chumathang 3,944 m, Panamik 3,206 m, Hunder 3,095 m.
- **3 What matters:** Clays and brines are named the most Mars-relevant targets, for biomarker preservation. The Moon is not mentioned.
- **4 Datasets:** Field measurements, ground photos and Google Earth mapping, HABIT instrument prototype.
- **5 Earth cannot:** Air is still denser than early Mars; UV-A exceeds present Mars; life present is Earth life; dune ages unknown; some springs contaminated by plastic and seepage.

#### Fudali et al. 1980, The Moon and the Planets. Morphology of Lonar Crater, India: comparisons and implications \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Diameter; depth; rim height; circularity; ejecta extent.
- **2 Numbers:** Pre-erosion diameter 1,710 m; depth 230–245 m; rim height 40 m; circularity 0.9; continuous ejecta to 1,410 m beyond the rim.
- **3 What matters:** Crater shape ratios, to compare across bodies. Concludes crater shape depends on gravity.
- **4 Datasets:** Five drill holes, a gravity survey, field mapping.
- **5 Earth cannot:** Fresh Earth craters are shallower with higher rims than lunar craters of the same size, so crater relief on Earth does not transfer one-to-one.

#### Bhattacharya et al. 2016, J. Geophys. Res. Planets 121(3):402–431. Jarosite in the Deccan Volcanic Province of Kachchh \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Sulfate and clay mineral assemblage; host rock; stratigraphic order of minerals.
- **2 Numbers:** Matanumadh, Kachchh. Spectra over 350–2500 nm and 4000–400 cm⁻¹. Jarosite and kaolinite above an iron-magnesium smectite zone in altered basalt. A 2025 follow-up (Ray et al., American Mineralogist) measures Fe³⁺ at about 0.83 of total iron.
- **3 What matters:** Mineral assemblage in altered basalt, as a marker of acidic, oxidizing water on early Mars.
- **4 Datasets:** Visible/near-infrared and mid-infrared spectroscopy, X-ray diffraction on field samples.
- **5 Earth cannot:** Not discussed in the abstract.

#### Ansari et al. 2025, Prog. Earth Planet. Sci. 12:49. Organics in high-altitude hot springs of Ladakh \[FULL\] \[ADDED\]

- **1 Variables:** Water temperature; pH; dissolved solids; salinity; altitude; organic compound classes.
- **2 Numbers:** Four springs (Chumathang, Panamik, Changlung, Puga); 50.4–84.9 °C; pH 7.01–8.08; 56–68 compounds per site; 27 bioactive compounds forming 38–98% of dissolved organics.
- **3 What matters:** Altitude: high-altitude springs carried more bioactive compounds than low-altitude ones.
- **4 Datasets:** Field multiparameter probe; GC-MS-MS.
- **5 Earth cannot:** Does not list differences from Mars and does not test survival under Mars conditions. Four springs only.

### Paper cards: America (Shaurya)

#### Stoker et al. 2011, Int. J. Astrobiology 10. Subsurface soil cores from Mars Desert Research Station, Utah \[SECONDARY\]

- **1 Variables:** Mineralogy; soluble ions; organic content; cell density; depth.
- **2 Numbers:** From the same team's LPSC 2011 abstract (paper 1231): cores to about 1 m; sulfate dominates every soil (up to 22,707 ppm); clays montmorillonite, illite, nontronite; organics 0–0.6%; 1.7–13.6 million cells per gram.
- **3 What matters:** Mineralogy: clay-plus-sulfate soils are what make MDRS relevant to Mars landing sites. Cell count did not track organics or chemistry.
- **4 Datasets:** Field X-ray diffraction (a commercial CheMin), wet chemistry, lipid analysis, PCR.
- **5 Earth cannot:** Journal paper not opened (link too long for my fetch tool). Soils are inhabited: millions of cells per gram.

#### Howell, Henley & Schowengerdt, about 2007 (NASA NTRS). PISCES: a stepping stone to international space exploration \[FULL\]

- **1 Variables:** Altitude; dryness; solar radiation; terrain resemblance; fine volcanic ash; access and services.
- **2 Numbers:** Candidate area about 9,000 ft; Apollo training valley about 11,000 ft; robotic test range over 1 km with 20–30° slopes and small craters.
- **3 What matters:** None stated. This is a programme proposal, with no measurements.
- **4 Datasets:** None.
- **5 Earth cannot:** Lists no shortfalls. Weak evidence: use it only as a record that Mauna Kea is an accepted lunar analog, and replace it with a paper that has data.

#### Young et al. 2013, Acta Astronautica 90:332. Tools and technologies for planetary field geology on EVA: 2010 Desert RATS \[FULL\]

- **1 Variables:** Not a site-selection paper. It describes operations: traverse length, crew, tools, communications.
- **2 Numbers:** San Francisco volcanic field, Arizona; two 7-day traverses; two pressurized rover prototypes; eight crew. Apollo traverses for comparison: 3.5, 28 and 35 km.
- **3 What matters:** None on site variables.
- **4 Datasets:** Field test records.
- **5 Earth cannot:** Shirt-sleeve backpack in place of a pressure suit; GPS used although none exists off Earth; acid tests dropped because low pressure changes the reaction.

#### Navarro-González et al. 2003, Science 302:1018–1021. Mars-like soils in the Atacama Desert and the dry limit of microbial life \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Organic content; culturable bacteria; recoverable DNA; soil oxidant activity.
- **2 Numbers:** Organics only at trace levels; culturable bacteria extremely low; no DNA recovered from two samples in the driest zone. (Chile, so South America.)
- **3 What matters:** Aridity is the control: it sets the dry limit of life and makes soil chemistry behave like the Viking results.
- **4 Datasets:** Soil sampling and Viking-style incubation experiments.
- **5 Earth cannot:** Not discussed in the abstract.

#### McKay et al. 1993, Lunar Planet. Sci. Conf. 24. JSC-1: a new lunar regolith simulant \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Bulk chemistry; mineralogy; glass content; grain-size distribution.
- **2 Numbers:** Glass-rich basaltic ash ground to a lunar-like grain size; produced in lots above 907 kg. Source is a cinder cone in the San Francisco volcanic field (from memory).
- **3 What matters:** Depends on use: geotechnical work needs grain size; resource work needs chemistry.
- **4 Datasets:** Laboratory characterisation.
- **5 Earth cannot:** Only 'approximates' some lunar soils. Earth ash carries water and oxidised iron and lacks the impact-welded glass particles of real regolith.

#### Skinner & Fortezzo 2013, Acta Astronautica 90(2):242–253. Photogeologic mapping in traverse planning: Desert RATS 2010 \[ABSTRACT\] \[ADDED\]

- **1 Variables:** Geologic units mapped from orbital-style imagery.
- **2 Numbers:** 1:24,000 map of the north-east San Francisco volcanic field; 4 surficial, 5 volcanic and 5 basement units; 62 recommended stops.
- **3 What matters:** Geologic diversity within driving range is what makes a traverse site useful.
- **4 Datasets:** Base maps standing in for precursor orbital data.
- **5 Earth cannot:** Not discussed.

### Paper card: cross-cutting

#### Foucher et al. 2021, Planet. Space Sci. 197:105162. Definition and use of functional analogues in planetary exploration \[FULL\] \[ADDED\]

- **1 Variables:** Classes of analog site: landform, process, rock and mineral, astrobiological, engineering.
- **2 Numbers:** Earth basalt spans 45–52 wt% silica and 0.1–5 wt% alkalis, so the word 'basalt' alone does not define an analog.
- **3 What matters:** The property that matters is the one needed for the stated use. Partial analogs are fine for most testing.
- **4 Datasets:** Review.
- **5 Earth cannot:** States that a perfect analog does not exist. Atmosphere, pressure, gravity and radiation are hard or impossible to reproduce; Earth rocks are weathered, oxidised or biologically altered; Martian basalt is richer in iron; lunar anorthosite is far more abundant than on Earth.

### Review of the team's research

#### Vihaan (Mars), reviewed 10 Nov

The notes summarise Morgan 2021, Grant 2018 and Harris 2022. Everything I could check is right; four numbers sit behind a paywall and need a second look before they are quoted.

| Claim                                                                                                              | Verdict        | Note                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------|
| Arcadia Planitia and Deuteronilus Mensae match the most remote-sensing criteria for ice                            | \[CORRECT\]    | Matches the Morgan 2021 abstract.                                                                           |
| Five datasets combined: neutron, thermal, radar surface, radar subsurface, geomorphology                           | \[CORRECT\]    | Matches the figures and reference list. This is the SWIM method.                                            |
| Ice at the poles is plentiful but landing there is hard; mid-latitude ice is the goal                              | \[CORRECT\]    | Same trade stated by Bussey & Hoffman 2016 and the NASA release on Piqueux 2019.                            |
| Mars 2020 priority list (science merit first, then setting, habitability, biosignature preservation)               | \[CORRECT\]    | Consistent with the Grant 2018 abstract: sites ranked mainly on science once engineering gates were passed. |
| Gale lakes up to 5,800 km²; minerals to match are Mg-carbonates, Fe/Mg clays, amorphous silica, hematite, sulfates | \[CORRECT\]    | Matches Harris 2022. It is an area, so write 'as large as', not 'as long as'.                               |
| Haughton, McMurdo Dry Valleys, Rio Tinto (95 km, pH 2–2.5), Lonar and its farming caveat                           | \[CORRECT\]    | Matches Harris 2022 Table 1 and text.                                                                       |
| Jezero at 18.5°N, 77.4°E                                                                                           | \[PARTLY\]     | Grant 2018 gives 18.4386°N, 77.5031°E. Use that.                                                            |
| Landing altitude below 1,380 m; roughness under 100 m; boulders under 1 m                                          | \[UNVERIFIED\] | Paywalled Methods. The sentence in the notes is garbled; re-read and copy it exactly with units.            |
| Mapping covers up to 60°N, 225°E to 290°E and beyond                                                               | \[UNVERIFIED\] | Plausible for SWIM; not in the abstract.                                                                    |
| Glacial deposits up to 1.5 km thick; ice maps to 3 m depth                                                         | \[UNVERIFIED\] | Not in anything I opened. Dundas 2018 gives more than 100 m for scarp ice.                                  |
| 'Refuel aircraft'                                                                                                  | \[WRONG\]      | Spacecraft. Small, but fix it before it reaches the methodology site.                                       |

#### Shiven (Moon), reviewed 10 Nov: what holds up

| Claim                                                                          | Verdict        | Note                                                                                                                                                      |
|--------------------------------------------------------------------------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Chang'E-7 site study caps slope at 8°                                          | \[CORRECT\]    | Jia et al. 2022, Remote Sensing 14:4863. It also uses PSR distance (within 8 km), rocks, illumination and maximum temperature.                            |
| Ice stable below about 110 K                                                   | \[CORRECT\]    | Hayne, Fisher, Li.                                                                                                                                        |
| Day-night range about 100–400 K                                                | \[CORRECT\]    | Right for the equator. It does not describe the polar targets.                                                                                            |
| Connecting Ridge lit 92.27% of the time at 2 m                                 | \[PARTLY\]     | This is Gläser 2014. Gläser 2018, the paper on your list, gives 88.0%. The table's '\> 90% continuous sun' is wrong: it is share of time, not continuous. |
| LCROSS water 5.6 wt%                                                           | \[PARTLY\]     | The number is the plume estimate for Cabeus regolith (from memory, 5.6 ± 2.9%). 'Among ejected volatiles' misstates it.                                   |
| HLS must land at 0–8°, goal 0–5°                                               | \[UNVERIFIED\] | Source is paywalled. Plausible and consistent with the Chang'E-7 limit; confirm before citing.                                                            |
| Area below 110 K in the south is 1.30 × 10⁴ km²                                | \[UNVERIFIED\] | Cited to Williams 2019; I could not open it.                                                                                                              |
| Regolith grain size, density 1,100 to 1,800 kg/m³, Chang'E-6 permittivity 2.22 | \[UNVERIFIED\] | Plausible. None of these can be measured for Earth tiles from orbit, so they go in the limits ledger.                                                     |
| Environmental processes get zero weight because Earth cannot match them        | \[CORRECT\]    | Same decision as the limits ledger.                                                                                                                       |
| Four-stage method: ingest, mask, score, zoom                                   | \[CORRECT\]    | Same pipeline as ours. Its NDVI mask idea is adopted.                                                                                                     |

#### Shiven (Moon): what does not hold up

| Claim                                                                                                    | Verdict        | Why                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The weight table (25, 20, 15 ...)                                                                        | \[WRONG\]      | The human column adds to 185% and the robotic column to 195%. No paper is cited for any value. The text says slope gets a 'slightly lower' weight for robots and then gives 30 against 25. |
| Barker 2021 used LOLA reflectance to quantify roughness                                                  | \[WRONG\]      | Barker 2021 builds 5 m elevation models and their error estimates. No reflectance.                                                                                                         |
| Rock abundance under 20% from the Chang'E-7 paper                                                        | \[WRONG\]      | That paper sets no rock threshold. InSight used under 10% on Mars (Golombek 2017).                                                                                                         |
| Sun visibility above 35%, Earth visibility above 15%                                                     | \[UNVERIFIED\] | The cited paper (Liu 2025) shows no such cut-offs in the text I could read.                                                                                                                |
| Good illumination geometry enables communication with Earth                                              | \[WRONG\]      | They are separate. Earth visibility depends on the near side and local horizon.                                                                                                            |
| Chandrayaan-3 measured 290–320 K; 'Chang'e-3' thermal inertia cited to the same paper                    | \[UNVERIFIED\] | The two missions are mixed up. I recall ChaSTE reporting about 355 K on its Sun-facing slope; check the paper.                                                                             |
| Manicouagan melt sheet: flat, boulder-free, sparse vegetation, at 52.8°N                                 | \[WRONG\]      | The structure is at about 51.1°N, 68.7°W, ringed by a 1,942 km² reservoir. The island is boreal forest (my knowledge), so the report's own NDVI mask removes it.                           |
| Tatacoa is hyper-arid                                                                                    | \[WRONG\]      | It is a tropical dry forest, at 3.23°N, 75.17°W.                                                                                                                                           |
| Samoylov Island as lunar polar analog                                                                    | \[WRONG\]      | Wet, vegetated, organic-rich tundra. It fails the vegetation gate, and lunar ice does not form thermokarst.                                                                                |
| Dallol for lunar ISRU                                                                                    | \[WRONG\]      | Acid brines and sulfates are a Mars analog. The Moon has neither, and oxygen-extraction reactors do not run in acid.                                                                       |
| Kolar Gold Fields: cool, stable, good for habitats                                                       | \[WRONG\]      | Rock at 3 km depth is hot, and the mines closed in 2001 (my knowledge). Lava tubes are also outside what a surface raster index can score.                                                 |
| Gosses Bluff at 24.1°S, 132.8°E, granite and shale                                                       | \[PARTLY\]     | Age 142.5 Ma, 22 km original rim and 5 km uplift are right. Coordinates are off (about 23.8°S, 132.3°E from memory) and the rock is mainly sandstone.                                      |
| Field vacuum trench at 10⁻¹⁰ Pa; argon chamber to 'approximate low pressure'; dust lofting by convection | \[WRONG\]      | That vacuum needs a laboratory chamber. Argon at one atmosphere is not low pressure. Convection is an atmosphere effect.                                                                   |

**Why the pattern matters.** Both of Shiven's files read like AI research reports: confident prose, plausible numbers, about 160 numbered citations of which many are blogs, and sites chosen by description with no data behind them. That is the failure the 9 Nov plan warned about. The fix is the one GSOP is built on: a site enters the list only after the index scores it.

#### Shaurya (India and America), reviewed 11 Nov

Six papers, each answered against the five questions. It agrees with the cards in this document on every point both of us could read, and it goes further into the full text of two papers. Nothing in it changes a weight. Three citations need fixing, and the two or three extra papers per region were not added; the ones I added are on the cards above.

| Paper                    | Verdict     | Note                                                                                                                                                                                                                                                                                                                    |
|--------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Maloof 2010 (Lonar)      | \[CORRECT\] | 1.88 km, basalt target, rim fold on 10–15% of the rim and the Mars ejecta comparison match the abstract. The extra numbers (135 m rim to lake, ejecta to about 7 m, 64,000 GPS points) are beyond what I could open; they read as full-text values. Note Fudali 1980 gives 230–245 m rim to floor, a different measure. |
| Hagerty & Newsom 2003    | \[PARTLY\]  | Numbers right (1.8 km, about 50,000 years, 130–200 °C, saponite). The title is wrong: it is 'Hydrothermal alteration at the Lonar Lake impact structure, India: Implications for impact cratering on Mars'.                                                                                                             |
| Pandey 2020 (Ladakh)     | \[PARTLY\]  | Altitudes, UV-A at 10× sea level, air density, spring temperatures and pH all match. The citation is wrong: volume 19(1), pages 78–98, not 19(4):286–311. Site altitudes and biomass values from the tables are plausible but were not visible to me.                                                                   |
| Stoker 2011 (MDRS)       | \[CORRECT\] | Matches the source, which is the LPSC 2011 conference abstract, as his heading says. The journal paper on the original list is still unread.                                                                                                                                                                            |
| Howell 2007 (PISCES)     | \[PARTLY\]  | Slopes of 20–30° and the 1 km test range match. The NTRS title is 'PISCES: A Stepping Stone to International Space Exploration and Development'. The high-vacuum facility and the mid-2010s timeline were not in the copy I read, which targets a full facility by 2011. He rightly calls it a programme paper.         |
| Young 2013 (Desert RATS) | \[CORRECT\] | Fourteen days, two rovers, eight crew match. Dates, page range and the two operations modes are unverified. Black Point Lava Flow is not named in the paper text I read.                                                                                                                                                |
| Answers to question 5    | \[CORRECT\] | Most are marked '(implied)'. Keep that label wherever they are quoted; the papers themselves rarely state what Earth cannot match.                                                                                                                                                                                      |

#### What was adopted

| From                                                  | Adopted into                    | How                                                                                                                                                                                                        |
|-------------------------------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Jia 2022 (via Shiven)                                 | Lunar weights, gates            | Scored as a source: PSR distance 3, slope 3, rocks 2, illumination 2, temperature 2. Adds a landing-profile gate at 8° median slope.                                                                       |
| Liu 2025 (via Shiven)                                 | Lunar weights, ML section       | A published random forest on south-pole sites: Earth visibility, hydrogen, kilometre-scale roughness and slope each carry over 10%. Scored as a source, and cited as precedent for our Model B.            |
| Shiven's regolith numbers                             | Limits ledger                   | Grain size, density, abrasiveness and permittivity listed as 'not scored: no global Earth data'.                                                                                                           |
| Shiven's site list                                    | Validation, as a test           | Manicouagan, Samoylov and Tatacoa become gate tests: a correct pipeline rejects all three for vegetation or water. McMurdo and Gosses Bluff join the candidate list.                                       |
| Vihaan's Mars 2020 list                               | Geology profile                 | Justifies composition ×2 and water ×1.5 in the geology and life-detection profile.                                                                                                                         |
| Vihaan's SWIM notes                                   | Water feature for Mars targets  | Use the SWIM ice-consistency map as the target-side water value.                                                                                                                                           |
| Liu 2025 Figure 7 and Jia 2022 Table 2 (sent by Yash) | Lunar weights, simulations      | Re-scored with the real values: no lunar weight moves by more than 0.01, so version 2 stands. Roughness range widened to 0.25. Liu's importances become an alternative weight set.                         |
| Mashal 2025, full text (sent by Yash)                 | Mars weights, gates, validation | Equal weights by design, so it gives no Mars weights. Its masks and eight criteria are re-scored (Mars version 2), and its check against 47 workshop sites is the precedent for our known-site validation. |

### Open gaps

- Morgan's scoring method and Grant's engineering table: not opened (tasks are in document 1).
- Earth data for permafrost, mineral maps and day-night temperature are leads I have not checked for coverage or licence.
- Shaurya's full-text numbers for Lonar and Ladakh are quoted on the cards as his, not checked by me.
- Match-ability values and the "set by you" gate thresholds are judgment with no paper behind the exact numbers.
- Three cards rest on secondary sources. Two details (Dundas instruments, the JSC-1 source cone) are from my memory and are marked so on their cards.
- The other chats in this project were not available to me; I worked from the 9 Nov PDF and your notes on the challenge choice.

### Sources opened

[Mazarico 2011 (NTRS)](https://ntrs.nasa.gov/citations/20120010094) · [Gläser 2018 (NTRS PDF)](https://ntrs.nasa.gov/api/citations/20170007365/downloads/20170007365.pdf) · [Barker 2021 (NTRS)](https://ntrs.nasa.gov/citations/20205009660) · [NASA Artemis III regions](https://www.nasa.gov/?p=142681) · [LROC polar illumination maps](https://lroc.im-ldi.com/images/991) · [Wikipedia, Peak of eternal light](https://en.wikipedia.org/wiki/Peak_of_eternal_light) · [Sefton-Nash 2019 (for Hayne 2015)](https://ora.ox.ac.uk/objects/uuid:515769db-8621-47bd-8082-d1cd8728891e/files/m5a0330e4fa0cf3f51643412442f36633) · [Fisher 2017](https://escholarship.org/uc/item/5wq6z2wf) · [Li 2018](https://escholarship.org/uc/item/2s7392x8) · [Paige 2010](https://pubs.usgs.gov/publication/70036393) · [Lemelin 2021](https://ntrs.nasa.gov/citations/20210016381) · [Morgan 2021](https://www.nature.com/articles/s41550-020-01290-z) · [Dundas 2018](https://pubs.usgs.gov/publication/70195019) · [Piqueux 2019](https://experts.nau.edu/en/publications/widespread-shallow-water-ice-on-mars-at-high-latitudesand-midlati/) · [NASA/JPL release on Piqueux](https://www.sciencedaily.com/releases/2019/12/191211112036.htm) · [Grant 2018](https://authors.library.caltech.edu/87552) · [Chojnacki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5859260) · [Harris 2022](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2022.849078/full) · [Mashal 2025 (PDF supplied by Yash)](https://doi.org/10.3847/PSJ/ade30e) · [Mashal 2025 data (Zenodo)](https://zenodo.org/records/15854213) · [Golombek 2012](https://pubs.usgs.gov/publication/70047380) · [Golombek 2017](https://pubs.usgs.gov/publication/70190673) · [Bussey & Hoffman 2016](https://ntrs.nasa.gov/api/citations/20160002370/downloads/20160002370.pdf) · [Maloof 2010](https://maloof.princeton.edu/publications/geology-lonar-crater-india) · [Hagerty & Newsom 2003](https://pubs.usgs.gov/publication/70202247) · [Pandey 2020](https://www.cambridge.org/core/journals/international-journal-of-astrobiology/article/ladakh-diverse-highaltitude-extreme-environments-for-offearth-analogue-and-astrobiology-research/F9FE95516A6B5D650333ABBCCF2F7512) · [Fudali 1980](https://www.usgs.gov/publications/morphology-lonar-crater-india-comparisons-and-implications) · [Bhattacharya 2016](https://eld.bl.uk/catalog/100031854579_0x000001) · [Ray 2025](https://www.citedrive.com/en/discovery/natrojarosite-formed-in-the-matanomadh-formation-kutch-india-a-na-analog-of-jarosite-on-mars) · [Ansari 2025](https://link.springer.com/article/10.1186/s40645-025-00699-2) · [Stoker 2011 (LPSC abstract)](https://ntrs.nasa.gov/api/citations/20110014304/downloads/20110014304.pdf) · [Howell 2007 (PISCES)](https://ntrs.nasa.gov/api/citations/20070037451/downloads/20070037451.pdf) · [Young 2013 (Desert RATS)](https://ntrs.nasa.gov/api/citations/20120012887/downloads/20120012887.pdf) · [Skinner & Fortezzo 2013](https://pubs.usgs.gov/publication/70005151) · [McKay 1993 (JSC-1)](https://ntrs.nasa.gov/citations/19940011925) · [Navarro-González 2003](https://pesquisa.bvsalud.org/portal/resource/es/mdl-14605363) · [Foucher 2021](https://kar.kent.ac.uk/109579/1/Foucher%20et%20al%20PSS%202021.pdf)
