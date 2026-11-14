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

TODO (phase B). One documented numpy operator for slope, relief and roughness on Earth, Moon and Mars (`src/gsop/operators.py`). If Earth Engine computes the Earth side, the matching operator is implemented there and `gsop check-operators` compares the two.

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
