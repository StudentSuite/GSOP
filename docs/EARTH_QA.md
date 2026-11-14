# Earth feature QA

**Status: NOT RUN.** The Earth file does not exist yet (it needs the Earth Engine export, issues B16 to B18), so no real number, plot or pass/fail is recorded here. Running `uv run python scripts/earth_qa.py` after the build overwrites this file with the real report.

## What the QA checks

Per feature: histogram, global map thumbnail, min / median / max, percentage NaN, percentage equal to a raw nodata sentinel (-9999, -32768, 65535).

Sanity checks (each is pass or fail; **any failure blocks the Moon and Mars runs**, verdict in `data/interim/earth_qa.json`):

- every available feature lies inside its plausibility bounds (`src/gsop/earth/sanity.py`): slope in [0, 90], relief >= 0, `water_index` and `dust_index` in [0, 1], temperatures in a physical range, and so on;
- `t_coldest <= t_mean`;
- `rough_b >= rough_a` on at least 80% of tiles (the 80% is a tunable, unsourced reading of "most");
- `slope_p90 >= slope_median`;
- NaN share of each feature below `nan_warn_fraction` (20%); a feature that is NaN everywhere is reported as UNAVAILABLE and fails its own check;
- no raw nodata value leaked into a feature; `tile_id` is unique.

A failing per-tile check also lists the 10-degree cells holding most of the failing tiles, under "Known bad areas".

**Expected failure today:** `water_index` is not built (see `docs/SPEC.md`), so the check "water_index built" will fail until Anay decides how to supply the permafrost layer.
