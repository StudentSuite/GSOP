# Results schema

What the pipeline writes and what the site reads. This file is extended by each stage that adds an output; the export stage (s8) mirrors it for `results/all/`. Types are as stored in parquet / JSON.

## Columns carried on every per-target output row

Every output row that belongs to a target (scores, Monte Carlo results, discovery candidates) carries these, so the site can show them next to the number:

| Column | Type | Meaning |
|---|---|---|
| `target` | string | target name (`connecting_ridge_c1`, `shackleton_rim`, `jezero`) |
| `profile` | string | purpose profile (`base`, `landing_mobility`, `power_night`, `ice_resources`, `geology_life` (Mars only)) |
| `confidence` | string, nullable | `low` for Mars `landing_mobility` and `power_night` (decision D5), otherwise null |
| `fails_own_gate` | bool | decision D6: the Mars target check (latitude within 50 deg, elevation below 2 km, thermal inertia at least 100) failed for this target. The target is still scored in `warn` mode. Always false for Moon targets |
| `own_gate_reasons` | string | the failing reasons joined by `;` (for example `MARS_TARGET_CHECK:ELEVATION`), empty when the check passed |

`validation.json` (written by stage s7) lists every target with `fails_own_gate: true`.

## Gates (stage s3), `results/<body>/`

| File | Rows | Columns |
|---|---|---|
| `gates_passed.parquet` | tiles that passed every **base** gate, per target | `target` string, `tile_id` int64, `near` uint8, `landing_ok` bool, `fails_own_gate` bool, `own_gate_reasons` string |
| `gates_rejected.parquet` | tiles that failed at least one base gate, per target | `target`, `tile_id`, `reasons` uint8, `near` uint8, `fails_own_gate`, `own_gate_reasons` |
| `gates_summary.json` | one object | `body`, `tiles`, `target_check_mode`, `bit_table`, `reason_codes`, per-target counts (`reasons`, `near`, `passed`, `rejected`, `passed_landing`, `fails_own_gate`, ...), `unscored`, `not_scored_by_target_check`, `validation` |

`reasons` and `near` are bitmasks: water_ice 1, vegetation 2, builtup_cropland 4, landing_mobility_slope 8, missing_data 16 (`bit_table` in the summary). For each target `passed + rejected` equals the number of land tiles. `landing_ok` is false where the landing slope gate fails: such a tile is usable by every profile except `landing_mobility`. The reason codes are `WATER_OR_ICE`, `NDVI_HIGH`, `BUILTUP_OR_CROPLAND`, `LANDING_SLOPE`, `MISSING_DATA`.
