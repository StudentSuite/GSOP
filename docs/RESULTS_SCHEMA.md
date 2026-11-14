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
