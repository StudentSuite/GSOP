# Decisions

Decisions D1 to D10 were made by **Anay** and recorded on **2026-11-14**. Do not reopen them. They are copied from the setup prompt (section 1) and win over the plan file when the two disagree (see `docs/source/README.md`).

The GitHub issue tracker uses phases A to F; the setup prompt's phases are 1 to 5. These are different numberings.

- **D1.** Tile size is **6 km x 6 km**, on the equal-area grid EPSG:6933 (6000 m cells). Land only, with polar ice gated out.
- **D2.** **2,000 Monte Carlo runs** per configuration (`--runs` overridable). Master seed `20261114`; per-run seeds from `numpy.random.SeedSequence(master).spawn(n)` so results do not depend on worker count.
- **D3.** **Lunar thermal inertia weight.** Use the weight derived exactly by the rule (0.019). The published table shows 0.03, which is a rounding-up artifact so the column sums to 1. Keep both in config; the derived value drives the run.
- **D4.** **Illumination.** The primary run follows the published weights (illumination 0.18 in the Moon similarity score). Also run a sensitivity variant `no_illumination` (illumination weight set to 0, rest renormalised) for the Moon, and report whether the robust tiles survive. Earth cannot match polar lighting, so also report illumination in the fidelity output.
- **D5.** **Purpose profiles.** Moon has 3 (landing_mobility, power_night, ice_resources) plus `base`. Mars has 4 (the three plus geology_life) plus `base`. **No Moon life-detection.** Mark Mars landing_mobility and power_night results `confidence: low`.
- **D6.** **Mars target check gate** (target within +/-50 deg latitude, elevation below 2 km, thermal inertia >= 100): implement it as `warn` by default. If a target fails (Arcadia may), still run it, but label every output for that target `fails_own_gate: true` and put it in `validation.json`.
- **D7.** **Mars ice variables are one composite** (`water_index`). Do not count ice depth, ice consistency and thermal inertia separately.
- **D8.** Monte Carlo jitters **weights and data**. See `docs/SPEC.md`, section Monte Carlo.
- **D9.** ML runs today as a second opinion only. It never produces the score.
- **D10.** Arcadia Planitia's centre is **not in our files**. Load it from the SWIM ice map if you can reach it; otherwise leave Arcadia as `unscored: no target data`. Do not guess coordinates.

## Environment notes

(Added by issue A03 if a wheel is missing for arm64 or Windows. None needed so far: on macOS arm64 every package resolved to a prebuilt wheel. Windows 11 and the M1 Mac are still to be checked by Anay and Shaurya.)
