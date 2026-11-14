# Performance of the Monte Carlo (issue C26)

Budget from the issue: under about 30 minutes per target x profile on 8 cores. **Measured: inside the budget, so nothing was optimised.**

Tool: `uv run python scripts/bench_mc.py --tiles 4000000 --runs 2000 --workers 7 --body moon` (add `--profile-runs 25` for a cProfile listing; works on Windows and macOS).

## Machine

Apple M2 laptop, 8 cores (4 performance + 4 efficiency), 8 GB RAM, macOS, Python 3.11. The Windows machine has **not** been measured: no numbers for it exist yet.
The same issue asks for numbers "for both machines": run the command above on the other machine and add a row.

## Measured

| Case | Tiles | Columns | Runs | Workers | Time | Runs/s |
|---|---|---|---|---|---|---|
| Moon, one target x profile | 4,000,000 | 12 | 2,000 | 7 | **791.7 s = 13.2 min** | 2.53 |
| Mars, one target x profile | 4,000,000 | 14 | 125 | 7 | 47.8 s | 2.61 |
| Moon | 4,000,000 | 12 | 125 | 7 | 41.7 s | 3.00 |
| Moon | 1,000,000 | 12 | 125 | 1 | 20.4 s | 6.12 |

Only the first row is a full 2,000-run measurement. The Mars row is 125 runs; scaling it by the Moon ratio (2,000 runs took 1.2x longer per run than 125 runs did) gives **about 15 min per Mars profile, an estimate, not a measurement**.

The data is Gaussian noise centred on the synthetic target, so the results table is far bigger than real data would give (~1.35 M rows). Real data is probably somewhat faster. It also assumes all 4 M tiles pass the gates; the real number after gating is smaller.
The real Earth file does not exist yet, so the second half of the issue task ("and, once available, on the real Earth file") is **not done**.

## Projection for the full runs (this machine, 7 workers)

- Moon: 2 targets (Connecting Ridge C1, Shackleton rim) x 4 profiles (base + landing_mobility, power_night, ice_resources) = 8 runs x 13.2 min = **about 1.8 h**.
- Mars: 1 scored target (Jezero; Arcadia has no centre, decision D10) x 5 profiles (base + 4) = 5 runs x ~15 min = **about 1.3 h (estimate)**.
- Both bodies: **about 3 h**. Resumable at block level, so an interruption loses at most one block of 125 runs.

## Where the time goes (cProfile, 25 runs, 1 process)

- `column_similarity` (the Gaussian `exp`): 8.4 s of 17.8 s (47%). This is the floor of the algorithm: one `exp` per tile per column per run.
- Pass 2 (rank percentiles for the candidate tiles) costs as much as pass 1 (it re-scores the candidates), about 49% of the total.
- `family_similarity` + `weighted_score`: ~3.8 s (21%); sorting/partition for top sets: ~3 s (17%); array copies 1.3 s (7%).

## Weaknesses, stated plainly

- **Worker scaling is poor: about 1.7x on 7 workers** (1 worker on 1 M tiles runs 6.1 runs/s, so 4 M tiles on 1 worker is about 1.5 runs/s versus 2.53 on 7). Likely causes, not isolated: 4 of the 8 M2 cores are efficiency cores, the work is memory-bandwidth-bound, and the merge in the parent is serial. A machine with 8 real performance cores may do better; this one does not.
- Numba and a lower-precision `exp` were not tried; the budget is met, so the issue's rule says to stop. The `exp` is 47% of the time, so numba would give at most about 2x, and that is only if it installs on both OSes.
- The 8 GB RAM of this machine is low: the memory estimate for 7 workers at 4 M tiles is above the doctor budget (see SPEC, Memory), so the full run here could stop with `MemoryBudgetError` unless other applications are closed. Use fewer workers (`--workers 4`) if it does.
