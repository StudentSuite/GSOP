# GSOP

[![test](https://github.com/StudentSuite/GSOP/actions/workflows/test.yml/badge.svg)](https://github.com/StudentSuite/GSOP/actions/workflows/test.yml)

GSOP (Geospatial Similarity Observation Platform) scores every 6 km land tile on Earth by how well it matches a Moon-base or Mars landing target.

## Documentation

- [docs/source/GSOP_claude_code_setup_prompt.md](docs/source/GSOP_claude_code_setup_prompt.md) and [docs/source/GSOP_combined_-_plan__weights_and_evidence.md](docs/source/GSOP_combined_-_plan__weights_and_evidence.md): the source of truth ([which one wins](docs/source/README.md)).
- [docs/SPEC.md](docs/SPEC.md): the specification, one section per pipeline stage s1 to s9.
- [docs/DECISIONS.md](docs/DECISIONS.md): decisions D1 to D10, made by Anay.

- [CONTRIBUTING.md](CONTRIBUTING.md): rules for issues, branches and pull requests.
- [docs/UNVERIFIED.md](docs/UNVERIFIED.md): everything not yet checked against a primary source.
- [docs/PROVENANCE.md](docs/PROVENANCE.md): where each dataset and package version comes from.

**No-invention rule:** an unverified item is allowed, an undocumented one is not. Never invent numbers, coordinates, dataset ids, URLs or paper values; anything unverified goes in `docs/UNVERIFIED.md` with where it is used.

**Rule:** every pull request that changes behaviour updates `docs/SPEC.md` in the same pull request.

## Tests

```
uv sync
uv run pytest -q              # fast suite, skips slow tests, under 2 minutes
uv run pytest -q --runslow    # also run tests marked @pytest.mark.slow
```

Mark a long-running test with `@pytest.mark.slow`; it is skipped unless you pass `--runslow`. Shared fixtures (`cfg`, `synth_grid_moon`, `synth_grid_mars`, `results_dir`, `small_seed`, `seed_sequences`) live in `tests/conftest.py`.

## Earth Engine login

Earth features are built in Google Earth Engine on **one** machine. The owner of that Earth Engine account logs in once:

```
uv run earthengine authenticate
export GSOP_EE_PROJECT=<your-cloud-project-id>      # Windows: setx GSOP_EE_PROJECT <your-cloud-project-id>
uv run gsop doctor                                  # must print: earthengine: authenticated
```

The project ID is read from `GSOP_EE_PROJECT` (or `earth_engine.project` in `config/run.yaml`, left empty). Never commit a project ID or credentials. `src/gsop/earth/gee_auth.py` prints these steps and exits if the login is missing; it never prompts.

## Earth Engine build runbook (small-region test first)

Run on the machine whose owner has logged in (see above). **Do the Lonar test box before any global export.**

```
uv run gsop doctor                                                   # earthengine: authenticated
uv run gsop s2 earth --source gee --region 73.64,17.28,79.38,22.68 --dry-run   # about 600 km around Lonar: lists chunks and bands
uv run gsop s2 earth --source gee --region 73.64,17.28,79.38,22.68             # starts, polls, retries; resumable
# download gsop_earth_region_r00_c00.tif from the Drive folder, then:
uv run python scripts/ee_probe.py --check-raster path/to/gsop_earth_region_r00_c00.tif
```

`--check-raster` prints one row per band (valid share, min, median, max) and exits 1 on a wrong CRS, a pixel size other than 6000 m, missing or unexpected bands, an all-empty band or values outside plausibility bounds. Write the wall time and the file size on issue B17 and decide go or no-go for the global export. Note that `water_index` is left out until Anay decides (see `docs/SPEC.md`), so the checker expects it to be absent.

## Provenance audit

`uv run python scripts/audit_provenance.py` checks that every feature column, every row of `data/processed/targets_*.parquet` and every column of the Earth file traces to an entry in `docs/PROVENANCE.md` or `docs/UNVERIFIED.md`. It prints only what is missing and exits 1 if anything is. CI runs it as a warning-only step.
