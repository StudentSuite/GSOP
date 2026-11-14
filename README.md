# GSOP

GSOP (Geospatial Similarity Observation Platform) scores every 6 km land tile on Earth by how well it matches a Moon-base or Mars landing target.

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the specification, one section per pipeline stage s1 to s9.
- [docs/DECISIONS.md](docs/DECISIONS.md): decisions D1 to D10, made by Anay.

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
