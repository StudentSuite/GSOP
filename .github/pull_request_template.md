Closes #

## What changed

<!-- One or two sentences. One pull request per issue. -->

## Checklist

- [ ] Branch is named `<phase>-<key>-<slug>` (for example `a-a05-doctor`) and this PR is for one issue only
- [ ] I am not pushing to `main`; this PR targets `main`
- [ ] Tests are included and `uv run pytest -q` is green
- [ ] Works on Windows and macOS: `pathlib` paths, no bash-only scripts, spawn-safe multiprocessing
- [ ] No invented numbers, coordinates, dataset ids, URLs or paper values; anything unverified is in `docs/UNVERIFIED.md`
- [ ] If behaviour changed, `docs/SPEC.md` is updated in this PR
- [ ] If a dependency changed, `uv.lock` is updated and committed
- [ ] No files from `data/` or `results/` are committed

## Notes for the reviewer

<!-- Anything weak, assumed or unverified. Be blunt. -->
