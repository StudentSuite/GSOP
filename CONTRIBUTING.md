# Contributing

Work is tracked as GitHub issues, one per task, grouped in phases A to F (the setup prompt's own phases 1 to 5 are a different numbering). Pick an issue, do exactly that issue, open a pull request.

## Source of truth

`docs/source/` holds the setup prompt and the plan file. Numbers, formulas and decisions D1 to D10 come from there (see `docs/source/README.md` for which file wins). Decisions are copied in `docs/DECISIONS.md`. Do not reopen them.

## Rules for every issue

1. **Never invent numbers, coordinates, dataset ids, URLs or paper values.** Anything unverified goes in `docs/UNVERIFIED.md` with where it is used. A feature you cannot build becomes `unavailable`; the missing-data gate handles it. An unverified item is allowed, an undocumented one is not.
2. **Must work on Windows and macOS.** Use `pathlib`, no bash-only scripts, one CLI (`uv run gsop ...`), spawn-safe multiprocessing.
3. **One branch and one pull request per issue.** Name the branch `<phase>-<key>-<slug>`, lowercase, for example `a-a05-doctor`. Commit messages start with the issue key: `[A05] Implement gsop doctor`.
4. **Tests included.** `uv run pytest -q` must be green before you open the PR. Mark long tests `@pytest.mark.slow`.
5. **Never push to `main`.** Open a pull request; `main` changes only by merge.
6. **A PR that changes behaviour updates `docs/SPEC.md` in the same PR.**
7. **Do not commit data or results.** `data/` and `results/` are git-ignored; parquet files are allowed only under `tests/golden/`.
8. **If blocked twice, comment on the issue and tag Anay.** Do not work around a failed login or download.

## Setup

```
uv sync
uv run gsop doctor
uv run pytest -q
```

Change dependencies with `uv add` / `uv remove` and commit the updated `uv.lock`. CI fails if `uv.lock` is out of date (`uv lock --check`).

## Branch rules for `main`

These are the rules; a repository admin (Anay) applies them in GitHub. Nobody in the team applies them from a script on a shared machine.

- Pull request required, at least 1 approving review.
- The `test` workflow must pass on all three systems (`windows-latest`, `macos-14`, `ubuntu-latest`) and the branch must be up to date.
- No force pushes, no deletion of `main`.

Command for Anay to run once (it changes the remote, so it is not run automatically):

```
gh api -X PUT repos/StudentSuite/GSOP/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["windows-latest", "macos-14", "ubuntu-latest"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 1},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Check the status-check names under the repository's Actions tab after the first CI run, and adjust `contexts` if GitHub shows them differently.
