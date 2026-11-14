# Handoff to S (Shaurya): macOS, Apple M1

Step-by-step instructions for running the Mars side. This file grows as the other handoff issues land (sim zip, tests, golden run, Mars targets, results zip). **This first section is the Earth feature file.**

## 1. Get the Earth feature file and prove it arrived intact

The Earth file is built on **one** machine (the one that owns the Earth Engine login) and shared through a Drive folder. **The link is sent to you privately by Yash or Anay; it is not in this repository.**

You receive three files:

| File | What it is |
|---|---|
| `earth_features_6km.parquet` | one row per Earth land tile, all features |
| `grid.json` | grid definition (CRS, cell size, origin, shape), dataset ids and dates, which features are unavailable |
| `MANIFEST.json` | name, size, row count and sha256 of every file above |

Steps:

1. Download all three files from the shared folder.
2. In the unzipped sim folder, put them into `data/processed/` (the manifest may also sit in `data/`; the check accepts both).
3. Run:

   ```
   uv run gsop verify-data
   ```

   It recomputes the sha256 of each file and compares it with `MANIFEST.json`. It prints `verify-data OK: N file(s) match ...` or names the file that is missing or wrong and exits with an error.
4. Send Yash and Anay the line it printed, plus the file sizes (`ls -l data/processed/`).

If `verify-data` fails: download the file again once. If it fails a second time, send the full output to Yash and Anay and **stop**. Do not edit any code or any file in `data/`.

## Notes

- Nothing under `data/` is committed to git.
- `grid.json` lists `unavailable_features`. At the time of writing `water_index` is expected to be on that list until Anay decides how to build it (see `docs/SPEC.md`, section Earth `water_index`).
