# Provenance

## Software

Resolved by `uv lock` (see `uv.lock`, which is the source of truth). Python is pinned to 3.11 in `.python-version`.

| Package | Resolved version | Platform checked | Wheel only (no build from source) | Checked on |
|---|---|---|---|---|
| numpy | 2.4.6 | macOS arm64 (Yash's machine) | yes | 2026-11-14 |
| scipy | 1.17.1 | macOS arm64 (Yash's machine) | yes | 2026-11-14 |
| scikit-learn | 1.9.1 | macOS arm64 (Yash's machine) | yes | 2026-11-14 |
| rasterio | 1.4.4 | macOS arm64 (Yash's machine) | yes | 2026-11-14 |

Not yet checked: Windows 11 (Anay) and Apple M1 (Shaurya). Both must run `uv sync --frozen` and the import line from issue A03 and paste the output into the issue.
