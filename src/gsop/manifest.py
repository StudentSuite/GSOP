"""``data/MANIFEST.json``: filename, size, sha256, row count and creation time of every file in
``data/processed/``, and the check that a copy of those files arrived intact.

Write it on the machine that builds the Earth file::

    uv run python -m gsop.manifest write

Then ``gsop verify-data`` on every other machine recomputes the hashes.

The manifest is read from ``data/MANIFEST.json``; if it is not there, from
``data/processed/MANIFEST.json`` (HANDOFF_S.md tells S to drop it into ``data/processed/``).
Either way the manifest never lists itself. Hashing is chunked, so a large parquet file
is never read fully into RAM.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gsop.atomic import atomic_write_text
from gsop.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
MANIFEST_NAME = "MANIFEST.json"
IGNORED_NAMES = {".gitkeep", MANIFEST_NAME}


def processed_files(processed_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(processed_dir).rglob("*")
        if p.is_file() and p.name not in IGNORED_NAMES and not p.name.endswith(".tmp")
    )


def _row_count(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".parquet":
        return None
    import pyarrow.parquet as pq  # noqa: PLC0415

    return int(pq.ParquetFile(path).metadata.num_rows)  # reads the footer only


def build_manifest(processed_dir: Path) -> dict:
    processed_dir = Path(processed_dir)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {}
    for path in processed_files(processed_dir):
        files[path.relative_to(processed_dir).as_posix()] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": _row_count(path),
            "created": now,
        }
    return {"schema_version": 1, "created": now, "files": files}


def write_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    data_dir = Path(data_dir)
    manifest = build_manifest(data_dir / "processed")
    return atomic_write_text(data_dir / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def find_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> Optional[Path]:
    for candidate in (Path(data_dir) / MANIFEST_NAME, Path(data_dir) / "processed" / MANIFEST_NAME):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Problem:
    file: str
    kind: str  # "missing" | "size" | "sha256" | "manifest"
    detail: str

    def __str__(self) -> str:
        return f"{self.file}: {self.kind}: {self.detail}"


@dataclass
class VerifyResult:
    manifest: Optional[Path]
    checked: int
    problems: list[Problem]
    extra_files: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_data(data_dir: Path = DEFAULT_DATA_DIR) -> VerifyResult:
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    manifest_path = find_manifest(data_dir)
    if manifest_path is None:
        return VerifyResult(None, 0, [Problem(MANIFEST_NAME, "manifest", f"not found in {data_dir} or {processed}")], [])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        if not isinstance(files, dict):
            raise TypeError("'files' must be a mapping")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return VerifyResult(manifest_path, 0, [Problem(MANIFEST_NAME, "manifest", f"cannot read: {exc}")], [])
    problems: list[Problem] = []
    for name, info in sorted(files.items()):
        path = processed / name
        if not path.is_file():
            problems.append(Problem(name, "missing", f"expected in {processed}"))
            continue
        size = path.stat().st_size
        if size != info.get("size"):
            problems.append(Problem(name, "size", f"manifest says {info.get('size')} bytes, file has {size}"))
            continue
        digest = sha256_file(path)
        if digest != info.get("sha256"):
            problems.append(Problem(name, "sha256", f"manifest {info.get('sha256')}, file {digest}"))
    listed = set(files)
    extra = [p.relative_to(processed).as_posix() for p in processed_files(processed)
             if p.relative_to(processed).as_posix() not in listed] if processed.is_dir() else []
    return VerifyResult(manifest_path, len(files), problems, extra)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["write"]:
        print("usage: python -m gsop.manifest write [DATA_DIR]")
        return 2
    data_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA_DIR
    path = write_manifest(data_dir)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
