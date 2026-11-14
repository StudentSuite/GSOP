"""sha256 helpers. Files are read in chunks, never fully into RAM."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Union[str, Path], chunk_bytes: int = CHUNK_BYTES) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_path(path: Union[str, Path]) -> str:
    """sha256 of a file, or of a folder (names and contents of all files, sorted, POSIX-style)."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8") + b"\0")
        h.update(sha256_file(child).encode("ascii") + b"\0")
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
