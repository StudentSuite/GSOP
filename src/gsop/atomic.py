"""Atomic file writes: write to a temp file in the same folder, then rename.

A crash mid-write leaves the old file (or nothing) in place, never a half-written
file that looks complete. ``os.replace`` is atomic on Windows and POSIX.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    # newline="" semantics: write LF exactly so files hash the same on Windows and macOS.
    return atomic_write_bytes(path, text.encode(encoding))
