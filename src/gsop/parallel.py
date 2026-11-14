"""Spawn-safe process pool with the big arrays in shared memory.

Why: Windows and macOS both start workers with *spawn* (a fresh interpreter that imports
your module), and the feature matrix has about 4 million rows. Passing it to each task
would pickle hundreds of MB per task. Instead the parent copies the matrix once into
``multiprocessing.shared_memory``; workers attach by name and get a read-only numpy view.
Only the tiny :class:`SharedSpec` (name, shape, dtype) is ever pickled.

Rules for callers:

* The task function must be a module-level function (importable by name), with signature
  ``func(task, shared) -> result`` where ``shared`` is ``{name: read-only ndarray}``.
  Tasks and results must be small and picklable.
* Call :func:`run_parallel` from code that is reachable only under
  ``if __name__ == "__main__":`` (the ``gsop`` CLI is) so spawn can re-import safely.

Cleanup: shared memory is released in a ``finally`` block, so it is freed after an
exception in the parent or in any worker. On Windows the segment disappears when the last
handle closes; on POSIX the parent also unlinks it.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import psutil

_SPAWN = mp.get_context("spawn")  # same start method on Windows, macOS and Linux (so CI matches)


def default_workers() -> int:
    """Physical cores (both team machines have 8); falls back to logical cores."""
    return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)


@dataclass(frozen=True)
class SharedSpec:
    """Everything a worker needs to attach to a shared array. Tiny and picklable."""

    name: str
    shape: tuple[int, ...]
    dtype: str


_LIVE: dict[str, shared_memory.SharedMemory] = {}  # parent-side segments, for the atexit net


def _release(shm: shared_memory.SharedMemory, unlink: bool) -> None:
    try:
        shm.close()
    finally:
        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


@atexit.register
def _cleanup_leftovers() -> None:  # last resort if the interpreter exits with segments alive
    for shm in list(_LIVE.values()):
        _release(shm, unlink=True)
    _LIVE.clear()


class SharedArray:
    """Copy ``array`` into shared memory once. Use as a context manager."""

    def __init__(self, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        self._shm = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
        _LIVE[self._shm.name] = self._shm
        self.array = np.ndarray(array.shape, dtype=array.dtype, buffer=self._shm.buf)
        self.array[...] = array
        self.spec = SharedSpec(self._shm.name, tuple(array.shape), array.dtype.str)

    def close(self) -> None:
        shm = _LIVE.pop(self._shm.name, None)
        if shm is not None:
            self.array = None  # drop the view so the buffer can be closed
            _release(shm, unlink=True)

    def __enter__(self) -> "SharedArray":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
_WORKER_SHARED: dict[str, np.ndarray] = {}
_WORKER_SEGMENTS: list[shared_memory.SharedMemory] = []  # keep attached for the worker's life


def _attach(spec: SharedSpec) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=spec.name)
    _WORKER_SEGMENTS.append(shm)
    view = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    view.flags.writeable = False
    return view


def _init_worker(specs: Mapping[str, SharedSpec]) -> None:
    _WORKER_SHARED.clear()
    for key, spec in specs.items():
        _WORKER_SHARED[key] = _attach(spec)


def _call(func: Callable[[Any, Mapping[str, np.ndarray]], Any], task: Any) -> Any:
    return func(task, _WORKER_SHARED)


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------
def run_parallel(
    func: Callable[[Any, Mapping[str, np.ndarray]], Any],
    tasks: Iterable[Any],
    shared: Optional[Mapping[str, np.ndarray]] = None,
    workers: Optional[int] = None,
    chunksize: int = 1,
) -> list[Any]:
    """Run ``func(task, shared)`` for every task and return the results **in task order**.

    ``workers=1`` runs inline in this process (same function, same views), so results never
    depend on the worker count as long as ``func`` depends only on its task and the shared data.
    """
    tasks = list(tasks)
    shared = dict(shared or {})
    workers = default_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = min(workers, max(len(tasks), 1))

    if workers == 1:
        views = {k: _readonly(v) for k, v in shared.items()}
        return [func(t, views) for t in tasks]

    segments: list[SharedArray] = []
    try:
        specs = {}
        for key, array in shared.items():
            seg = SharedArray(array)
            segments.append(seg)
            specs[key] = seg.spec
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=_SPAWN, initializer=_init_worker, initargs=(specs,)
        ) as pool:
            return list(pool.map(_call, [func] * len(tasks), tasks, chunksize=chunksize))
    finally:
        for seg in segments:
            seg.close()


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


_DONE = object()


def iter_parallel(
    func: Callable[[Any, Mapping[str, np.ndarray]], Any],
    tasks: Iterable[Any],
    shared: Optional[Mapping[str, np.ndarray]] = None,
    workers: Optional[int] = None,
    window: Optional[int] = None,
):
    """Like :func:`run_parallel` but a generator: yields results **in task order, one at a time**, with at most ``window`` tasks in flight.

    Use it when each result is large (Monte Carlo blocks return per-tile arrays): the caller merges and discards each result as it arrives instead
    of holding all of them. On ``KeyboardInterrupt`` or any exception the pool is shut down at once (pending tasks cancelled) and the shared
    memory is released; whatever the caller already saved is kept.
    """
    from collections import deque

    tasks = list(tasks)
    shared = dict(shared or {})
    workers = default_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = min(workers, max(len(tasks), 1))
    if workers == 1:
        views = {k: _readonly(v) for k, v in shared.items()}
        for t in tasks:
            yield func(t, views)
        return
    window = window or workers + 1
    segments: list[SharedArray] = []
    pool = None
    try:
        specs = {}
        for key, array in shared.items():
            seg = SharedArray(array)
            segments.append(seg)
            specs[key] = seg.spec
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=_SPAWN, initializer=_init_worker, initargs=(specs,))
        pending: "deque" = deque()
        it = iter(tasks)
        for t in it:
            pending.append(pool.submit(_call, func, t))
            if len(pending) >= window:
                break
        while pending:
            result = pending.popleft().result()
            nxt = next(it, _DONE)
            if nxt is not _DONE:
                pending.append(pool.submit(_call, func, nxt))
            yield result
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        for seg in segments:
            seg.close()
