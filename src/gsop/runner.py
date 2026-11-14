"""Resumable stage runner.

Each stage declares its inputs, outputs, the config files it depends on and a
function. A stage is **skipped** when all its outputs exist and the hash of

    (stage name, input file hashes, config hash of its config parts, code version)

equals the hash stored in its completion marker ``results/<body>/.stages/<name>.json``.
Otherwise it is **rerun**, and the reason is logged.

Crash safety:
  * the marker is deleted before the stage starts and written (atomically) only after
    the stage function returned and every output exists, so a killed stage is rerun;
  * stage functions should write their outputs with :mod:`gsop.atomic` so a crash never
    leaves a half-written file that looks complete.

``force`` reruns named stages; ``from_stage`` reruns that stage and every later one.
Because a later stage lists the earlier stage's outputs as inputs, a changed output
propagates by hash without any extra bookkeeping.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from gsop import __version__
from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.hashing import sha256_file, sha256_path, sha256_text

MARKER_DIR = ".stages"


class RunnerError(RuntimeError):
    pass


@dataclass
class Stage:
    name: str
    func: Callable[["StageContext"], None]
    inputs: Sequence[Path] = ()
    outputs: Sequence[Path] = ()
    config_parts: Sequence[str] = ()
    code_version: Optional[str] = None


@dataclass
class StageContext:
    stage: Stage
    results_dir: Path
    cfg: Optional[Config]


@dataclass
class StageResult:
    name: str
    action: str  # "ran" or "skipped"
    reason: str
    seconds: float = 0.0


def default_code_version(func: Callable) -> str:
    """Hash of the file the stage function lives in, so editing that file reruns the stage."""
    try:
        source = inspect.getsourcefile(func)
    except TypeError:
        source = None
    if source and Path(source).is_file():
        return sha256_file(source)
    return f"gsop-{__version__}"


@dataclass
class Runner:
    results_dir: Path
    cfg: Optional[Config] = None
    body: Optional[str] = None
    runs: Optional[int] = None
    base_dir: Optional[Path] = None  # inputs/outputs are named relative to this in hashes
    write_meta: bool = True  # False for body-independent stages (s1) that have no results/<body>/run_meta.json
    log: Callable[[str], None] = print
    results: list[StageResult] = field(default_factory=list)

    # -- helpers --------------------------------------------------------------
    @property
    def marker_dir(self) -> Path:
        return Path(self.results_dir) / MARKER_DIR

    def marker_path(self, stage: Stage) -> Path:
        return self.marker_dir / f"{stage.name}.json"

    def _label(self, path: Path) -> str:
        path = Path(path)
        if self.base_dir is not None:
            try:
                return path.resolve().relative_to(Path(self.base_dir).resolve()).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def stage_hash(self, stage: Stage) -> str:
        inputs = {}
        for path in stage.inputs:
            if not Path(path).exists():
                raise RunnerError(f"{stage.name}: missing input {path}")
            inputs[self._label(path)] = sha256_path(path)
        if stage.config_parts:
            if self.cfg is None:
                raise RunnerError(f"{stage.name}: depends on config {list(stage.config_parts)} but no config was given")
            config_hash = self.cfg.config_hash(stage.config_parts)
        else:
            config_hash = ""
        code = stage.code_version or default_code_version(stage.func)
        payload = json.dumps(
            {"stage": stage.name, "inputs": inputs, "config": config_hash, "code": code},
            sort_keys=True, separators=(",", ":"),
        )
        return sha256_text(payload)

    def _read_marker(self, stage: Stage) -> Optional[dict]:
        path = self.marker_path(stage)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _decide(self, stage: Stage, current: str, forced: bool) -> tuple[bool, str]:
        """Return (must_run, reason)."""
        if forced:
            return True, "forced (--force or --from)"
        missing = [self._label(p) for p in stage.outputs if not Path(p).exists()]
        marker = self._read_marker(stage)
        if marker is None:
            return True, "no completion marker (never finished, or interrupted)"
        if missing:
            return True, f"output missing: {', '.join(missing)}"
        if marker.get("hash") != current:
            return True, "inputs, config or code changed since the last run"
        return False, "outputs exist and the input + config + code hash is unchanged"

    # -- main entry -----------------------------------------------------------
    def run(
        self,
        stages: Sequence[Stage],
        force: Iterable[str] = (),
        from_stage: Optional[str] = None,
    ) -> list[StageResult]:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise RunnerError("stage names must be unique")
        force = set(force)
        for n in [*force, *([from_stage] if from_stage else [])]:
            if n not in names:
                raise RunnerError(f"unknown stage '{n}' (known: {', '.join(names)})")
        if from_stage:
            force |= set(names[names.index(from_stage):])

        self.results = []
        for stage in stages:
            current = self.stage_hash(stage)
            must_run, reason = self._decide(stage, current, stage.name in force)
            if not must_run:
                self.log(f"[{stage.name}] skipped: {reason}")
                self.results.append(StageResult(stage.name, "skipped", reason))
                continue
            self.log(f"[{stage.name}] running: {reason}")
            self._execute(stage, current, reason)
        return self.results

    def _execute(self, stage: Stage, current: str, reason: str) -> None:
        marker = self.marker_path(stage)
        if marker.exists():
            marker.unlink()  # a killed run must not keep a stale marker
        start = time.perf_counter()
        stage.func(StageContext(stage, Path(self.results_dir), self.cfg))
        seconds = time.perf_counter() - start
        missing = [self._label(p) for p in stage.outputs if not Path(p).exists()]
        if missing:
            raise RunnerError(f"{stage.name}: finished but did not write {', '.join(missing)}")
        atomic_write_text(
            marker,
            json.dumps(
                {
                    "stage": stage.name,
                    "hash": current,
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "seconds": round(seconds, 3),
                },
                indent=2,
            ) + "\n",
        )
        self.results.append(StageResult(stage.name, "ran", reason, seconds))
        if self.cfg is not None and self.write_meta:
            from gsop.runmeta import record_stage  # local import: runmeta imports config only

            record_stage(Path(self.results_dir), self.cfg, stage.name, seconds, self.body, self.runs)
