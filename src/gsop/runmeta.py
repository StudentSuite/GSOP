"""``run_meta.json``: what produced a result folder.

Results from two machines are comparable only if each carries its code commit,
config hash, seeds and machine info. Two runs with identical inputs give the same
hashes and seeds; only timings (and the write time) differ.

Written to ``results/<body>/run_meta.json`` at the end of every stage and of every
``gsop run`` (:func:`record_stage`, :func:`write_run_meta`).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

from gsop import __version__
from gsop.atomic import atomic_write_text
from gsop.config import CONFIG_FILES, Config
from gsop.doctor import package_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_META_NAME = "run_meta.json"
SEED_POLICY = (
    "per-run seeds are numpy.random.SeedSequence(master_seed).spawn(runs); "
    "run i uses child i, so results do not depend on worker count"
)
# Keys that legitimately differ between two otherwise identical runs.
VOLATILE_KEYS = ("stage_timings", "written_at")


def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit and dirty flag. A zip has no .git, so fall back to a stamped ``GIT_COMMIT`` file."""
    commit = _git(["rev-parse", "HEAD"], repo_root)
    if commit:
        status = _git(["status", "--porcelain"], repo_root)
        return {"commit": commit, "dirty": bool(status), "source": "git"}
    stamp = repo_root / "GIT_COMMIT"
    if stamp.is_file():
        first = (stamp.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip()
        if first:
            return {"commit": first, "dirty": None, "source": "GIT_COMMIT file"}
    return {"commit": "unknown", "dirty": None, "source": "none (no .git and no GIT_COMMIT file)"}


def machine_info() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }


def seed_fingerprint(master_seed: int, runs: int, sample: int = 20) -> str:
    """Hash of the first spawned seeds: equal on two machines iff they draw the same seeds."""
    children = np.random.SeedSequence(master_seed).spawn(max(1, min(runs, sample)))
    h = hashlib.sha256()
    for child in children:
        h.update(np.asarray(child.generate_state(4), dtype="<u4").tobytes())
    return h.hexdigest()


def input_hashes(manifest_path: Optional[Path]) -> dict[str, Any]:
    """sha256 of every data file, read from ``data/MANIFEST.json`` (see manifest.py)."""
    if manifest_path is None or not Path(manifest_path).is_file():
        return {"manifest": None, "files": {}}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = {name: info["sha256"] for name, info in manifest.get("files", {}).items()}
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    return {"manifest": manifest_hash, "files": files}


def build_run_meta(
    cfg: Config,
    body: Optional[str] = None,
    runs: Optional[int] = None,
    stage_timings: Optional[dict[str, float]] = None,
    manifest_path: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    runs = cfg.run.runs if runs is None else runs
    if manifest_path is None:
        manifest_path = repo_root / "data" / "MANIFEST.json"
    return {
        "schema_version": 1,
        "gsop_version": __version__,
        "body": body,
        "git": git_info(repo_root),
        "config_hash": cfg.config_hash(),
        "config_file_hashes": {name: cfg.config_hash([name]) for name in CONFIG_FILES},
        "packages": package_versions(),
        "machine": machine_info(),
        "seeds": {
            "master_seed": cfg.run.seed,
            "runs": runs,
            "policy": SEED_POLICY,
            "fingerprint": seed_fingerprint(cfg.run.seed, runs),
        },
        "inputs": input_hashes(manifest_path),
        "stage_timings": dict(stage_timings or {}),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def stable_part(meta: dict[str, Any]) -> dict[str, Any]:
    """The part of run_meta that must be identical for identical inputs."""
    return {k: v for k, v in meta.items() if k not in VOLATILE_KEYS}


def run_meta_path(results_dir: Path) -> Path:
    return Path(results_dir) / RUN_META_NAME


def write_run_meta(results_dir: Path, meta: dict[str, Any]) -> Path:
    return atomic_write_text(run_meta_path(results_dir), json.dumps(meta, indent=2, sort_keys=True) + "\n")


def record_stage(
    results_dir: Path,
    cfg: Config,
    stage: str,
    seconds: float,
    body: Optional[str] = None,
    runs: Optional[int] = None,
    manifest_path: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Refresh run_meta.json after a stage, keeping the timings of earlier stages."""
    path = run_meta_path(results_dir)
    timings: dict[str, float] = {}
    if path.is_file():
        try:
            timings = dict(json.loads(path.read_text(encoding="utf-8")).get("stage_timings", {}))
        except (OSError, ValueError):
            timings = {}
    timings[stage] = round(float(seconds), 3)
    meta = build_run_meta(cfg, body, runs, timings, manifest_path, repo_root)
    write_run_meta(results_dir, meta)
    return meta
