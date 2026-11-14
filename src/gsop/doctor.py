"""``gsop doctor``: machine report and the memory-safe chunk size.

Never prompts and never raises because Earth Engine is not set up.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

import psutil
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

KEY_PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "rasterio",
    "pyproj",
    "shapely",
    "psutil",
    "pyyaml",
    "typer",
    "rich",
    "tqdm",
    "requests",
    "earthengine-api",
    "pytest",
]

# Used only until config/features.yaml exists (issue A10). It is the column
# list of issue A10 (slope_med ... air_density), 15 columns, gate layers excluded.
_FALLBACK_FEATURE_COLUMNS = 15
# "At most 40% of free RAM" (issue A05). Overridden by run.yaml when present.
_DEFAULT_CHUNK_RAM_FRACTION = 0.4
_BYTES_PER_VALUE = 4  # float32


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def feature_column_count(config_dir: Path = CONFIG_DIR) -> tuple[int, str]:
    """Number of feature columns and where the number came from."""
    data = _read_yaml(config_dir / "features.yaml")
    if data and isinstance(data.get("features"), dict) and data["features"]:
        return len(data["features"]), "config/features.yaml"
    return _FALLBACK_FEATURE_COLUMNS, "fallback (config/features.yaml not found)"


def chunk_ram_fraction(config_dir: Path = CONFIG_DIR) -> float:
    data = _read_yaml(config_dir / "run.yaml")
    try:
        return float(data["chunk_ram_fraction"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return _DEFAULT_CHUNK_RAM_FRACTION


def chunk_cells(free_ram_bytes: int, n_columns: int, fraction: float) -> int:
    """Largest cell chunk whose float32 feature block fits in ``fraction`` of free RAM."""
    return max(int(free_ram_bytes * fraction) // (n_columns * _BYTES_PER_VALUE), 0)


def earth_engine_status() -> dict[str, Any]:
    """Try ``ee.Initialize`` once. Never prompts, never raises. Does not print the project ID."""
    from gsop.earth.gee_auth import check_ee, ee_project  # noqa: PLC0415

    project = ee_project()
    try:
        import ee  # noqa: PLC0415

        version = getattr(ee, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "authenticated": False, "project_configured": bool(project), "detail": repr(exc)}
    ok, reason = check_ee(project)
    return {
        "installed": True,
        "version": version,
        "authenticated": ok,
        "project_configured": bool(project),
        "detail": reason,
    }


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def collect(config_dir: Path = CONFIG_DIR, disk_path: Path = REPO_ROOT) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(disk_path)
    n_cols, cols_source = feature_column_count(config_dir)
    fraction = chunk_ram_fraction(config_dir)
    cells = chunk_cells(vm.available, n_cols, fraction)
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cores_logical": psutil.cpu_count(logical=True),
        "cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_free_gb": round(vm.available / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "packages": package_versions(),
        "earthengine": earth_engine_status(),
        "chunk": {
            "feature_columns": n_cols,
            "feature_columns_source": cols_source,
            "bytes_per_value": _BYTES_PER_VALUE,
            "ram_fraction": fraction,
            "cells": cells,
            "megabytes": round(cells * n_cols * _BYTES_PER_VALUE / 1e6, 1),
        },
    }


def render(info: dict[str, Any]) -> str:
    ee = info["earthengine"]
    if not ee["installed"]:
        ee_line = f"earthengine-api not importable ({ee['detail']})"
    elif ee["authenticated"]:
        ee_line = "authenticated"
    else:
        ee_line = f"NOT authenticated ({ee['detail']}); see README, section Earth Engine login"
    chunk = info["chunk"]
    lines = [
        f"OS:            {info['os']} ({info['machine']})",
        f"Python:        {info['python']}",
        f"Cores:         {info['cores_physical']} physical, {info['cores_logical']} logical",
        f"RAM:           {info['ram_total_gb']} GB total, {info['ram_free_gb']} GB free",
        f"Disk free:     {info['disk_free_gb']} GB",
        f"earthengine:   {ee_line}",
        f"EE project:    {'configured' if ee.get('project_configured') else 'not set (GSOP_EE_PROJECT)'}",
        "Packages:",
    ]
    lines += [f"  {name:<18} {ver}" for name, ver in info["packages"].items()]
    lines += [
        "Chunk size:",
        f"  {chunk['cells']:,} cells x {chunk['feature_columns']} columns x float32"
        f" = {chunk['megabytes']} MB (at most {chunk['ram_fraction']:.0%} of free RAM)",
        f"  feature columns from: {chunk['feature_columns_source']}",
    ]
    return "\n".join(lines)


def run_doctor(json_path: Optional[Path] = None) -> None:
    info = collect()
    print(render(info))
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"\nWrote {json_path}")
