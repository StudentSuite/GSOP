"""The Mars target-check gate (decision D6), applied to the **target** profile, not to Earth tiles.

A Mars target must lie within +-``max_abs_lat_deg`` (50) of the equator, below ``max_elevation_m`` (2000 m, MOLA, above the areoid), and have thermal
inertia of at least ``min_thermal_inertia`` (100, THEMIS). Thresholds are in ``config/gates.yaml``.

Mode (``mars_target_check.mode``, overridable by ``--target-check``):
  * ``warn`` (default): the target is still scored; every output row for it carries ``fails_own_gate: true`` and the failing reasons, and it is listed
    in ``validation.json``. Arcadia may fail this check.
  * ``reject`` (``strict`` on the command line, for experiments): a target that fails is not scored.

An input that is missing (no elevation or no thermal inertia value for the target) cannot be shown to pass, so it counts as a failure with a reason
``...:INPUT_MISSING_...``. Moon targets have no such check. Reason strings are ``<MARS_TARGET_CHECK code>:<LATITUDE|ELEVATION|THERMAL_INERTIA|INPUT_MISSING_...>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from gsop.config import Config

ELEVATION_COLUMN = "elevation_mean"
THERMAL_INERTIA_COLUMN = "thermal_inertia_themis"
MODES = ("warn", "reject")
CLI_MODES = {"warn": "warn", "strict": "reject", "reject": "reject"}


@dataclass(frozen=True)
class TargetCheck:
    target: str
    mode: str
    passed: bool
    reasons: tuple[str, ...]
    inputs: Mapping[str, Optional[float]] = field(default_factory=dict)
    applicable: bool = True

    @property
    def fails_own_gate(self) -> bool:
        return self.applicable and not self.passed

    @property
    def scored(self) -> bool:
        """warn: always scored; reject: only if the check passed."""
        return (not self.applicable) or self.mode == "warn" or self.passed

    def flags(self) -> dict:
        """Columns added to every output row of this target."""
        return {"fails_own_gate": bool(self.fails_own_gate), "own_gate_reasons": ";".join(self.reasons)}

    def validation_entry(self) -> dict:
        return {"target": self.target, "mode": self.mode, "passed": self.passed, "fails_own_gate": self.fails_own_gate,
                "reasons": list(self.reasons), "inputs": dict(self.inputs), "scored": self.scored}


def effective_mode(cfg: Config, override: Optional[str] = None) -> str:
    """``warn`` or ``reject``: the command-line value (``warn``, ``strict``, ``reject``) if given, else ``gates.yaml``."""
    if override is None:
        return cfg.gates.thresholds["mars_target_check"]["mode"]
    if override not in CLI_MODES:
        raise ValueError(f"--target-check must be one of {', '.join(CLI_MODES)}, got {override!r}")
    return CLI_MODES[override]


def check_target(cfg: Config, body: str, target: str, values: Mapping[str, float], mode: Optional[str] = None) -> TargetCheck:
    """Run the check for one target. ``values`` is the target's column -> value mapping (NaN or absent means missing)."""
    mode = effective_mode(cfg, mode)
    if body != "mars":
        return TargetCheck(target, mode, True, (), {}, applicable=False)
    thr = cfg.gates.thresholds["mars_target_check"]
    code = cfg.gates.reason_codes["mars_target_check"]
    lat = cfg.targets("mars")[target].lat
    elevation = _get(values, ELEVATION_COLUMN)
    ti = _get(values, THERMAL_INERTIA_COLUMN)
    reasons: list[str] = []
    if lat is None:
        reasons.append(f"{code}:INPUT_MISSING_LATITUDE")
    elif abs(lat) > thr["max_abs_lat_deg"]:
        reasons.append(f"{code}:LATITUDE")
    if elevation is None:
        reasons.append(f"{code}:INPUT_MISSING_ELEVATION")
    elif not elevation < thr["max_elevation_m"]:  # "elevation below 2 km": equal fails
        reasons.append(f"{code}:ELEVATION")
    if ti is None:
        reasons.append(f"{code}:INPUT_MISSING_THERMAL_INERTIA")
    elif not ti >= thr["min_thermal_inertia"]:  # "thermal inertia >= 100": equal passes
        reasons.append(f"{code}:THERMAL_INERTIA")
    return TargetCheck(target, mode, not reasons, tuple(reasons), {"lat": lat, "elevation_m": elevation, "thermal_inertia": ti})


def _get(values: Mapping[str, float], key: str) -> Optional[float]:
    v = values.get(key)
    return float(v) if v is not None and np.isfinite(v) else None
