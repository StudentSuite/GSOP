"""Gates (stage s3): reject tiles that look numerically similar but are not analog candidates, and say why.

Gates (thresholds in ``config/gates.yaml``, never in code; reason codes are the stable strings of ``gates.reason_codes``):

  water_ice               open water or permanent ice covers MORE than ``water_ice_fraction`` of the tile
  vegetation              median NDVI is ABOVE ``ndvi_median``
  builtup_cropland        built-up or farmed land covers MORE than ``builtup_cropland_fraction`` of the tile
  landing_mobility_slope  the body's landing_mobility extra gate (``config/purposes.yaml``): Moon ``slope_med <= 8``, Mars ``slope_p90 <= 30``
                          (a value equal to the limit passes). It is evaluated for every tile and *applied* only by the landing_mobility profile
  missing_data            any feature the target's weights use is NaN, or a gate layer is NaN (a tile whose water, vegetation or land use is
                          unknown cannot be shown to be an analog candidate)

The Mars target check is a different kind of gate (on the target, not on tiles): ``gsop.target_check``.

A tile can fail several gates: the result is a per-tile bitmask. ``near`` marks gates a tile PASSED within ``near_miss_fraction`` (10%) of the
threshold, for the discovery cards. Everything is vectorised and works on any chunk of rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from gsop.config import Config, ExtraGate

# bit position of each gate in the per-tile mask (stable: stored in output files)
BITS = {"water_ice": 1, "vegetation": 2, "builtup_cropland": 4, "landing_mobility_slope": 8, "missing_data": 16}
GATE_LAYER_COLUMNS = {"water_ice": "water_ice_fraction", "vegetation": "ndvi_median", "builtup_cropland": "builtup_cropland_fraction"}
THRESHOLD_KEYS = {"water_ice": "water_ice_fraction", "vegetation": "ndvi_median", "builtup_cropland": "builtup_cropland_fraction"}
BASE_MASK = BITS["water_ice"] | BITS["vegetation"] | BITS["builtup_cropland"] | BITS["missing_data"]  # gates every profile applies
LANDING_BIT = BITS["landing_mobility_slope"]


@dataclass
class GateResult:
    reasons: np.ndarray  # uint8 bitmask of failed gates, 0 = passed everything
    near: np.ndarray  # uint8 bitmask of gates passed within near_miss_fraction of the threshold

    def __len__(self) -> int:
        return len(self.reasons)

    def passed(self, applies_landing_gate: bool = False) -> np.ndarray:
        """Tiles that passed every gate the profile applies (the landing slope gate only for landing_mobility)."""
        mask = BASE_MASK | (LANDING_BIT if applies_landing_gate else 0)
        return (self.reasons & mask) == 0


def reason_codes(cfg: Config) -> dict[str, str]:
    """``{gate name: stable reason code string}`` from ``gates.yaml``."""
    return {name: cfg.gates.reason_codes[name] for name in BITS}


def decode(mask: int, cfg: Config, applies_landing_gate: bool = True) -> list[str]:
    """Reason code strings for one tile's bitmask (the landing slope bit is reported only if the profile applies it)."""
    codes = reason_codes(cfg)
    return [codes[name] for name, bit in BITS.items() if mask & bit and (applies_landing_gate or bit != LANDING_BIT)]


def landing_gate(cfg: Config, body: str) -> Optional[ExtraGate]:
    p = cfg.purposes(body).get("landing_mobility")
    return p.extra_gate if p else None


def evaluate_gates(cfg: Config, body: str, columns: Mapping[str, np.ndarray], required: Sequence[str]) -> GateResult:
    """Gate a block of tiles. ``columns`` maps column names to equal-length arrays; ``required`` are the feature columns the target's weights use."""
    thr = cfg.gates.thresholds
    near_frac = thr["near_miss_fraction"]
    n = len(next(iter(columns.values())))
    reasons = np.zeros(n, dtype=np.uint8)
    near = np.zeros(n, dtype=np.uint8)

    missing = np.zeros(n, dtype=bool)
    for c in required:
        missing |= ~np.isfinite(np.asarray(columns[c], dtype="float64"))
    for name, col in GATE_LAYER_COLUMNS.items():
        v = np.asarray(columns[col], dtype="float64")
        missing |= ~np.isfinite(v)
        limit = thr[THRESHOLD_KEYS[name]]
        with np.errstate(invalid="ignore"):
            fail = v > limit
            reasons |= np.where(fail, BITS[name], 0).astype(np.uint8)
            near |= np.where(~fail & (v > limit * (1.0 - near_frac)), BITS[name], 0).astype(np.uint8)

    gate = landing_gate(cfg, body)
    if gate is not None:
        v = np.asarray(columns[gate.column], dtype="float64")
        with np.errstate(invalid="ignore"):
            passes = np.where(np.isfinite(v), _vector_passes(gate, v), False)
            reasons |= np.where(np.isfinite(v) & ~passes, LANDING_BIT, 0).astype(np.uint8)
            if gate.op in ("<=", "<"):
                near |= np.where(np.isfinite(v) & passes & (v > gate.value * (1.0 - near_frac)), LANDING_BIT, 0).astype(np.uint8)
            else:
                near |= np.where(np.isfinite(v) & passes & (v < gate.value * (1.0 + near_frac)), LANDING_BIT, 0).astype(np.uint8)
    reasons |= np.where(missing, BITS["missing_data"], 0).astype(np.uint8)
    return GateResult(reasons, near)


def _vector_passes(gate: ExtraGate, v: np.ndarray) -> np.ndarray:
    return {"<=": v <= gate.value, ">=": v >= gate.value, "<": v < gate.value, ">": v > gate.value}[gate.op]


def counts_by_reason(result: GateResult, cfg: Config, applies_landing_gate: bool = False) -> dict[str, int]:
    """Number of tiles per reason code (a tile failing two gates counts in both) plus ``REJECTED`` and ``PASSED`` totals."""
    codes = reason_codes(cfg)
    out = {codes[name]: int(((result.reasons & bit) != 0).sum()) for name, bit in BITS.items() if applies_landing_gate or bit != LANDING_BIT}
    passed = result.passed(applies_landing_gate)
    out["PASSED"] = int(passed.sum())
    out["REJECTED"] = int((~passed).sum())
    return out
