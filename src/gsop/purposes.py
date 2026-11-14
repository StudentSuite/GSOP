"""Purpose profiles: multiply the base family weights, renormalise to sum 1, and carry the profile's extra gate.

Profiles live in ``config/purposes.yaml``: Moon has landing_mobility, power_night, ice_resources plus ``base``; Mars adds geology_life
(decision D5). There is **no Moon life-detection profile**. Mars landing_mobility and power_night carry ``confidence: low``.

The extra gate is a string such as ``slope_med <= 8``. It is parsed by a tiny whitelisted parser (:func:`parse_gate`) that accepts exactly
``<column> <comparison> <number>`` and nothing else; ``eval`` is never used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from gsop.config import BODIES, Config, ExtraGate

_GATE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


class PurposeError(ValueError):
    pass


def parse_gate(expression: str) -> ExtraGate:
    """Parse ``'<column> <comparison> <number>'``. Anything else (``==``, ``and``, calls, attributes, a number on the left) is refused."""
    if not isinstance(expression, str):
        raise PurposeError(f"a gate expression must be a string, got {type(expression).__name__}")
    m = _GATE_RE.match(expression)
    if not m:
        raise PurposeError(f"gate expression {expression!r} is not allowed: only '<column> <=|>=|<|> <number>' is accepted")
    return ExtraGate(m.group(1), m.group(2), float(m.group(3)), expression.strip())


@dataclass(frozen=True)
class PurposeResult:
    name: str
    body: str
    weights: dict[str, float]
    extra_gate: Optional[ExtraGate]
    confidence: Optional[str]  # "low" for Mars landing_mobility and power_night, else None


def purpose_names(cfg: Config, body: str) -> list[str]:
    return list(cfg.purposes(body))


def apply_purpose(cfg: Config, weights: dict[str, float], purpose: str, body: str) -> PurposeResult:
    """Multiply ``weights`` by the profile's multipliers and renormalise to sum 1. Raises for an unknown profile."""
    if body not in BODIES:
        raise PurposeError(f"unknown body '{body}' (expected moon or mars)")
    known = cfg.purposes(body)
    if purpose not in known:
        if purpose == "geology_life" and body == "moon":
            raise PurposeError("there is no Moon life-detection profile (decision D5): geology_life exists for Mars only")
        raise PurposeError(f"unknown purpose profile '{purpose}' for {body}; known: {', '.join(known)}")
    prof = known[purpose]
    out = {fam: w * prof.multipliers.get(fam, 1.0) for fam, w in weights.items()}
    total = sum(out.values())
    if not total > 0:
        raise PurposeError(f"{body}.{purpose}: weights sum to {total}, cannot renormalise")
    out = {fam: w / total for fam, w in out.items()}
    gate = parse_gate(prof.extra_gate.text) if prof.extra_gate else None
    return PurposeResult(purpose, body, out, gate, prof.confidence)
