"""Config loader: reads every YAML in ``config/``, validates it, hashes it.

Errors are one line and name the file and the key, for example
``weights.yaml: moon.slope.range: lower bound 0.5 must be below upper bound 0.2``.
Unknown keys are warnings, never errors (they are listed in ``Config.warnings``
and also raised through :mod:`warnings`).

The hash (:meth:`Config.config_hash`) is computed over the *parsed* data as
canonical JSON, so comments, line endings and key order in the YAML files do
not change it, and Windows and macOS agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

CONFIG_FILES = (
    "weights",
    "gates",
    "purposes",
    "targets",
    "known_sites",
    "ledger",
    "features",
    "run",
)
BODIES = ("moon", "mars")
EVIDENCE_GRADES = ("A", "B", "C")
AVAILABILITY_STATUSES = ("available", "usable_with_caveat", "unavailable")
LEDGER_STATUSES = ("Broken", "Partly", "Matchable")
_EXTRA_GATE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


class ConfigError(ValueError):
    """A config file is broken. The message names the file and the key."""


class ConfigWarning(UserWarning):
    """A config file has something unexpected that is not an error."""


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------
def _fail(where: str, msg: str) -> None:
    raise ConfigError(f"{where}: {msg}")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _need(d: dict, key: str, where: str) -> Any:
    if key not in d:
        _fail(where, f"missing required key '{key}'")
    return d[key]


def _map(v: Any, where: str) -> dict:
    if not isinstance(v, dict):
        _fail(where, f"expected a mapping, got {type(v).__name__}")
    return v


def _list(v: Any, where: str, *, non_empty: bool = False) -> list:
    if not isinstance(v, list):
        _fail(where, f"expected a list, got {type(v).__name__}")
    if non_empty and not v:
        _fail(where, "must not be empty")
    return v


def _str(v: Any, where: str) -> str:
    if not isinstance(v, str) or not v.strip():
        _fail(where, f"expected a non-empty string, got {v!r}")
    return v


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        _fail(where, f"expected true or false, got {v!r}")
    return v


def _num(v: Any, where: str, *, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if not _is_num(v):
        _fail(where, f"expected a number, got {v!r}")
    if lo is not None and v < lo:
        _fail(where, f"{v} is below the minimum {lo}")
    if hi is not None and v > hi:
        _fail(where, f"{v} is above the maximum {hi}")
    return float(v)


def _int(v: Any, where: str, *, lo: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(where, f"expected an integer, got {v!r}")
    if lo is not None and v < lo:
        _fail(where, f"{v} is below the minimum {lo}")
    return v


def _choice(v: Any, allowed: Iterable[str], where: str) -> str:
    allowed = tuple(allowed)
    if v not in allowed:
        _fail(where, f"{v!r} is not one of {', '.join(allowed)}")
    return v


def _body_list(v: Any, where: str) -> tuple[str, ...]:
    items = _list(v, where, non_empty=True)
    return tuple(_choice(b, BODIES, f"{where}[{i}]") for i, b in enumerate(items))


# ---------------------------------------------------------------------------
# typed config objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyWeight:
    name: str
    paper_score: float
    matchability: float
    range: tuple[float, float]
    published: float
    columns: tuple[str, ...]


@dataclass(frozen=True)
class AltWeightSet:
    name: str
    weights: Optional[dict[str, float]]  # explicit family weights, or None
    rule: Optional[str]  # textual rule, or None


@dataclass(frozen=True)
class ExtraGate:
    column: str
    op: str
    value: float
    text: str

    def passes(self, x: float) -> bool:
        return {"<=": x <= self.value, ">=": x >= self.value, "<": x < self.value, ">": x > self.value}[
            self.op
        ]


@dataclass(frozen=True)
class Purpose:
    name: str
    bodies: tuple[str, ...]
    multipliers: dict[str, float]
    extra_gate: Optional[ExtraGate]
    confidence: Optional[str]


@dataclass(frozen=True)
class Gates:
    thresholds: dict[str, Any]
    reason_codes: dict[str, str]


@dataclass(frozen=True)
class Target:
    name: str
    body: str
    lat: Optional[float]
    lon_e: Optional[float]
    source: str
    status: Optional[str]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class TargetSettings:
    target_window_km: float
    tile_stride_km: float
    receiver_height_m: float
    spread_iqr_divisor: float


@dataclass(frozen=True)
class Site:
    name: str
    group: str
    lat: float
    lon: float
    bodies: tuple[str, ...]
    confirmed: bool
    radius_km: Optional[float]
    expect: Optional[str]
    note: Optional[str]


@dataclass(frozen=True)
class KnownSites:
    site_match_radius_km: float
    pass_criteria: dict[str, Any]
    primary: dict[str, Site]
    secondary: dict[str, Site]
    gate_tests: dict[str, Site]
    negative_controls: dict[str, Site]

    @property
    def all_sites(self) -> dict[str, Site]:
        out: dict[str, Site] = {}
        for group in (self.primary, self.secondary, self.gate_tests, self.negative_controls):
            out.update(group)
        return out

    @property
    def unconfirmed(self) -> list[str]:
        return [name for name, s in self.all_sites.items() if not s.confirmed]

    @property
    def unconfirmed_count(self) -> int:
        return len(self.unconfirmed)


@dataclass(frozen=True)
class LedgerRow:
    id: str
    property: str
    target: str
    best_earth: str
    status: str
    source: str
    provisional: bool


@dataclass(frozen=True)
class Ledger:
    rows: dict[str, LedgerRow]
    status_values: dict[str, float]
    relevance_provisional: bool
    relevance: dict[str, dict[str, str]]  # purpose -> row id -> reason


@dataclass(frozen=True)
class Feature:
    name: str
    definition: str
    units: str
    sources: dict[str, Optional[dict[str, Any]]]
    evidence_grade: str
    uncertainty: dict[str, Any]  # body -> mapping or the string "TODO"
    proxy: bool
    availability: dict[str, dict[str, str]] = field(default_factory=dict)  # body -> {status, reason}
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorSettings:
    grid_crs: str
    tile_m: float
    baseline_m: float
    slope_method: str
    max_nan_fraction: float
    nasadem_lat_range: tuple[float, float]


@dataclass(frozen=True)
class Features:
    columns: dict[str, Feature]
    gate_layers: dict[str, dict[str, Any]]
    operators: OperatorSettings


@dataclass(frozen=True)
class RunConfig:
    runs: int
    seed: int
    top_fractions: tuple[float, ...]
    robust_top_fraction: float
    robust_min_run_share: float
    candidate_fraction: float
    chunk_ram_fraction: float
    site_match_radius_km: float
    novelty_radius_km: float
    convergence_checkpoints: tuple[int, ...]
    golden_score_tolerance: float
    golden_top1_jaccard_min: float
    golden_tiles: int
    golden_runs: int
    operator_check_tolerance: float
    ee_project: Optional[str] = None


# ---------------------------------------------------------------------------
# per-file parsers
# ---------------------------------------------------------------------------
class _Warn:
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    def unknown(self, d: dict, allowed: Iterable[str], where: str) -> None:
        allowed = set(allowed)
        for key in d:
            if key not in allowed:
                msg = f"{where}: unknown key '{key}' (ignored)"
                self.sink.append(msg)
                warnings.warn(msg, ConfigWarning, stacklevel=3)


def _parse_weights(raw: Any, w: _Warn) -> tuple[dict, dict]:
    f = "weights.yaml"
    raw = _map(raw, f)
    w.unknown(raw, (*BODIES, "alt_weight_sets"), f)
    families: dict[str, dict[str, FamilyWeight]] = {}
    for body in BODIES:
        block = _map(_need(raw, body, f), f"{f}: {body}")
        if not block:
            _fail(f"{f}: {body}", "must not be empty")
        out: dict[str, FamilyWeight] = {}
        for fam, spec in block.items():
            where = f"{f}: {body}.{fam}"
            spec = _map(spec, where)
            w.unknown(spec, ("paper_score", "matchability", "range", "published", "columns"), where)
            rng = _list(_need(spec, "range", where), f"{where}.range")
            if len(rng) != 2:
                _fail(f"{where}.range", "must have exactly two numbers [lo, hi]")
            lo = _num(rng[0], f"{where}.range[0]")
            hi = _num(rng[1], f"{where}.range[1]")
            if not lo < hi:
                _fail(f"{where}.range", f"lower bound {lo} must be below upper bound {hi}")
            cols = _list(_need(spec, "columns", where), f"{where}.columns", non_empty=True)
            out[fam] = FamilyWeight(
                name=fam,
                paper_score=_num(_need(spec, "paper_score", where), f"{where}.paper_score", lo=0),
                matchability=_num(
                    _need(spec, "matchability", where), f"{where}.matchability", lo=0, hi=1
                ),
                range=(lo, hi),
                published=_num(_need(spec, "published", where), f"{where}.published", lo=0, hi=1),
                columns=tuple(_str(c, f"{where}.columns[{i}]") for i, c in enumerate(cols)),
            )
        families[body] = out

    alts: dict[str, dict[str, AltWeightSet]] = {b: {} for b in BODIES}
    alt_raw = raw.get("alt_weight_sets")
    if alt_raw is not None:
        alt_raw = _map(alt_raw, f"{f}: alt_weight_sets")
        w.unknown(alt_raw, BODIES, f"{f}: alt_weight_sets")
        for body in BODIES:
            for name, spec in _map(alt_raw.get(body, {}), f"{f}: alt_weight_sets.{body}").items():
                where = f"{f}: alt_weight_sets.{body}.{name}"
                spec = _map(spec, where)
                if "rule" in spec:
                    alts[body][name] = AltWeightSet(name, None, _str(spec["rule"], f"{where}.rule"))
                    continue
                weights = {}
                for fam, val in spec.items():
                    if fam not in families[body]:
                        _fail(where, f"unknown family '{fam}' for {body}")
                    weights[fam] = _num(val, f"{where}.{fam}", lo=0, hi=1)
                if not weights:
                    _fail(where, "needs either 'rule' or family weights")
                alts[body][name] = AltWeightSet(name, weights, None)
    return families, alts


_PURPOSE_KEYS = ("bodies", "multipliers", "extra_gate", "confidence")


def _parse_extra_gate(text: Any, where: str) -> ExtraGate:
    text = _str(text, where)
    m = _EXTRA_GATE_RE.match(text)
    if not m:
        _fail(where, f"expected '<column> <op> <number>' such as 'slope_med <= 8', got {text!r}")
    return ExtraGate(m.group(1), m.group(2), float(m.group(3)), text.strip())


def _parse_purposes(raw: Any, families: dict, w: _Warn) -> dict[str, dict[str, Purpose]]:
    f = "purposes.yaml"
    raw = _map(raw, f)
    if not raw:
        _fail(f, "must not be empty")
    out: dict[str, dict[str, Purpose]] = {b: {} for b in BODIES}
    for name, spec in raw.items():
        where = f"{f}: {name}"
        spec = _map(spec, where)
        w.unknown(spec, _PURPOSE_KEYS, where)
        bodies = _body_list(spec["bodies"], f"{where}.bodies") if "bodies" in spec else BODIES
        mult_raw = _map(_need(spec, "multipliers", where), f"{where}.multipliers")
        mult = {}
        for fam, val in mult_raw.items():
            mult[fam] = _num(val, f"{where}.multipliers.{fam}", lo=0)
            if mult[fam] == 0:
                _fail(f"{where}.multipliers.{fam}", "must be above 0")
            if not any(fam in families[b] for b in bodies):
                _fail(f"{where}.multipliers.{fam}", f"unknown family (not in weights.yaml for {', '.join(bodies)})")
        gates_raw = _map(spec.get("extra_gate", {}), f"{where}.extra_gate")
        conf_raw = _map(spec.get("confidence", {}), f"{where}.confidence")
        for body in bodies:
            gate = None
            if body in gates_raw:
                gate = _parse_extra_gate(gates_raw[body], f"{where}.extra_gate.{body}")
            conf = None
            if body in conf_raw:
                conf = _choice(conf_raw[body], ("low", "medium", "high"), f"{where}.confidence.{body}")
            out[body][name] = Purpose(
                name=name,
                bodies=bodies,
                multipliers={k: v for k, v in mult.items() if k in families[body]},
                extra_gate=gate,
                confidence=conf,
            )
        for body in (*gates_raw, *conf_raw):
            _choice(body, BODIES, f"{where} (body key)")
            if body not in bodies:
                _fail(where, f"has settings for {body} but bodies is {list(bodies)}")
    return out


_GATE_REASON_KEYS = (
    "water_ice",
    "vegetation",
    "builtup_cropland",
    "landing_mobility_slope",
    "missing_data",
    "mars_target_check",
)


def _parse_gates(raw: Any, w: _Warn) -> Gates:
    f = "gates.yaml"
    raw = _map(raw, f)
    w.unknown(raw, ("thresholds", "reason_codes"), f)
    thr = _map(_need(raw, "thresholds", f), f"{f}: thresholds")
    where = f"{f}: thresholds"
    _num(_need(thr, "water_ice_fraction", where), f"{where}.water_ice_fraction", lo=0, hi=1)
    _num(_need(thr, "ndvi_median", where), f"{where}.ndvi_median", lo=-1, hi=1)
    _num(_need(thr, "builtup_cropland_fraction", where), f"{where}.builtup_cropland_fraction", lo=0, hi=1)
    mtc = _map(_need(thr, "mars_target_check", where), f"{where}.mars_target_check")
    mw = f"{where}.mars_target_check"
    _choice(_need(mtc, "mode", mw), ("warn", "reject"), f"{mw}.mode")
    _num(_need(mtc, "max_abs_lat_deg", mw), f"{mw}.max_abs_lat_deg", lo=0, hi=90)
    _num(_need(mtc, "max_elevation_m", mw), f"{mw}.max_elevation_m")
    _num(_need(mtc, "min_thermal_inertia", mw), f"{mw}.min_thermal_inertia", lo=0)
    codes = _map(_need(raw, "reason_codes", f), f"{f}: reason_codes")
    for key in _GATE_REASON_KEYS:
        _str(_need(codes, key, f"{f}: reason_codes"), f"{f}: reason_codes.{key}")
    for key, val in codes.items():
        _str(val, f"{f}: reason_codes.{key}")
    values = list(codes.values())
    if len(set(values)) != len(values):
        _fail(f"{f}: reason_codes", "reason code strings must be unique")
    return Gates(thresholds=thr, reason_codes=dict(codes))


def _opt_coord(v: Any, where: str, lo: float, hi: float) -> Optional[float]:
    return None if v is None else _num(v, where, lo=lo, hi=hi)


def _parse_targets(raw: Any, w: _Warn) -> tuple[dict[str, dict[str, Target]], TargetSettings]:
    f = "targets.yaml"
    raw = _map(raw, f)
    settings_keys = ("target_window_km", "tile_stride_km", "receiver_height_m", "spread_iqr_divisor")
    w.unknown(raw, (*BODIES, *settings_keys), f)
    settings = TargetSettings(
        target_window_km=_num(_need(raw, "target_window_km", f), f"{f}: target_window_km", lo=0),
        tile_stride_km=_num(_need(raw, "tile_stride_km", f), f"{f}: tile_stride_km", lo=0),
        receiver_height_m=_num(_need(raw, "receiver_height_m", f), f"{f}: receiver_height_m", lo=0),
        spread_iqr_divisor=_num(_need(raw, "spread_iqr_divisor", f), f"{f}: spread_iqr_divisor", lo=0),
    )
    targets: dict[str, dict[str, Target]] = {}
    for body in BODIES:
        block = _map(_need(raw, body, f), f"{f}: {body}")
        out = {}
        for name, spec in block.items():
            where = f"{f}: {body}.{name}"
            spec = _map(spec, where)
            lat = _opt_coord(_need(spec, "lat", where), f"{where}.lat", -90, 90)
            lon = _opt_coord(_need(spec, "lon_e", where), f"{where}.lon_e", 0, 360)
            status = spec.get("status")
            if status is not None:
                status = _str(status, f"{where}.status")
            if (lat is None or lon is None) and not status:
                _fail(where, "lat/lon is null, so 'status' must explain why (for example unscored_no_target_data)")
            attrs = {
                k: v for k, v in spec.items() if k not in ("lat", "lon_e", "source", "status")
            }
            for k, v in attrs.items():
                if not (_is_num(v) or isinstance(v, str)):
                    _fail(f"{where}.{k}", f"expected a number or string, got {v!r}")
            out[name] = Target(name, body, lat, lon, _str(_need(spec, "source", where), f"{where}.source"), status, attrs)
        targets[body] = out
    return targets, settings


_SITE_GROUPS = ("primary", "secondary", "gate_tests", "negative_controls")


def _parse_known_sites(raw: Any, w: _Warn) -> KnownSites:
    f = "known_sites.yaml"
    raw = _map(raw, f)
    w.unknown(raw, ("site_match_radius_km", "pass_criteria", *_SITE_GROUPS), f)
    pc = _map(_need(raw, "pass_criteria", f), f"{f}: pass_criteria")
    pw = f"{f}: pass_criteria"
    _num(_need(pc, "primary_median_rank_percentile_min", pw), f"{pw}.primary_median_rank_percentile_min", lo=0, hi=100)
    _num(_need(pc, "negative_control_rank_percentile_below", pw), f"{pw}.negative_control_rank_percentile_below", lo=0, hi=100)
    _choice(_need(pc, "gate_tests", pw), ("all_rejected",), f"{pw}.gate_tests")
    groups: dict[str, dict[str, Site]] = {}
    for group in _SITE_GROUPS:
        block = _map(_need(raw, group, f), f"{f}: {group}")
        out = {}
        for name, spec in block.items():
            where = f"{f}: {group}.{name}"
            spec = _map(spec, where)
            w.unknown(spec, ("lat", "lon", "bodies", "confirmed", "radius_km", "expect", "note"), where)
            expect = spec.get("expect")
            if group == "gate_tests":
                expect = _choice(expect, ("rejected",), f"{where}.expect")
            radius = spec.get("radius_km")
            out[name] = Site(
                name=name,
                group=group,
                lat=_num(_need(spec, "lat", where), f"{where}.lat", lo=-90, hi=90),
                lon=_num(_need(spec, "lon", where), f"{where}.lon", lo=-180, hi=180),
                bodies=_body_list(_need(spec, "bodies", where), f"{where}.bodies"),
                confirmed=_bool(_need(spec, "confirmed", where), f"{where}.confirmed"),
                radius_km=None if radius is None else _num(radius, f"{where}.radius_km", lo=0),
                expect=expect,
                note=None if spec.get("note") is None else _str(spec["note"], f"{where}.note"),
            )
        groups[group] = out
    names = [n for g in groups.values() for n in g]
    if len(set(names)) != len(names):
        _fail(f, "site names must be unique across all groups")
    return KnownSites(
        site_match_radius_km=_num(_need(raw, "site_match_radius_km", f), f"{f}: site_match_radius_km", lo=0),
        pass_criteria=pc,
        **groups,
    )


def _parse_ledger(raw: Any, purposes: dict, w: _Warn) -> Ledger:
    f = "ledger.yaml"
    raw = _map(raw, f)
    w.unknown(raw, ("status_values", "rows", "relevance"), f)
    sv = _map(_need(raw, "status_values", f), f"{f}: status_values")
    status_values = {s: _num(_need(sv, s, f"{f}: status_values"), f"{f}: status_values.{s}", lo=0, hi=1) for s in LEDGER_STATUSES}
    rows: dict[str, LedgerRow] = {}
    for i, row in enumerate(_list(_need(raw, "rows", f), f"{f}: rows", non_empty=True)):
        where = f"{f}: rows[{i}]"
        row = _map(row, where)
        w.unknown(row, ("id", "property", "target", "best_earth", "status", "source", "provisional"), where)
        rid = _str(_need(row, "id", where), f"{where}.id")
        if rid in rows:
            _fail(where, f"duplicate row id '{rid}'")
        rows[rid] = LedgerRow(
            id=rid,
            property=_str(_need(row, "property", where), f"{where}.property"),
            target=_str(_need(row, "target", where), f"{where}.target"),
            best_earth=_str(_need(row, "best_earth", where), f"{where}.best_earth"),
            status=_choice(_need(row, "status", where), LEDGER_STATUSES, f"{where}.status"),
            source=_str(_need(row, "source", where), f"{where}.source"),
            provisional=_bool(row.get("provisional", False), f"{where}.provisional"),
        )
    rel = _map(_need(raw, "relevance", f), f"{f}: relevance")
    w.unknown(rel, ("provisional", "purposes"), f"{f}: relevance")
    known_purposes = set(purposes["moon"]) | set(purposes["mars"])
    matrix: dict[str, dict[str, str]] = {}
    for pname, mapping in _map(_need(rel, "purposes", f"{f}: relevance"), f"{f}: relevance.purposes").items():
        if pname not in known_purposes:
            _fail(f"{f}: relevance.purposes.{pname}", "unknown purpose (not in purposes.yaml)")
        matrix[pname] = {}
        for rid, reason in _map(mapping, f"{f}: relevance.purposes.{pname}").items():
            if rid not in rows:
                _fail(f"{f}: relevance.purposes.{pname}.{rid}", "unknown ledger row id")
            matrix[pname][rid] = _str(reason, f"{f}: relevance.purposes.{pname}.{rid}")
    for pname in known_purposes:
        if pname not in matrix:
            _fail(f"{f}: relevance.purposes", f"missing purpose '{pname}'")
    return Ledger(
        rows=rows,
        status_values=status_values,
        relevance_provisional=_bool(_need(rel, "provisional", f"{f}: relevance"), f"{f}: relevance.provisional"),
        relevance=matrix,
    )


def _parse_features(raw: Any, w: _Warn) -> Features:
    f = "features.yaml"
    raw = _map(raw, f)
    w.unknown(raw, ("features", "gate_layers", "operators"), f)
    ow = f"{f}: operators"
    op = _map(_need(raw, "operators", f), ow)
    w.unknown(op, ("grid_crs", "tile_m", "baseline_m", "slope_method", "max_nan_fraction", "earth_dem"), ow)
    dem = _map(_need(op, "earth_dem", ow), f"{ow}.earth_dem")
    w.unknown(dem, ("nasadem_lat_range",), f"{ow}.earth_dem")
    lat_range = _list(_need(dem, "nasadem_lat_range", f"{ow}.earth_dem"), f"{ow}.earth_dem.nasadem_lat_range")
    if len(lat_range) != 2:
        _fail(f"{ow}.earth_dem.nasadem_lat_range", "must be [south, north]")
    south = _num(lat_range[0], f"{ow}.earth_dem.nasadem_lat_range[0]", lo=-90, hi=90)
    north = _num(lat_range[1], f"{ow}.earth_dem.nasadem_lat_range[1]", lo=-90, hi=90)
    if not south < north:
        _fail(f"{ow}.earth_dem.nasadem_lat_range", f"south {south} must be below north {north}")
    operators = OperatorSettings(
        grid_crs=_str(_need(op, "grid_crs", ow), f"{ow}.grid_crs"),
        tile_m=_num(_need(op, "tile_m", ow), f"{ow}.tile_m", lo=1),
        baseline_m=_num(_need(op, "baseline_m", ow), f"{ow}.baseline_m", lo=1),
        slope_method=_choice(_need(op, "slope_method", ow), ("horn",), f"{ow}.slope_method"),
        max_nan_fraction=_num(_need(op, "max_nan_fraction", ow), f"{ow}.max_nan_fraction", lo=0, hi=1),
        nasadem_lat_range=(south, north),
    )
    if abs(operators.tile_m / operators.baseline_m - round(operators.tile_m / operators.baseline_m)) > 1e-9:
        _fail(ow, f"tile_m {operators.tile_m} must be a whole multiple of baseline_m {operators.baseline_m}")
    cols: dict[str, Feature] = {}
    known = ("definition", "units", "sources", "evidence_grade", "uncertainty", "proxy", "availability")
    for name, spec in _map(_need(raw, "features", f), f"{f}: features").items():
        where = f"{f}: features.{name}"
        spec = _map(spec, where)
        sources = _map(_need(spec, "sources", where), f"{where}.sources")
        for body in sources:
            _choice(body, ("earth", *BODIES), f"{where}.sources (key)")
        for body, src in sources.items():
            if src is not None:
                _map(src, f"{where}.sources.{body}")
                _str(_need(src, "name", f"{where}.sources.{body}"), f"{where}.sources.{body}.name")
        unc = _map(_need(spec, "uncertainty", where), f"{where}.uncertainty")
        for body, val in unc.items():
            _choice(body, ("earth", *BODIES), f"{where}.uncertainty (key)")
            uw = f"{where}.uncertainty.{body}"
            if val == "TODO":
                continue
            val = _map(val, uw)
            lo = _num(_need(val, "min", uw), f"{uw}.min", lo=0)
            hi = _num(_need(val, "max", uw), f"{uw}.max", lo=0)
            if lo > hi:
                _fail(uw, f"min {lo} is above max {hi}")
            _str(_need(val, "unit", uw), f"{uw}.unit")
            _str(_need(val, "source", uw), f"{uw}.source (use the string TODO instead of the whole mapping if there is none)")
        avail: dict[str, dict[str, str]] = {}
        for body, entry in _map(spec.get("availability", {}), f"{where}.availability").items():
            aw = f"{where}.availability.{body}"
            _choice(body, ("earth", *BODIES), f"{where}.availability (key)")
            entry = _map(entry, aw)
            status = _choice(_need(entry, "status", aw), AVAILABILITY_STATUSES, f"{aw}.status")
            reason = _str(_need(entry, "reason", aw), f"{aw}.reason")
            avail[body] = {"status": status, "reason": reason}
        cols[name] = Feature(
            name=name,
            definition=_str(_need(spec, "definition", where), f"{where}.definition"),
            units=_str(_need(spec, "units", where), f"{where}.units"),
            sources=dict(sources),
            evidence_grade=_choice(_need(spec, "evidence_grade", where), EVIDENCE_GRADES, f"{where}.evidence_grade"),
            uncertainty=dict(unc),
            proxy=_bool(spec.get("proxy", False), f"{where}.proxy"),
            availability=avail,
            extra={k: v for k, v in spec.items() if k not in known},
        )
    if not cols:
        _fail(f"{f}: features", "must not be empty")
    layers = {}
    for name, spec in _map(_need(raw, "gate_layers", f), f"{f}: gate_layers").items():
        where = f"{f}: gate_layers.{name}"
        spec = _map(spec, where)
        _str(_need(spec, "definition", where), f"{where}.definition")
        layers[name] = dict(spec)
    return Features(columns=cols, gate_layers=layers, operators=operators)


def _parse_run(raw: Any, w: _Warn) -> RunConfig:
    f = "run.yaml"
    raw = _map(raw, f)
    allowed = (
        "runs", "master_seed", "top_fractions", "robust", "candidate_fraction", "chunk_ram_fraction",
        "site_match_radius_km", "novelty_radius_km", "convergence_checkpoints", "golden",
        "operator_check_tolerance", "earth_engine",
    )
    w.unknown(raw, allowed, f)
    robust = _map(_need(raw, "robust", f), f"{f}: robust")
    golden = _map(_need(raw, "golden", f), f"{f}: golden")
    fr = _list(_need(raw, "top_fractions", f), f"{f}: top_fractions", non_empty=True)
    cp = _list(_need(raw, "convergence_checkpoints", f), f"{f}: convergence_checkpoints", non_empty=True)
    ee_block = _map(raw.get("earth_engine") or {}, f"{f}: earth_engine")
    w.unknown(ee_block, ("project",), f"{f}: earth_engine")
    ee_project = ee_block.get("project")
    if ee_project is not None:
        ee_project = _str(ee_project, f"{f}: earth_engine.project")
    runs = _int(_need(raw, "runs", f), f"{f}: runs", lo=1)
    checkpoints = tuple(_int(c, f"{f}: convergence_checkpoints[{i}]", lo=1) for i, c in enumerate(cp))
    if list(checkpoints) != sorted(set(checkpoints)):
        _fail(f"{f}: convergence_checkpoints", "must be strictly increasing")
    return RunConfig(
        runs=runs,
        seed=_int(_need(raw, "master_seed", f), f"{f}: master_seed", lo=0),
        top_fractions=tuple(_num(x, f"{f}: top_fractions[{i}]", lo=0, hi=1) for i, x in enumerate(fr)),
        robust_top_fraction=_num(_need(robust, "top_fraction", f"{f}: robust"), f"{f}: robust.top_fraction", lo=0, hi=1),
        robust_min_run_share=_num(_need(robust, "min_run_share", f"{f}: robust"), f"{f}: robust.min_run_share", lo=0, hi=1),
        candidate_fraction=_num(_need(raw, "candidate_fraction", f), f"{f}: candidate_fraction", lo=0, hi=1),
        chunk_ram_fraction=_num(_need(raw, "chunk_ram_fraction", f), f"{f}: chunk_ram_fraction", lo=0, hi=1),
        site_match_radius_km=_num(_need(raw, "site_match_radius_km", f), f"{f}: site_match_radius_km", lo=0),
        novelty_radius_km=_num(_need(raw, "novelty_radius_km", f), f"{f}: novelty_radius_km", lo=0),
        convergence_checkpoints=checkpoints,
        golden_score_tolerance=_num(_need(golden, "score_tolerance", f"{f}: golden"), f"{f}: golden.score_tolerance", lo=0),
        golden_top1_jaccard_min=_num(_need(golden, "top1_jaccard_min", f"{f}: golden"), f"{f}: golden.top1_jaccard_min", lo=0, hi=1),
        golden_tiles=_int(_need(golden, "tiles", f"{f}: golden"), f"{f}: golden.tiles", lo=1),
        golden_runs=_int(_need(golden, "runs", f"{f}: golden"), f"{f}: golden.runs", lo=1),
        operator_check_tolerance=_num(_need(raw, "operator_check_tolerance", f), f"{f}: operator_check_tolerance", lo=0),
        ee_project=ee_project,
    )


# ---------------------------------------------------------------------------
# canonical hashing
# ---------------------------------------------------------------------------
def _canonical(obj: Any) -> Any:
    """Make parsed YAML JSON-stable: sorted keys, LF-only strings, fixed float format."""
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return format(obj, ".12g")
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n")
    return str(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# the Config object
# ---------------------------------------------------------------------------
class Config:
    """All validated config, with typed accessors and a stable hash."""

    def __init__(self, config_dir: Path, raw: dict[str, Any], warnings_: list[str]) -> None:
        self.config_dir = config_dir
        self.raw = raw
        self.warnings = warnings_
        w = _Warn(self.warnings)
        self._families, self._alt_sets = _parse_weights(raw["weights"], w)
        self._purposes = _parse_purposes(raw["purposes"], self._families, w)
        self.gates: Gates = _parse_gates(raw["gates"], w)
        self._targets, self.target_settings = _parse_targets(raw["targets"], w)
        self.known_sites: KnownSites = _parse_known_sites(raw["known_sites"], w)
        self.ledger: Ledger = _parse_ledger(raw["ledger"], self._purposes, w)
        self.features: Features = _parse_features(raw["features"], w)
        self.run: RunConfig = _parse_run(raw["run"], w)
        self._cross_check()

    def _cross_check(self) -> None:
        for body in BODIES:
            for fam in self._families[body].values():
                for col in fam.columns:
                    if col not in self.features.columns:
                        _fail(
                            f"weights.yaml: {body}.{fam.name}.columns",
                            f"column '{col}' is not defined in features.yaml",
                        )
            for purpose in self._purposes[body].values():
                if purpose.extra_gate and purpose.extra_gate.column not in self.features.columns:
                    _fail(
                        f"purposes.yaml: {purpose.name}.extra_gate.{body}",
                        f"column '{purpose.extra_gate.column}' is not defined in features.yaml",
                    )

    # -- typed accessors ----------------------------------------------------
    @staticmethod
    def _body(body: str) -> str:
        if body not in BODIES:
            raise ConfigError(f"unknown body '{body}' (expected moon or mars)")
        return body

    def weights(self, body: str) -> dict[str, FamilyWeight]:
        return dict(self._families[self._body(body)])

    def alt_weight_sets(self, body: str) -> dict[str, AltWeightSet]:
        return dict(self._alt_sets[self._body(body)])

    def purposes(self, body: str) -> dict[str, Purpose]:
        return dict(self._purposes[self._body(body)])

    def targets(self, body: str) -> dict[str, Target]:
        return dict(self._targets[self._body(body)])

    # -- hashing ------------------------------------------------------------
    def config_hash(self, parts: Optional[Iterable[str]] = None) -> str:
        """sha256 over canonical JSON of the parsed config.

        ``parts`` limits the hash to some files (for example ``["weights", "run"]``)
        so a stage can depend on just the files it reads. Default: every file.
        """
        names = sorted(CONFIG_FILES if parts is None else parts)
        for name in names:
            if name not in CONFIG_FILES:
                raise ConfigError(f"unknown config part '{name}'")
        payload = canonical_json({name: self.raw[name] for name in names})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(config_dir: Optional[Path] = None) -> Config:
    """Load and validate every YAML file in ``config_dir`` (default: repo ``config/``)."""
    config_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    raw: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = config_dir / f"{name}.yaml"
        if not path.is_file():
            raise ConfigError(f"{name}.yaml: file not found in {config_dir}")
        try:
            raw[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            first = str(exc).strip().splitlines()[0]
            raise ConfigError(f"{name}.yaml: not valid YAML ({first})") from exc
        if raw[name] is None:
            raise ConfigError(f"{name}.yaml: file is empty")
    return Config(config_dir, raw, [])
