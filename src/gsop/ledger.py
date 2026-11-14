"""Limits ledger and Fidelity (issue C27).

The ledger (``config/ledger.yaml``) lists what Earth cannot match about the Moon or Mars. It is reported beside every score and **never** enters it:
nothing in the scoring or Monte Carlo path imports this module and no stage that scores hashes the ledger file (both are asserted by ``tests/test_ledger.py``).

``fidelity = 100 * mean(status_value over the ledger rows relevant to the purpose)`` with Broken 0, Partly 0.5, Matchable 1 (``status_values`` in the file).
The rows relevant to a purpose are the ones listed under it in the relevance matrix; a row not listed is not counted.

Two limits, stated plainly:

* The ledger has no body or target column, so fidelity depends on the purpose only: every target of a body, and both bodies, get the same number for the same
  purpose name. The ledger is also not edited per body (the Moon's "sterility" row counts for the Moon's base profile). Making it target-specific would
  need new, sourced ledger rows, which this module does not invent.
* The relevance matrix is a DRAFT (``relevance.provisional``) and two rows are marked provisional (they come from an unverified report). Both flags are carried to
  every output so the site can show "provisional" beside the number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gsop.atomic import atomic_write_text
from gsop.config import Config
from gsop.purposes import purpose_names


class LedgerError(ValueError):
    pass


def purpose_rows(cfg: Config, purpose: str) -> list[dict[str, Any]]:
    """The ledger rows counted for ``purpose``, in ledger order, each with its status value, provisional flag and the reason it is relevant."""
    led = cfg.ledger
    if purpose not in led.relevance:
        raise LedgerError(f"no relevance entry for purpose '{purpose}'")
    wanted = led.relevance[purpose]
    out = []
    for rid, row in led.rows.items():
        if rid in wanted:
            out.append({
                "id": rid, "property": row.property, "target": row.target, "best_earth": row.best_earth, "status": row.status,
                "value": led.status_values[row.status], "source": row.source, "provisional": row.provisional, "reason": wanted[rid],
            })
    return out


def fidelity(cfg: Config, purpose: str) -> dict[str, Any]:
    """Fidelity 0-100 for one purpose, with the rows behind it and the provisional flags."""
    rows = purpose_rows(cfg, purpose)
    if not rows:
        raise LedgerError(f"purpose '{purpose}' has no relevant ledger rows: fidelity is undefined")
    value = 100.0 * sum(r["value"] for r in rows) / len(rows)
    prow = [r["id"] for r in rows if r["provisional"]]
    return {
        "purpose": purpose, "fidelity": value, "n_rows": len(rows), "rows": rows,
        "provisional": bool(cfg.ledger.relevance_provisional or prow),
        "relevance_provisional": bool(cfg.ledger.relevance_provisional), "provisional_rows": prow,
    }


def fidelity_report(cfg: Config, body: str) -> dict[str, Any]:
    """Fidelity for every target x purpose of ``body`` (the purposes the body has, per ``purposes.yaml``)."""
    purposes = purpose_names(cfg, body)
    by_purpose = {p: fidelity(cfg, p) for p in purposes}
    entries = []
    for tname, t in cfg.targets(body).items():
        for p in purposes:
            e = dict(by_purpose[p])
            e.update({"target": tname, "scored": t.lat is not None})
            entries.append(e)
    return {
        "body": body,
        "never_used_in_scores": True,
        "relevance_provisional": bool(cfg.ledger.relevance_provisional),
        "status_values": dict(cfg.ledger.status_values),
        "ledger_hash": cfg.config_hash(["ledger"]),
        "entries": entries,
        "rows": [dict(r.__dict__, value=cfg.ledger.status_values[r.status]) for r in cfg.ledger.rows.values()],
    }


def write_fidelity(cfg: Config, body: str, out_dir: Path) -> Path:
    """Write ``fidelity.json`` (entries per target x purpose, plus every ledger row for the site's table)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fidelity.json"
    atomic_write_text(path, json.dumps(fidelity_report(cfg, body), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
