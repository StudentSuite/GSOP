"""Earth Engine login check.

``ensure_ee()`` tries ``ee.Initialize(project=...)`` once. If it fails it prints the exact steps and
exits with code 1. It never loops, never opens a browser and never prompts: the person who owns the
Earth Engine account runs ``earthengine authenticate`` themselves.

The Cloud project ID is never stored in the repo. It comes from the environment variable
``GSOP_EE_PROJECT`` or from ``earth_engine.project`` in ``config/run.yaml`` (left empty).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

ENV_PROJECT = "GSOP_EE_PROJECT"

LOGIN_STEPS = """\
Earth Engine is not ready on this machine. The owner of the Earth Engine account must do this once:

  1. uv run earthengine authenticate          (a browser opens; log in as the account that owns the project)
  2. Find the Google Cloud project ID that is registered for Earth Engine (console.cloud.google.com).
  3. Tell gsop the project, either
       macOS/Linux:  export GSOP_EE_PROJECT=<your-project-id>
       Windows:      setx GSOP_EE_PROJECT <your-project-id>   (then open a new terminal)
     or put it in config/run.yaml under earth_engine: project:   (do not commit it)
  4. Check:  uv run gsop doctor      (it must say: earthengine: authenticated)

Reason reported by Earth Engine: {reason}
"""


def _import_ee() -> Any:
    import ee  # noqa: PLC0415

    return ee


def ee_project(cfg: Any = None) -> Optional[str]:
    """Cloud project ID from ``GSOP_EE_PROJECT`` or ``config/run.yaml``; ``None`` if neither is set."""
    env = os.environ.get(ENV_PROJECT, "").strip()
    if env:
        return env
    if cfg is None:
        try:
            from gsop.config import load_config  # noqa: PLC0415

            cfg = load_config()
        except Exception:  # noqa: BLE001  (a broken config must not break the login check)
            return None
    return getattr(cfg.run, "ee_project", None) or None


def check_ee(project: Optional[str] = None) -> tuple[bool, str]:
    """Try to initialise Earth Engine once. Returns ``(ok, reason)``. Never raises, never prompts."""
    try:
        ee = _import_ee()
    except Exception as exc:  # noqa: BLE001
        return False, f"earthengine-api cannot be imported: {exc!r}"
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as exc:  # noqa: BLE001
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        return False, f"{type(exc).__name__}: {first}"
    return True, ""


def ensure_ee(project: Optional[str] = None, cfg: Any = None) -> Any:
    """Return the initialised ``ee`` module, or print the login steps and exit with code 1."""
    project = project or ee_project(cfg)
    ok, reason = check_ee(project)
    if not ok:
        print(LOGIN_STEPS.format(reason=reason), file=sys.stderr)
        raise SystemExit(1)
    return _import_ee()
