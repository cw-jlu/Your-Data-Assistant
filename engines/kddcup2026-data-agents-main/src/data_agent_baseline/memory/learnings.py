"""Load / save the bundled ``learnings.json`` file.

The JSON file is the persistent memory of public-set patterns we want
applied to hidden-set tasks at eval time. It is bundled INSIDE the
container (lives next to this module) so the runner can read it without
network or external state.

Schema (top-level)::

    {
      "version": "v7",
      "last_updated": "2026-05-08",
      "shape_policies": [ <PolicyEntry>, ... ],
      "metrics": { ... }   // optional, written by the recorder
    }

Each ``PolicyEntry`` is parsed by :func:`policies._parse_entry`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LEARNINGS_PATH: Path = _THIS_DIR / "learnings.json"


_FALLBACK_LEARNINGS: dict[str, Any] = {
    "version": "v7-fallback",
    "shape_policies": [],
    "metrics": {},
}


def load_learnings(path: Path | None = None) -> dict[str, Any]:
    """Load the bundled learnings.json. Returns a safe empty fallback if the
    file is missing or unparseable — callers should never crash on a bad file.
    """
    target = path or DEFAULT_LEARNINGS_PATH
    if not target.is_file():
        return dict(_FALLBACK_LEARNINGS)
    try:
        text = target.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, ValueError):
        return dict(_FALLBACK_LEARNINGS)
    if not isinstance(payload, dict):
        return dict(_FALLBACK_LEARNINGS)
    payload.setdefault("shape_policies", [])
    payload.setdefault("metrics", {})
    return payload


def save_learnings(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write the learnings JSON. Used by the host-side recorder, NEVER by
    the runtime path inside the container (read-only at eval time).
    """
    target = path or DEFAULT_LEARNINGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(serialized, encoding="utf-8")
    return target


__all__ = ["DEFAULT_LEARNINGS_PATH", "load_learnings", "save_learnings"]
