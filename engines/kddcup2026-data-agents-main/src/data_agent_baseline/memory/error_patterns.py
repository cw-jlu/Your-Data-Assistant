"""Error pattern memory (v7.1 / N-1).

Companion to ``learnings.json`` — instead of accumulating *positive* policy
adjustments (timeout multipliers, prompt hints), this module accumulates
*negative* observations: which (action, error_signature) pairs the agent
hits on which task shapes. The bundled ``error_patterns.json`` ships
inside the container, gets read by the runner at task start, and gets
injected into the prompt as a "previously observed failure modes for
similar tasks" advisory — so the agent can avoid repeating the mistake
on hidden-set tasks of the same shape.

Why this layer exists
---------------------

Within a single task, the ReAct loop already feeds tool errors back as
observations. Across tasks, that information was previously lost — every
new task started blind. With error_patterns.json bundled, the agent gets
a one-shot "by the way, last time a hard task hit `no such column`, the
fix was to call inspect_sqlite_schema first" before it touches a tool.

Schema
------

::

    {
      "version": "v7.1",
      "last_updated": "YYYY-MM-DD",
      "patterns": [
        {
          "match": {<TaskShape attribute filter>},
          "action": "execute_context_sql" | "execute_python" | "*",
          "error_signature": "no such column",
          "frequency": 4,
          "evidence_tasks": ["task_330", "task_408"],
          "hint": "Call inspect_sqlite_schema before composing SELECT clauses.",
          "first_seen": "2026-05-08",
          "last_seen": "2026-05-10"
        }
      ]
    }

Patterns are matched the same way as policies (AND across keys; range
expressions allowed). When multiple patterns match a task, all of them
contribute their hint (deduped) to the prompt.

The classifier function ``error_signature_for(message)`` strips numeric
literals, file paths, and quoted identifiers so that two messages
differing only in a row count or a column name collapse to the same
class.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.memory.task_shape import TaskShape


_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ERROR_PATTERNS_PATH: Path = _THIS_DIR / "error_patterns.json"


# Regex strippers used by ``error_signature_for``. Order matters — strip
# the longest-tokened replacements first so a quoted path doesn't get
# half-stripped by the numeric stripper.
_QUOTED_TOKEN_RE = re.compile(r'(?P<q>["\'`])([^"\'`\n]*)(?P=q)')
_PATH_RE = re.compile(r"(?:/[\w\.\-]+)+|[A-Za-z]:\\[\w\\.\-]+")
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WHITESPACE_RE = re.compile(r"\s+")
_FRAME_LOCATION_RE = re.compile(r'File "[^"]+", line \d+,? in \S+')

# Coarse error class lookup — first match wins.
_ERROR_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sqlite_no_such_column", re.compile(r"no such column", re.IGNORECASE)),
    ("sqlite_no_such_table", re.compile(r"no such table", re.IGNORECASE)),
    ("sqlite_syntax", re.compile(r"sqlite syntax|near\s+", re.IGNORECASE)),
    ("python_keyerror", re.compile(r"\bKeyError\b")),
    ("python_attributeerror", re.compile(r"\bAttributeError\b")),
    ("python_indexerror", re.compile(r"\bIndexError\b")),
    ("python_valueerror", re.compile(r"\bValueError\b")),
    ("python_typeerror", re.compile(r"\bTypeError\b")),
    ("python_filenotfound", re.compile(r"FileNotFoundError|No such file or directory", re.IGNORECASE)),
    ("python_jsondecodeerror", re.compile(r"JSONDecodeError|Expecting value|Extra data", re.IGNORECASE)),
    ("python_memoryerror", re.compile(r"\bMemoryError\b")),
    ("python_timeout", re.compile(r"\bTimeoutError\b|timed out|timeout", re.IGNORECASE)),
    ("pandas_columnmissing", re.compile(r"\bnone of \[.+\] are in the \[columns\]|\['?[^'\]]+'?\] not in index", re.IGNORECASE)),
    ("execute_python_capped", re.compile(r"output capped at|execute_python_timeout", re.IGNORECASE)),
    ("read_json_capped", re.compile(r"file is \d+ bytes; max_bytes=", re.IGNORECASE)),
)


def error_signature_for(error_message: str | None) -> str:
    """Reduce a free-text error message to a stable, low-cardinality class.

    Returns one of the ``_ERROR_CLASS_PATTERNS`` keys when one matches; otherwise
    a sanitized lowercase prefix of the message (numbers/quotes/paths stripped).
    Empty / None → ``"unknown"``.
    """
    if not error_message:
        return "unknown"
    text = str(error_message)
    for label, pattern in _ERROR_CLASS_PATTERNS:
        if pattern.search(text):
            return label
    # Generic fallback: collapse the message into a short canonical token.
    sanitized = _FRAME_LOCATION_RE.sub("", text)
    sanitized = _QUOTED_TOKEN_RE.sub('"_"', sanitized)
    sanitized = _PATH_RE.sub("/_", sanitized)
    sanitized = _NUMERIC_RE.sub("N", sanitized)
    sanitized = _WHITESPACE_RE.sub(" ", sanitized).strip().lower()
    if not sanitized:
        return "unknown"
    return sanitized[:120]


_FALLBACK_PATTERNS: dict[str, Any] = {
    "version": "v7.1-fallback",
    "patterns": [],
}


def load_error_patterns(path: Path | None = None) -> dict[str, Any]:
    """Load the bundled error_patterns.json. Safe fallback on any error."""
    target = path or DEFAULT_ERROR_PATTERNS_PATH
    if not target.is_file():
        return dict(_FALLBACK_PATTERNS)
    try:
        text = target.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, ValueError):
        return dict(_FALLBACK_PATTERNS)
    if not isinstance(payload, dict):
        return dict(_FALLBACK_PATTERNS)
    payload.setdefault("patterns", [])
    return payload


def save_error_patterns(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write error_patterns.json. Host-side recorder use only."""
    target = path or DEFAULT_ERROR_PATTERNS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(serialized, encoding="utf-8")
    return target


@dataclass(frozen=True, slots=True)
class ErrorAdvisory:
    """One hint surfaced to the agent at task start."""
    action: str
    error_signature: str
    hint: str
    frequency: int


def _entry_matches(match: dict[str, Any], shape: TaskShape) -> bool:
    """Reuse the ShapePolicy match algebra (kept inline to avoid coupling)."""
    shape_dict = shape.to_dict()
    for key, expected in match.items():
        actual = shape_dict.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif isinstance(expected, dict):
            if "gte" in expected and not (isinstance(actual, (int, float)) and actual >= expected["gte"]):
                return False
            if "gt" in expected and not (isinstance(actual, (int, float)) and actual > expected["gt"]):
                return False
            if "lte" in expected and not (isinstance(actual, (int, float)) and actual <= expected["lte"]):
                return False
            if "lt" in expected and not (isinstance(actual, (int, float)) and actual < expected["lt"]):
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
        else:
            if actual != expected:
                return False
    return True


def resolve_advisories(
    shape: TaskShape,
    error_patterns: dict[str, Any] | None = None,
    *,
    max_advisories: int = 5,
) -> tuple[ErrorAdvisory, ...]:
    """Return per-task error advisories ordered by frequency (desc).

    Caps at ``max_advisories`` so the prompt does not balloon when the
    error_patterns file grows large.
    """
    payload = error_patterns or load_error_patterns()
    advisories: list[ErrorAdvisory] = []
    for raw in payload.get("patterns") or []:
        if not isinstance(raw, dict):
            continue
        match = raw.get("match") or {}
        if not isinstance(match, dict):
            continue
        if not _entry_matches(match, shape):
            continue
        action = str(raw.get("action") or "*")
        sig = str(raw.get("error_signature") or "")
        hint = str(raw.get("hint") or "").strip()
        if not hint:
            continue
        try:
            freq = int(raw.get("frequency") or 1)
        except (TypeError, ValueError):
            freq = 1
        advisories.append(
            ErrorAdvisory(action=action, error_signature=sig, hint=hint, frequency=freq)
        )
    advisories.sort(key=lambda a: a.frequency, reverse=True)
    return tuple(advisories[:max_advisories])


def render_advisories_for_prompt(advisories: tuple[ErrorAdvisory, ...]) -> str:
    """Render the advisories as a compact, human-readable hint block."""
    if not advisories:
        return ""
    lines = ["Past failure modes observed on similar tasks (avoid repeating):"]
    for adv in advisories:
        action_label = "any tool" if adv.action == "*" else adv.action
        lines.append(
            f"- [{action_label} → {adv.error_signature} × {adv.frequency}]"
            f" {adv.hint}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_ERROR_PATTERNS_PATH",
    "ErrorAdvisory",
    "error_signature_for",
    "load_error_patterns",
    "save_error_patterns",
    "resolve_advisories",
    "render_advisories_for_prompt",
]
