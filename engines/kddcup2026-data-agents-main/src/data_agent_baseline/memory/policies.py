"""ShapePolicy + resolver — turn a TaskShape into per-task adjustments.

A ``ShapePolicy`` packages everything the runtime needs to know about how
to handle a class of tasks: timeout multiplier, max_steps multiplier, prompt
hints to inject, and tool preferences. Policies are stored declaratively in
``learnings.json`` and merged in priority order so the most-specific match
wins.

The runtime layers consume policies via ``resolve_policy(shape, learnings)``:

- ``runner.py`` reads ``timeout_multiplier`` + ``max_steps_multiplier`` and
  scales the per-task budget.
- ``prompt.py`` reads ``prompt_hints`` and injects them into the task prompt.
- ``react.py`` (via ``ReActAgentConfig``) reads ``preferred_tools`` /
  ``avoid_tools`` and surfaces them in the tool catalog description.

Match semantics
---------------

A learnings entry looks like::

    {
      "match": {"is_heavy": true, "has_large_json": true},
      "policy": { ... },
      "priority": 100,
      "evidence": ["task_249", "task_259"]
    }

The match is an AND over keys/values — every key in ``match`` must equal
the corresponding attribute on ``TaskShape``. Higher ``priority`` wins
when multiple entries match. Hints and tool lists are merged across all
matches (priority decides which wins for scalars).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.memory.task_shape import TaskShape


@dataclass(frozen=True, slots=True)
class ShapePolicy:
    """Per-task overrides derived from a TaskShape match."""

    timeout_multiplier: float = 1.0
    max_steps_multiplier: float = 1.0
    prompt_hints: tuple[str, ...] = ()
    preferred_tools: tuple[str, ...] = ()
    avoid_tools: tuple[str, ...] = ()
    enable_dual_path: bool = False
    notes: tuple[str, ...] = ()  # human-readable evidence trail

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_multiplier": self.timeout_multiplier,
            "max_steps_multiplier": self.max_steps_multiplier,
            "prompt_hints": list(self.prompt_hints),
            "preferred_tools": list(self.preferred_tools),
            "avoid_tools": list(self.avoid_tools),
            "enable_dual_path": self.enable_dual_path,
            "notes": list(self.notes),
        }


_DEFAULT_POLICY = ShapePolicy()


@dataclass(frozen=True, slots=True)
class _PolicyEntry:
    match: dict[str, Any]
    policy: ShapePolicy
    priority: int = 0
    evidence: tuple[str, ...] = ()


def _entry_matches(entry: _PolicyEntry, shape: TaskShape) -> bool:
    shape_dict = shape.to_dict()
    for key, expected in entry.match.items():
        actual = shape_dict.get(key)
        if isinstance(expected, list):
            # Match if any value in the list equals the actual.
            if actual not in expected:
                return False
        elif isinstance(expected, dict):
            # Range expression: {"gte": 100} / {"lt": 50} / {"in": [...]}
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


def _parse_policy(raw: dict[str, Any]) -> ShapePolicy:
    return ShapePolicy(
        timeout_multiplier=float(raw.get("timeout_multiplier", 1.0)),
        max_steps_multiplier=float(raw.get("max_steps_multiplier", 1.0)),
        prompt_hints=tuple(str(x) for x in raw.get("prompt_hints", []) or []),
        preferred_tools=tuple(str(x) for x in raw.get("preferred_tools", []) or []),
        avoid_tools=tuple(str(x) for x in raw.get("avoid_tools", []) or []),
        enable_dual_path=bool(raw.get("enable_dual_path", False)),
        notes=tuple(str(x) for x in raw.get("notes", []) or []),
    )


def _parse_entry(raw: dict[str, Any]) -> _PolicyEntry:
    return _PolicyEntry(
        match=dict(raw.get("match") or {}),
        policy=_parse_policy(raw.get("policy") or {}),
        priority=int(raw.get("priority") or 0),
        evidence=tuple(str(x) for x in raw.get("evidence") or []),
    )


def _merge(base: ShapePolicy, overlay: ShapePolicy, *, overlay_priority_wins: bool) -> ShapePolicy:
    """Merge two policies. Scalars from ``overlay`` win when ``overlay_priority_wins``;
    lists / hints are concatenated (deduped) regardless.
    """
    timeout = overlay.timeout_multiplier if overlay_priority_wins else base.timeout_multiplier
    max_steps = overlay.max_steps_multiplier if overlay_priority_wins else base.max_steps_multiplier
    enable_dp = overlay.enable_dual_path if overlay_priority_wins else base.enable_dual_path

    def _dedupe(seq: tuple[str, ...]) -> tuple[str, ...]:
        seen: list[str] = []
        for item in seq:
            if item and item not in seen:
                seen.append(item)
        return tuple(seen)

    return ShapePolicy(
        timeout_multiplier=timeout if timeout != 1.0 else base.timeout_multiplier,
        max_steps_multiplier=max_steps if max_steps != 1.0 else base.max_steps_multiplier,
        prompt_hints=_dedupe(base.prompt_hints + overlay.prompt_hints),
        preferred_tools=_dedupe(base.preferred_tools + overlay.preferred_tools),
        avoid_tools=_dedupe(base.avoid_tools + overlay.avoid_tools),
        enable_dual_path=enable_dp or base.enable_dual_path,
        notes=_dedupe(base.notes + overlay.notes),
    )


def resolve_policy(shape: TaskShape, learnings: dict[str, Any]) -> ShapePolicy:
    """Look up the ShapePolicy that applies to ``shape`` given ``learnings``.

    All matching entries contribute. Scalars (timeout / max_steps / dual_path)
    take the value from the highest-priority matching entry; hints + tool lists
    are concatenated across every matching entry (deduped, in priority order).
    """
    entries_raw = learnings.get("shape_policies") or []
    matched: list[_PolicyEntry] = []
    for raw in entries_raw:
        try:
            entry = _parse_entry(raw)
        except (TypeError, ValueError):
            continue
        if _entry_matches(entry, shape):
            matched.append(entry)

    if not matched:
        return _DEFAULT_POLICY

    matched.sort(key=lambda e: e.priority, reverse=True)
    accumulator = ShapePolicy()
    for i, entry in enumerate(matched):
        # The first (highest priority) entry wins scalars; subsequent entries
        # only contribute hint/tool concatenation.
        accumulator = _merge(
            accumulator,
            entry.policy,
            overlay_priority_wins=(i == 0),
        )
    return accumulator


__all__ = ["ShapePolicy", "resolve_policy"]
