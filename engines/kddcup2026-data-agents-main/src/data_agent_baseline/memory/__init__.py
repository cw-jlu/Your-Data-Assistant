"""Persistent learning layer (v7).

Captures observations from public-set benchmark runs and turns them into
shape-keyed policies that can be applied to hidden-set tasks. The whole
system is leaderboard-independent — every public benchmark run feeds the
recorder, the recorder updates ``learnings.json``, and the next container
build embeds the updated learnings so eval-time tasks get the current
best policy without us ever needing leaderboard feedback.

Modules
-------
- ``task_shape``  — TaskShape dataclass + ``classify_task(task)``
- ``policies``    — ShapePolicy + ``resolve_policy(shape, learnings)``
- ``learnings``   — load/save the bundled JSON file
- ``recorder``    — host-side analyzer that ingests trace.json + scores
                    and proposes ``learnings.json`` updates
"""

from data_agent_baseline.memory.error_patterns import (
    DEFAULT_ERROR_PATTERNS_PATH,
    ErrorAdvisory,
    error_signature_for,
    load_error_patterns,
    render_advisories_for_prompt,
    resolve_advisories,
    save_error_patterns,
)
from data_agent_baseline.memory.learnings import (
    DEFAULT_LEARNINGS_PATH,
    load_learnings,
    save_learnings,
)
from data_agent_baseline.memory.policies import (
    ShapePolicy,
    resolve_policy,
)
from data_agent_baseline.memory.task_shape import TaskShape, classify_task

__all__ = [
    "TaskShape",
    "classify_task",
    "ShapePolicy",
    "resolve_policy",
    "DEFAULT_LEARNINGS_PATH",
    "load_learnings",
    "save_learnings",
    "DEFAULT_ERROR_PATTERNS_PATH",
    "ErrorAdvisory",
    "error_signature_for",
    "load_error_patterns",
    "render_advisories_for_prompt",
    "resolve_advisories",
    "save_error_patterns",
]
