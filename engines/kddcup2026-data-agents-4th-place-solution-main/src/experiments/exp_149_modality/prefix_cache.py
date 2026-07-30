"""Task-scoped prefix-cache routing headers for exp_149 runtime calls."""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
HEADER_NAME = "X-Prefix-Cache-Key"


def prefix_cache_enabled() -> bool:
    raw = os.environ.get("EXP149_PREFIX_CACHE", "1").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    return True


def prefix_cache_key(task_id: str | None) -> str | None:
    """Return a stable key shared by all model requests for one task."""
    if not prefix_cache_enabled() or not task_id:
        return None
    namespace = os.environ.get("EXP149_PREFIX_CACHE_NAMESPACE", "kobushi-phase2-exp149").strip()
    if not namespace:
        return task_id
    return f"{namespace}:{task_id}"


def with_prefix_cache_header(
    headers: dict[str, str] | None,
    task_id: str | None,
) -> dict[str, str]:
    merged = dict(headers or {})
    key = prefix_cache_key(task_id)
    if key:
        merged[HEADER_NAME] = key
    return merged
