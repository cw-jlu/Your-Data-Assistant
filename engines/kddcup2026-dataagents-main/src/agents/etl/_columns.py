"""Column-name canonicalization helpers shared by ETL stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

T = TypeVar("T")


def metadata_by_name(
    names: Iterable[str],
    metadata: Mapping[str, T] | None,
) -> dict[str, T]:
    """Return metadata remapped to the spelling used by *names*.

    ETL schema names can pass through LLM prompts, governance docs, and CSV
    headers with case-only spelling drift.  Metadata dictionaries should follow
    the active schema/header spelling instead of requiring exact key matches.
    """
    if not metadata:
        return {}
    by_lower = {key.lower(): value for key, value in metadata.items()}
    return {name: by_lower[name.lower()] for name in names if name.lower() in by_lower}
