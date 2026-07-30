"""Shared Pydantic field factories for tool input models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from agents.tools.contracts import PATH_CONVENTION_NOTE


def path_field(*, file_kind: str, examples: tuple[str, ...]) -> Any:
    """Path field with a standardized description."""
    example_text = ", ".join(repr(ex) for ex in examples)
    return Field(
        description=(
            f"Relative path to a {file_kind}. Examples: {example_text}. " + PATH_CONVENTION_NOTE
        ),
    )


def positive_int_field(*, description: str) -> Any:
    """Positive integer field constraint (ge=1)."""
    return Field(ge=1, description=description)
