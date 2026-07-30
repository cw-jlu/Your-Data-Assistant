"""Tool registry: ``FunctionTool`` / ``ToolRegistry`` + factory.

Responsibilities:
- ``FunctionTool`` / ``ToolRegistry``: immutable structures
- ``ToolExecutionResult`` / ``ToolHandler``: canonical return type + handler signature
- ``create_default_tool_registry()``: builds the standard tool set
- Input validation via ``tool.input_model`` (no global lookup table)

Implementation details live in sibling modules:
- ``tools/contracts.py``        — shared constants
- ``tools/*.py``                — self-contained tool definitions (incl. descriptions)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from agents.benchmark.schema import AnswerTable, PublicTask


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Uniform return structure for tool handlers."""

    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, Any], ToolExecutionResult]


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Complete tool definition: schema, handler, and metadata."""

    name: str
    description: str
    json_schema: dict[str, Any] | None
    handler: ToolHandler
    is_terminal: bool = False
    input_model: type[BaseModel] | None = None

    def with_description(self, description: str) -> FunctionTool:
        """Return a copy with a different description."""
        return dataclasses.replace(self, description=description)


def _format_validation_error(tool_name: str, error: ValidationError) -> ValueError:
    first = error.errors()[0]
    err_type = str(first["type"])
    loc = cast(tuple[Any, ...], first.get("loc", ()))
    field_path = ".".join(str(seg) for seg in loc) if loc else ""
    ctx = first.get("ctx", {})

    if err_type == "missing":
        return ValueError(f"{tool_name}: field '{field_path}' is required")
    if err_type == "extra_forbidden":
        return ValueError(f"{tool_name}: extra field '{field_path}' is not permitted")
    if err_type.endswith("_type"):
        expected = err_type.removesuffix("_type")
        actual = type(first.get("input")).__name__
        return ValueError(f"{tool_name}: field '{field_path}' must be a {expected}, got {actual}")
    if err_type == "greater_than_equal":
        bound = ctx.get("ge")
        actual_value = first.get("input")
        return ValueError(
            f"{tool_name}: field '{field_path}' must be >= {bound}, got {actual_value}"
        )
    if err_type == "too_short":
        min_len = ctx.get("min_length")
        return ValueError(
            f"{tool_name}: field '{field_path}' must contain at least {min_len} item(s)"
        )
    if err_type == "value_error":
        msg = str(first["msg"]).removeprefix("Value error, ")
        if field_path:
            return ValueError(f"{tool_name}: field '{field_path}' {msg}")
        return ValueError(f"{tool_name}: {msg}")
    return ValueError(f"{tool_name}: field '{field_path}' {first['msg']}")


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    """Immutable name → FunctionTool mapping."""

    definitions: Mapping[str, FunctionTool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        if action not in self.definitions:
            raise KeyError(f"Unknown tool: {action}")
        defn = self.definitions[action]
        model_cls = defn.input_model
        if model_cls is None:
            return defn.handler(task, action_input)
        try:
            validated = model_cls.model_validate(action_input)
        except ValidationError as exc:
            raise _format_validation_error(action, exc) from exc
        return defn.handler(task, validated)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for name in sorted(self.definitions):
            definition = self.definitions[name]
            if definition.json_schema is None:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.json_schema,
                    },
                }
            )
        return tools


def create_default_tool_registry(
    *,
    explore_tool: FunctionTool | None = None,
    run_etl_tool: FunctionTool | None = None,
) -> ToolRegistry:
    """Build the default tool registry from top-level tool specs.

    Registration order (fixed by convention):
    1. explore / inspect_files  — first-step exploration
    2. run_etl                  — on-demand prose-to-CSV conversion
    3. preview_file             — single-file preview (CSV/JSON/SQLite/doc/PDF)
    4. execute_context_sql      — virtual-context SQLite queries and indexes
    5. grep_context             — regex search across files
    6. execute_python           — subprocess Python execution
    7. answer                   — terminal: submit answer (only terminal tool)
    """
    from agents.tools.answer import answer
    from agents.tools.execute_python import execute_python
    from agents.tools.execute_sql import execute_context_sql
    from agents.tools.grep_context import grep_context
    from agents.tools.inspect_files import inspect_files
    from agents.tools.preview import preview_file
    from agents.tools.run_etl import create_run_etl_tool_definition

    first_tool = explore_tool or inspect_files
    etl_tool = run_etl_tool or create_run_etl_tool_definition()
    definitions: list[FunctionTool] = [
        first_tool,
        etl_tool,
        preview_file,
        execute_context_sql,
        grep_context,
        execute_python,
        answer,
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})


__all__ = [
    "FunctionTool",
    "ToolExecutionResult",
    "ToolHandler",
    "ToolRegistry",
    "create_default_tool_registry",
]
