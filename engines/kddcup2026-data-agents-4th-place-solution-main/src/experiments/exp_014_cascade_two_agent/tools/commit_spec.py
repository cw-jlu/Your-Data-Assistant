"""commit_spec tool: validates and commits the exploration spec to terminate Phase 1."""
from __future__ import annotations

from typing import Any

from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_014_cascade_two_agent.tools.registry import (
    ToolExecutionResult,
    ToolSpec,
)

# These keys must be present in the spec for Phase 2 to proceed meaningfully.
_REQUIRED_SPEC_KEYS = frozenset(
    {"data_sources", "key_columns", "join_plan", "filters", "answer_columns"}
)


def _validate_spec(spec: dict[str, Any]) -> None:
    missing = _REQUIRED_SPEC_KEYS - spec.keys()
    if missing:
        raise ValueError(f"spec is missing required keys: {sorted(missing)}")


def handle_commit_spec(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    spec = action_input.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(
            "commit_spec requires action_input.spec to be a JSON object. "
            'Example: {"action": "commit_spec", "action_input": {"spec": {...}}}'
        )
    _validate_spec(spec)
    return ToolExecutionResult(
        ok=True,
        is_terminal=True,
        content={"status": "spec_committed", "spec": spec},
    )


COMMIT_SPEC_SPEC = ToolSpec(
    name="commit_spec",
    description=(
        "Commit your exploration plan as a structured spec and end Phase 1. "
        "Phase 2 will use this spec to write a single DuckDB SQL query. "
        "Required spec keys: data_sources, key_columns, join_plan, filters, answer_columns. "
        "Do NOT include executable SQL here — describe intent only. "
        "This is a terminating action."
    ),
    input_schema={
        "spec": {
            "data_sources": [{"path": "csv/table.csv", "role": "main", "type": "csv"}],
            "key_columns": [{"table": "table", "column": "col", "used_for": "select"}],
            "join_plan": [],
            "filters": [{"natural": "col = 'value'", "rationale": "from question"}],
            "answer_columns": [{"meaning": "count of X", "source": "computed"}],
            "aggregation": {"type": "count", "target": "rows", "notes": ""},
            "domain_notes": [],
        }
    },
)
