"""Phase 1 tool registry: exploration tools + commit_spec. No answer tools.

Excluding answer/answer_from_* forces the agent to commit a spec rather
than jumping straight to an answer, ensuring Phase 2 always gets a plan.
"""
from __future__ import annotations

from experiments.exp_014_cascade_two_agent.tools.commit_spec import (
    COMMIT_SPEC_SPEC,
    handle_commit_spec,
)
from experiments.exp_014_cascade_two_agent.tools.registry import (
    ToolRegistry,
    ToolSpec,
    _execute_context_sql,
    _execute_python,
    _inspect_sqlite_schema,
    _list_context,
    _read_csv,
    _read_doc,
    _read_json,
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
)


def create_exploration_tool_registry() -> ToolRegistry:
    specs = {
        "commit_spec": COMMIT_SPEC_SPEC,
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sql": "SELECT ...",
                "limit": 200,
            },
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    handlers = {
        "commit_spec": handle_commit_spec,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
