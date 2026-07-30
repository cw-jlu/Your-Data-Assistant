from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from experiments.exp_052_filetype_labels.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from experiments.exp_052_filetype_labels.tools.python_exec import (
    execute_python_code,
    execute_python_for_answer,
)
from experiments.exp_052_filetype_labels.tools.sqlite import (
    execute_read_only_sql,
    inspect_sqlite_schema,
)

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


def _coerce_sql_input(task: PublicTask, action_input: dict[str, Any] | str) -> dict[str, Any]:
    """Coerce a bare SQL string to {"path": ..., "sql": ...} using context heuristic.

    When action_input is a plain string, the agent passed raw SQL instead of a
    proper object. We locate the first .db/.sqlite in context to fill the path.
    """
    if not isinstance(action_input, str):
        return action_input
    db_files = sorted(task.context_dir.glob("**/*.db")) + sorted(
        task.context_dir.glob("**/*.sqlite")
    )
    if not db_files:
        raise ValueError(
            "action_input was a bare SQL string but no .db/.sqlite file found in context. "
            'Provide {"path": "...", "sql": "..."} instead.'
        )
    rel_path = db_files[0].relative_to(task.context_dir).as_posix()
    return {"path": rel_path, "sql": action_input}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    action_input = _coerce_sql_input(task, action_input)
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    if isinstance(action_input, str):
        action_input = {"code": action_input}
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer_from_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    if isinstance(action_input, str):
        action_input = {"code": action_input}
    code = str(action_input["code"])
    content = execute_python_for_answer(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    if not content.get("success"):
        return ToolExecutionResult(ok=False, content=content)
    extracted = content.get("answer") or {}
    columns = extracted.get("columns") or []
    rows = extracted.get("rows") or []
    if not columns:
        return ToolExecutionResult(
            ok=False,
            content={
                **content,
                "error": "Extracted answer had no columns.",
            },
        )
    answer = AnswerTable(columns=list(columns), rows=[list(r) for r in rows])
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "output": content.get("output", ""),
        },
        is_terminal=True,
        answer=answer,
    )


def _answer_from_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    action_input = _coerce_sql_input(task, action_input)
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    result = execute_read_only_sql(path, sql, limit=10000)
    columns = [str(c) for c in (result.get("columns") or [])]
    rows = [
        [("" if v is None else str(v)) for v in row]
        for row in (result.get("rows") or [])
    ]
    if not columns:
        return ToolExecutionResult(
            ok=False,
            content={"error": "SQL returned no columns.", **result},
        )
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "truncated": result.get("truncated", False),
        },
        is_terminal=True,
        answer=answer,
    )


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "answer_from_python": ToolSpec(
            name="answer_from_python",
            description=(
                "Run Python in the task context dir and submit the result as the final answer. "
                "Your code MUST set either `answer_df` (a pandas DataFrame) or `answer_table` "
                "(a dict {'columns': [...], 'rows': [[...], ...]}). The columns and rows are "
                "extracted automatically — preferred over manually retyping values into `answer`. "
                "This is a terminating action."
            ),
            input_schema={
                "code": "import pandas as pd\nanswer_df = pd.DataFrame({'count': [1]})",
            },
        ),
        "answer_from_sql": ToolSpec(
            name="answer_from_sql",
            description=(
                "Run a read-only SQL query against a sqlite/db file and submit the result rows "
                "as the final answer table. Use this when the answer is exactly one SQL query — "
                "much safer than retyping result values into `answer`. This is a terminating "
                "action."
            ),
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sql": "SELECT ... FROM ...",
            },
        ),
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
        "answer": _answer,
        "answer_from_python": _answer_from_python,
        "answer_from_sql": _answer_from_sql,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
