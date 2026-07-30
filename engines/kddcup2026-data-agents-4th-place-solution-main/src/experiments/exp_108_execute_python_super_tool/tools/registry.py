from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from experiments.exp_108_execute_python_super_tool.tools.filesystem import (
    resolve_context_path,
)
from experiments.exp_108_execute_python_super_tool.tools.python_exec import (
    execute_python_code,
    execute_python_for_answer,
)
from experiments.exp_108_execute_python_super_tool.tools.sqlite import (
    execute_read_only_sql,
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
                "Pre-bound names available: pd, np, duckdb, sqlite3, json, Path, ctx. "
                "This is a terminating action."
            ),
            input_schema={
                "code": "answer_df = pd.read_csv(ctx / 'items.csv')\nanswer_df = answer_df[['col']]",
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
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python code with the task context directory as the working directory. "
                "Returns captured stdout as `output`. "
                f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS}s. "
                "Pre-bound names: pd, np, duckdb, sqlite3, json, Path, ctx. "
                "Use `pd.read_csv(ctx / 'foo.csv')` for CSVs, "
                "`duckdb.query('SELECT ...').df()` for analytical SQL, "
                "`sqlite3.connect(ctx / 'db.sqlite')` for SQLite inspection."
            ),
            input_schema={
                "code": "df = pd.read_csv(ctx / 'items.csv')\nprint(df.head())",
            },
        ),
    }
    handlers = {
        "answer": _answer,
        "answer_from_python": _answer_from_python,
        "answer_from_sql": _answer_from_sql,
        "execute_python": _execute_python,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
