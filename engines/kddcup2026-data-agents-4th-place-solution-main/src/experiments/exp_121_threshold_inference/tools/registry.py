"""SQL-only tool registry for exp_111.

NO Python execution. All computation happens in a single DuckDB
connection that exposes every CSV/JSON/sqlite-DB file in the task
context as views/tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from experiments.exp_121_threshold_inference.tools.filesystem import (
    grep_context,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_121_threshold_inference.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
    get_catalog,
)


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
    search = action_input.get("search") or None
    offset = int(action_input.get("offset", 0))
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task, path, max_chars=max_chars, search=search, offset=offset,
        ),
    )


def _grep(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(action_input["pattern"])
    path = action_input.get("path") or None
    glob = action_input.get("glob") or None
    output_mode = str(action_input.get("output_mode", "content"))
    ctx = int(action_input.get("context_lines", 0))
    ci = bool(action_input.get("case_insensitive", True))
    max_results = int(action_input.get("max_results", 50))
    max_chars = int(action_input.get("max_chars", 8000))
    return ToolExecutionResult(
        ok=True,
        content=grep_context(
            task, pattern=pattern, path=path, glob=glob,
            output_mode=output_mode, context_lines=ctx,
            case_insensitive=ci, max_results=max_results, max_chars=max_chars,
        ),
    )


def _describe_data(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content={
            "catalog": get_catalog(task.context_dir),
            "summary": describe_catalog(task.context_dir),
        },
    )


def _coerce_sql_input(action_input: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(action_input, str):
        return {"sql": action_input}
    return action_input


def _execute_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    action_input = _coerce_sql_input(action_input)
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    try:
        result = execute_sql(task.context_dir, sql, limit=limit)
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": str(exc), "sql": sql},
        )
    return ToolExecutionResult(ok=True, content=result)


def _answer_from_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    action_input = _coerce_sql_input(action_input)
    sql = str(action_input["sql"])
    try:
        result = execute_sql(task.context_dir, sql, limit=10000)
    except Exception as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc), "sql": sql})
    columns = [str(c) for c in result.get("columns") or []]
    rows = [
        [("" if v is None else str(v)) for v in row]
        for row in (result.get("rows") or [])
    ]
    if not columns:
        return ToolExecutionResult(ok=False, content={"error": "SQL returned no columns.", **result})
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
            description="Submit the final answer table directly. Terminating action. Prefer answer_from_sql when possible.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "answer_from_sql": ToolSpec(
            name="answer_from_sql",
            description=(
                "Run a SQL query against the unified DuckDB connection and submit the result rows "
                "as the final answer. The DuckDB conn already has every CSV/JSON/sqlite-DB file in "
                "context loaded as a view. Terminating action — use this for the final query."
            ),
            input_schema={"sql": "SELECT ... FROM ..."},
        ),
        "describe_data": ToolSpec(
            name="describe_data",
            description=(
                "Show all data sources in this task — every CSV, JSON, and sqlite-DB is exposed "
                "as a view in DuckDB. Returns view names, source file, and columns. "
                "Call this once at the start to know what is available."
            ),
            input_schema={},
        ),
        "execute_sql": ToolSpec(
            name="execute_sql",
            description=(
                "Run a read-only SQL query (SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN) against "
                "the unified DuckDB connection. Use this for inspection and intermediate queries. "
                "Returns up to `limit` rows."
            ),
            input_schema={"sql": "SELECT ...", "limit": 200},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories under the task context dir.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Preview a CSV file (= inspection only; use SQL for analysis).",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document. Two modes:\n"
                "  - search='<term>': returns paragraphs containing the term "
                "(case-insensitive), joined by `---`. Use this for long docs "
                "to extract specific facts (= names, IDs, definitions).\n"
                "  - offset=<int>: returns text[offset:offset+max_chars]. "
                "Use next_offset from the response to paginate long docs.\n"
                "Default (no search, no offset): returns the first max_chars."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "max_chars": 4000,
                "search": "optional keyword (= grep paragraphs)",
                "offset": 0,
            },
        ),
        "grep": ToolSpec(
            name="grep",
            description=(
                "Regex search across text files in the task context (= md, txt, "
                "csv, json). Inspired by Claude Code's Grep. Three output modes:\n"
                "  - 'content' (default): line-by-line `path:line:text` with "
                "    optional ±N context_lines.\n"
                "  - 'files_with_matches': just the list of paths.\n"
                "  - 'count': match-count per file.\n"
                "Pattern is Python regex. Case-insensitive by default. Optional "
                "`path` (= one file) or `glob` (= e.g. 'doc/*.md') filter."
            ),
            input_schema={
                "pattern": "regex (Python re syntax)",
                "path": "optional, e.g. 'doc/major.md'",
                "glob": "optional, e.g. 'doc/*.md'",
                "output_mode": "'content' | 'files_with_matches' | 'count'",
                "context_lines": 0,
                "case_insensitive": True,
                "max_results": 50,
                "max_chars": 8000,
            },
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Preview a JSON file (= inspection only; use SQL on the view for analysis).",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    handlers = {
        "answer": _answer,
        "answer_from_sql": _answer_from_sql,
        "describe_data": _describe_data,
        "execute_sql": _execute_sql,
        "grep": _grep,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
