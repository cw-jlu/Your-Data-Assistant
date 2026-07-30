"""Orchestrator + sub-agents tool registry for exp_112.

The orchestrator is a planning agent. It does NOT compute directly — for any
data manipulation it dispatches to a SQL or Python specialist (= 1-shot
sub-agents). The orchestrator's direct tools are limited to inspection
(describe_data, list_context, read_*) and dispatch (ask_sql, ask_python).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter
from experiments.exp_112_orchestrator.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_112_orchestrator.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
    get_catalog,
)
from experiments.exp_112_orchestrator.subagent import (
    call_python_specialist,
    call_sql_specialist,
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
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


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


def _make_ask_sql(model: ModelAdapter) -> ToolHandler:
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        sub_q = action_input.get("sub_question") or action_input.get("question") or action_input.get("q")
        if not sub_q:
            return ToolExecutionResult(ok=False, content={"error": "ask_sql requires sub_question"})
        result = call_sql_specialist(task=task, sub_question=str(sub_q), model=model)
        if not result.ok:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": result.error,
                    "sql": result.code,
                    "hint": "Try rephrasing the sub_question or check the catalog with describe_data.",
                },
            )
        return ToolExecutionResult(
            ok=True,
            content={
                "sql": result.code,
                "columns": result.columns,
                "rows": result.rows[:200],
                "row_count": len(result.rows),
                "truncated": len(result.rows) > 200,
            },
        )
    return handler


def _make_ask_python(model: ModelAdapter) -> ToolHandler:
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        sub_q = action_input.get("sub_question") or action_input.get("question") or action_input.get("q")
        if not sub_q:
            return ToolExecutionResult(ok=False, content={"error": "ask_python requires sub_question"})
        input_table = action_input.get("input_table")
        result = call_python_specialist(
            task=task, sub_question=str(sub_q), model=model, input_table=input_table,
        )
        if not result.ok:
            return ToolExecutionResult(
                ok=False,
                content={"error": result.error, "code": result.code[:1500]},
            )
        return ToolExecutionResult(
            ok=True,
            content={
                "code": result.code[:1500],
                "columns": result.columns,
                "rows": result.rows[:200],
                "row_count": len(result.rows),
                "truncated": len(result.rows) > 200,
            },
        )
    return handler


def _make_answer_from_subagent(model: ModelAdapter, kind: str) -> ToolHandler:
    """Terminal version of ask_sql / ask_python: their output IS the final answer."""
    runner = call_sql_specialist if kind == "sql" else call_python_specialist
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        sub_q = action_input.get("sub_question") or action_input.get("question") or action_input.get("q")
        if not sub_q:
            return ToolExecutionResult(ok=False, content={"error": f"answer_from_{kind} requires sub_question"})
        kwargs: dict[str, Any] = dict(task=task, sub_question=str(sub_q), model=model)
        if kind == "python" and action_input.get("input_table"):
            kwargs["input_table"] = action_input["input_table"]
        result = runner(**kwargs)
        if not result.ok:
            return ToolExecutionResult(
                ok=False, content={"error": result.error, "code": result.code[:1500]},
            )
        answer = AnswerTable(
            columns=list(result.columns),
            rows=[[("" if v is None else str(v)) for v in r] for r in result.rows],
        )
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "via": f"answer_from_{kind}",
                "code": result.code[:1500],
                "column_count": len(result.columns),
                "row_count": len(result.rows),
            },
            is_terminal=True,
            answer=answer,
        )
    return handler


def create_default_tool_registry(model: ModelAdapter | None = None) -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table directly. Terminating action. Prefer answer_from_sql when possible.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "ask_sql": ToolSpec(
            name="ask_sql",
            description=(
                "Dispatch a focused sub-question to the SQL specialist. "
                "The specialist sees the catalog + knowledge.md and returns one DuckDB query result "
                "(rows + columns). Use this for any data computation. Non-terminal — the orchestrator "
                "still decides whether the answer is final."
            ),
            input_schema={"sub_question": "Plain-English data question (be specific about columns and aggregations)."},
        ),
        "ask_python": ToolSpec(
            name="ask_python",
            description=(
                "Dispatch a focused sub-question to the Python specialist. The specialist runs "
                "pandas / sqlite3 / duckdb. Use this when SQL is awkward (= row-wise iteration, "
                "regex, multi-step transforms). Optionally pass `input_table` (= rows from a prior "
                "ask_sql) to avoid the specialist re-fetching."
            ),
            input_schema={
                "sub_question": "Plain-English question.",
                "input_table": {"columns": ["..."], "rows": [["..."]]},
            },
        ),
        "answer_from_sql": ToolSpec(
            name="answer_from_sql",
            description=(
                "Terminal: dispatch a sub-question to the SQL specialist AND submit its result "
                "directly as the final answer. Skip this if you need to inspect the result first."
            ),
            input_schema={"sub_question": "Plain-English question."},
        ),
        "answer_from_python": ToolSpec(
            name="answer_from_python",
            description=(
                "Terminal: dispatch a sub-question to the Python specialist AND submit its result "
                "as the final answer."
            ),
            input_schema={
                "sub_question": "Plain-English question.",
                "input_table": {"columns": ["..."], "rows": [["..."]]},
            },
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
                "Direct (non-specialist) SQL for QUICK inspection — schema sanity checks, "
                "value lookups, simple aggregates. For real computation, prefer ask_sql so the "
                "specialist's prompt is loaded. Read-only (SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN)."
            ),
            input_schema={"sql": "SELECT ...", "limit": 50},
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
            description="Read a text-like document (knowledge.md, README, etc.).",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Preview a JSON file (= inspection only; use SQL on the view for analysis).",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    if model is None:
        raise ValueError(
            "exp_112_orchestrator's tool registry requires a model — sub-agent dispatch tools "
            "internally invoke an LLM."
        )
    handlers = {
        "answer": _answer,
        "answer_from_sql": _make_answer_from_subagent(model, "sql"),
        "answer_from_python": _make_answer_from_subagent(model, "python"),
        "ask_sql": _make_ask_sql(model),
        "ask_python": _make_ask_python(model),
        "describe_data": _describe_data,
        "execute_sql": _execute_sql,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
