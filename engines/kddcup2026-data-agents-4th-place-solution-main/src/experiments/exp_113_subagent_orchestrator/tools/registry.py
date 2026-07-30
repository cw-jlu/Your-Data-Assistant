"""Orchestrator + full-loop SQL sub-agent registry for exp_113.

The orchestrator's own tools are limited to inspection (describe_data,
read_doc) + dispatch (ask_sql / answer_from_sql) — both dispatches
spawn a full-loop ReAct sub-agent restricted to SQL tools. Sub-agent's
intermediate steps stay isolated from the orchestrator's context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter
from experiments.exp_113_subagent_orchestrator.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_113_subagent_orchestrator.tools.duckdb_unified import (
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


_SUBAGENT_DELEGATE_REF: dict[str, Any] = {}


def _make_ask_sql_subagent(model: ModelAdapter, parent_tools_ref: dict[str, Any]) -> ToolHandler:
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        from experiments.exp_113_subagent_orchestrator.subagent import run_sql_subagent
        sub_q = action_input.get("sub_question") or action_input.get("question") or action_input.get("q")
        if not sub_q:
            return ToolExecutionResult(ok=False, content={"error": "ask_sql_subagent requires sub_question"})
        result = run_sql_subagent(
            task=task,
            sub_question=str(sub_q),
            model=model,
            parent_tools=parent_tools_ref["tools"],
            max_steps=int(action_input.get("max_steps", 8)),
        )
        if not result.ok:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": result.failure_reason,
                    "n_steps": result.n_steps,
                    "hint": "Rephrase the sub_question more precisely or split into smaller questions.",
                },
            )
        return ToolExecutionResult(
            ok=True,
            content={
                "columns": result.columns,
                "rows": result.rows[:200],
                "row_count": len(result.rows),
                "truncated": len(result.rows) > 200,
                "n_subagent_steps": result.n_steps,
            },
        )
    return handler


def _make_answer_from_sql_subagent(model: ModelAdapter, parent_tools_ref: dict[str, Any]) -> ToolHandler:
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        from experiments.exp_113_subagent_orchestrator.subagent import run_sql_subagent
        sub_q = action_input.get("sub_question") or action_input.get("question") or action_input.get("q")
        if not sub_q:
            return ToolExecutionResult(ok=False, content={"error": "answer_from_sql_subagent requires sub_question"})
        result = run_sql_subagent(
            task=task,
            sub_question=str(sub_q),
            model=model,
            parent_tools=parent_tools_ref["tools"],
            max_steps=int(action_input.get("max_steps", 8)),
        )
        if not result.ok:
            return ToolExecutionResult(
                ok=False,
                content={"error": result.failure_reason, "n_steps": result.n_steps},
            )
        answer = AnswerTable(
            columns=list(result.columns),
            rows=[[("" if v is None else str(v)) for v in r] for r in result.rows],
        )
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "via": "sql_subagent",
                "column_count": len(result.columns),
                "row_count": len(result.rows),
                "n_subagent_steps": result.n_steps,
            },
            is_terminal=True,
            answer=answer,
        )
    return handler


_ALL_SPECS: dict[str, ToolSpec] = {
    "answer": ToolSpec(
        name="answer",
        description="Submit the final answer table directly (= for small literals). Prefer answer_from_sql_subagent.",
        input_schema={"columns": ["column_name"], "rows": [["value_1"]]},
    ),
    "answer_from_sql": ToolSpec(
        name="answer_from_sql",
        description=(
            "Run a SQL query against the unified DuckDB connection and submit its rows as the "
            "final answer. Terminating action. Used by the SUB-AGENT only."
        ),
        input_schema={"sql": "SELECT ... FROM ..."},
    ),
    "describe_data": ToolSpec(
        name="describe_data",
        description=(
            "Show all data sources — every CSV/JSON/sqlite-DB is exposed as a view in DuckDB. "
            "Use this once at the start to plan."
        ),
        input_schema={},
    ),
    "execute_sql": ToolSpec(
        name="execute_sql",
        description=(
            "Run a read-only SQL query against the unified DuckDB connection. Sub-agent's main tool."
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
        description="Preview a CSV file (= inspection only).",
        input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
    ),
    "read_doc": ToolSpec(
        name="read_doc",
        description="Read a text-like document (knowledge.md, README, etc.).",
        input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
    ),
    "read_json": ToolSpec(
        name="read_json",
        description="Preview a JSON file (= inspection only).",
        input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
    ),
}


_ALL_HANDLERS: dict[str, ToolHandler] = {
    "answer": _answer,
    "answer_from_sql": _answer_from_sql,
    "describe_data": _describe_data,
    "execute_sql": _execute_sql,
    "list_context": _list_context,
    "read_csv": _read_csv,
    "read_doc": _read_doc,
    "read_json": _read_json,
}


_ORCHESTRATOR_TOOLS = {
    "answer", "describe_data", "read_doc", "list_context",
    # ask_sql_subagent and answer_from_sql_subagent are added dynamically
    # (= they need a model handle).
}


def _build_orchestrator_specs(model: ModelAdapter) -> dict[str, ToolSpec]:
    specs = {n: s for n, s in _ALL_SPECS.items() if n in _ORCHESTRATOR_TOOLS}
    specs["ask_sql_subagent"] = ToolSpec(
        name="ask_sql_subagent",
        description=(
            "Dispatch a focused sub-question to a full-loop SQL sub-agent. The sub-agent runs "
            "its own ReAct loop (= up to 8 steps) over DuckDB, returns rows + columns. Use this "
            "for any data computation. NON-terminal — orchestrator decides whether the result "
            "is final."
        ),
        input_schema={
            "sub_question": "Plain-English question. Be specific about which columns, "
                            "filters, and aggregation are required.",
            "max_steps": 8,
        },
    )
    specs["answer_from_sql_subagent"] = ToolSpec(
        name="answer_from_sql_subagent",
        description=(
            "Terminal: dispatch a sub-question to the SQL sub-agent AND submit its result as "
            "the final answer. Use when the orchestrator has decomposed the problem into a "
            "single self-contained query."
        ),
        input_schema={"sub_question": "Plain-English question."},
    )
    return specs


def create_default_tool_registry(model: ModelAdapter | None = None) -> ToolRegistry:
    """Build the orchestrator's tool registry (= dispatch + inspection only)."""
    if model is None:
        raise ValueError(
            "exp_113 orchestrator registry requires a model — sub-agent dispatch needs an LLM."
        )
    parent_tools_ref: dict[str, Any] = {}
    handlers = {n: h for n, h in _ALL_HANDLERS.items() if n in _ORCHESTRATOR_TOOLS}
    handlers["ask_sql_subagent"] = _make_ask_sql_subagent(model, parent_tools_ref)
    handlers["answer_from_sql_subagent"] = _make_answer_from_sql_subagent(model, parent_tools_ref)
    specs = _build_orchestrator_specs(model)
    registry = ToolRegistry(specs=specs, handlers=handlers)
    parent_tools_ref["tools"] = create_subagent_tool_registry()
    return registry


def create_subagent_tool_registry() -> ToolRegistry:
    """Build the sub-agent's restricted SQL-only registry."""
    sub_names = {"answer", "answer_from_sql", "describe_data", "execute_sql", "read_doc"}
    specs = {n: s for n, s in _ALL_SPECS.items() if n in sub_names}
    handlers = {n: h for n, h in _ALL_HANDLERS.items() if n in sub_names}
    return ToolRegistry(specs=specs, handlers=handlers)
