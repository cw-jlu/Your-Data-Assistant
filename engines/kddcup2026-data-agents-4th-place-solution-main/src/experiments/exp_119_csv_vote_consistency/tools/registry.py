"""SQL-only tool registry for exp_111.

NO Python execution. All computation happens in a single DuckDB
connection that exposes every CSV/JSON/sqlite-DB file in the task
context as views/tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter
from experiments.exp_119_csv_vote_consistency.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_119_csv_vote_consistency.tools.duckdb_unified import (
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


def _make_voted_answer_handler(
    voter_model: ModelAdapter,
    sub_question_provider,  # () → str (= the original task question)
    format_csv_provider,    # () → str (= the inferred CSV header)
    k: int = 5,
) -> ToolHandler:
    """Replace simple answer_from_sql with voting + consistency check.

    The agent submits its candidate SQL via `answer_from_sql(sql=...)`. The
    handler treats it as candidate 0, generates K-1 alternatives at temp=1.0,
    executes all, and votes. Commits ONLY if a unique max-vote winner exists.
    Otherwise returns a non-terminal error so the agent retries.
    """
    from experiments.exp_119_csv_vote_consistency.voted_terminal import voted_terminal

    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        ai = _coerce_sql_input(action_input)
        sql = str(ai["sql"])
        result = voted_terminal(
            task=task, agent_sql=sql,
            sub_question=sub_question_provider(),
            format_csv=format_csv_provider(),
            voter_model=voter_model, k=k,
        )
        if not result.ok:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": result.error,
                    "decision": result.decision,
                    "cluster_sizes": result.cluster_sizes,
                    "hint": ("Vote tied — multiple SQL interpretations gave different "
                             "results. Re-examine the question, especially aggregation "
                             "axis vs. row-level filter, and resubmit a more decisive SQL.")
                    if result.decision == "tied" else
                    "All candidates failed; check your SQL syntax.",
                },
            )
        answer = AnswerTable(
            columns=list(result.columns),
            rows=[[("" if v is None else str(v)) for v in r] for r in result.rows],
        )
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "via": "voted_terminal",
                "winning_sql": result.winning_sql,
                "cluster_sizes": result.cluster_sizes,
                "decision": result.decision,
                "column_count": len(result.columns),
                "row_count": len(result.rows),
            },
            is_terminal=True,
            answer=answer,
        )
    return handler


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


def create_default_tool_registry(
    voter_model: ModelAdapter | None = None,
    sub_question_provider=None,
    format_csv_provider=None,
    k: int = 5,
) -> ToolRegistry:
    """Build the registry. If voter_model is provided, answer_from_sql is
    replaced by the voted-terminal handler (= K-candidate vote + unique-max
    consistency check). Otherwise behaves like exp_111 sql_only."""
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
                "Submit a candidate SQL as your final answer. The system internally generates "
                f"{k}-candidate alternatives at temperature 1.0 and votes on result rows; the "
                "answer is committed only when a UNIQUE max-vote winner exists. If the vote ties, "
                "you'll be asked to retry with a more decisive SQL. Terminating action when committed."
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
            description="Read a text-like document (knowledge.md, README, etc.).",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Preview a JSON file (= inspection only; use SQL on the view for analysis).",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    if voter_model is None:
        # Backward-compat: use simple answer_from_sql like exp_111
        from kobushi_core.benchmark.schema import AnswerTable as _AT  # noqa
        def _simple_answer_from_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
            ai = _coerce_sql_input(action_input)
            sql = str(ai["sql"])
            try:
                result = execute_sql(task.context_dir, sql, limit=10000)
            except Exception as exc:
                return ToolExecutionResult(ok=False, content={"error": str(exc), "sql": sql})
            cols = [str(c) for c in (result.get("columns") or [])]
            rows = [[("" if v is None else str(v)) for v in r] for r in (result.get("rows") or [])]
            if not cols:
                return ToolExecutionResult(ok=False, content={"error": "no columns", **result})
            return ToolExecutionResult(
                ok=True,
                content={"status": "submitted", "column_count": len(cols), "row_count": len(rows)},
                is_terminal=True,
                answer=AnswerTable(columns=cols, rows=rows),
            )
        ans_handler = _simple_answer_from_sql
    else:
        if sub_question_provider is None or format_csv_provider is None:
            raise ValueError("voter_model requires sub_question_provider and format_csv_provider")
        ans_handler = _make_voted_answer_handler(
            voter_model, sub_question_provider, format_csv_provider, k=k,
        )
    handlers = {
        "answer": _answer,
        "answer_from_sql": ans_handler,
        "describe_data": _describe_data,
        "execute_sql": _execute_sql,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
