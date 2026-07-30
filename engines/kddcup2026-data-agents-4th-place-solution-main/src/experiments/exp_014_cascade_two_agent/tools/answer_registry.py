"""Phase 2 tool registry: answer_from_duckdb only.

Providing only this one tool makes it physically impossible for the agent to
submit a hand-written answer, eliminating the row/column transcription errors
that caused ~7 zero_recall failures in exp_012.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from kobushi_core.benchmark.schema import AnswerTable, PublicTask

from experiments.exp_014_cascade_two_agent.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
)


def _safe_cell_str(v: object) -> str:
    """Convert a DuckDB result cell to string.

    pd.isna() raises "truth value of array is ambiguous" when v is a
    nested struct/list column (DuckDB returns these as numpy arrays).
    We catch that and fall through to str().
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _answer_from_duckdb(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    # Coerce bare SQL string — the model occasionally omits the wrapping object.
    if isinstance(action_input, str):
        action_input = {"sql": action_input}
    sql = action_input.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(
            'answer_from_duckdb requires action_input.sql to be a non-empty string. '
            'Example: {"sql": "SELECT col FROM read_csv_auto(\'file.csv\')"}'
        )

    import duckdb  # local import; duckdb is in pyproject.toml as duckdb>=1.5.0

    original_cwd = os.getcwd()
    try:
        # CWD = context_dir so relative paths in read_csv_auto / ATTACH work directly.
        os.chdir(task.context_dir)
        conn = duckdb.connect()
        try:
            result_df = conn.execute(sql).fetchdf()
        finally:
            conn.close()
    except Exception as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    finally:
        os.chdir(original_cwd)

    try:
        columns = [str(c) for c in result_df.columns]
        rows = [
            [_safe_cell_str(v) for v in row]
            for row in result_df.to_numpy(dtype=object)
        ]
    except Exception as exc:
        return ToolExecutionResult(ok=False, content={"error": f"result serialization failed: {exc}"})

    if not columns:
        return ToolExecutionResult(
            ok=False, content={"error": "DuckDB query returned no columns."}
        )
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        is_terminal=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
        },
        answer=answer,
    )


def create_answer_tool_registry() -> ToolRegistry:
    specs = {
        "answer_from_duckdb": ToolSpec(
            name="answer_from_duckdb",
            description=(
                "Execute a DuckDB SQL query and submit the result as the final answer. "
                "The working directory is the task context dir, so use relative paths. "
                "DuckDB supports read_csv_auto('file.csv'), read_json_auto('file.json'), "
                "and ATTACH 'file.sqlite' AS db (TYPE sqlite) for multi-source JOINs. "
                "This is the ONLY available tool — use it to produce the answer. "
                "This is a terminating action."
            ),
            input_schema={
                "sql": (
                    "SELECT col FROM read_csv_auto('file.csv') WHERE ...;\n"
                    "-- or multi-source:\n"
                    "WITH a AS (SELECT * FROM read_json_auto('data.json', records=true)),\n"
                    "     b AS (SELECT * FROM read_csv_auto('other.csv'))\n"
                    "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
                )
            },
        ),
    }
    handlers = {"answer_from_duckdb": _answer_from_duckdb}
    return ToolRegistry(specs=specs, handlers=handlers)
