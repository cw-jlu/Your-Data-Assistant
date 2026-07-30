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

from experiments.exp_015_cascade_json_fix.tools.registry import (
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


def _make_answer_handler(spec_data: dict[str, Any]):
    expected_col_count = len(spec_data.get("answer_columns", []))

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

        # Auto-unwrap {"table": "...", "records": [...]} envelope.
        # DuckDB reads root-object JSON as a 1-row table with struct columns,
        # so a {table, records} file produces a single row where "records" is a list.
        if (
            len(result_df) == 1
            and "records" in result_df.columns
            and isinstance(result_df["records"].iloc[0], list)
        ):
            result_df = pd.DataFrame(result_df["records"].iloc[0])

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

        # Shape sanity check: force a retry if column count doesn't match spec.
        if expected_col_count > 0 and len(columns) != expected_col_count:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": (
                        f"Shape mismatch: SQL returned {len(columns)} column(s) but "
                        f"spec.answer_columns has {expected_col_count}. "
                        "Rewrite the SQL to produce exactly the expected columns."
                    )
                },
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

    return _answer_from_duckdb


def create_answer_tool_registry(spec_data: dict[str, Any] | None = None) -> ToolRegistry:
    _spec = spec_data or {}
    handler = _make_answer_handler(_spec)
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
                    "-- or JSON {table,records} format:\n"
                    "WITH rows AS (SELECT unnest(records, recursive:=true) FROM read_json_auto('data.json')),\n"
                    "     b AS (SELECT * FROM read_csv_auto('other.csv'))\n"
                    "SELECT rows.x, b.y FROM rows JOIN b ON rows.id = b.id"
                )
            },
        ),
    }
    handlers = {"answer_from_duckdb": handler}
    return ToolRegistry(specs=specs, handlers=handlers)
