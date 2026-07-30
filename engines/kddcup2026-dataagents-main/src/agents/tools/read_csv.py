"""read_csv tool: CSV schema + head/tail preview."""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Annotated, Any

from agents.benchmark.schema import PublicTask
from agents.tools import constants
from agents.tools._fields import path_field
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


def classify_dtype(values: list[str]) -> str:
    """Infer a coarse dtype from non-empty string samples."""
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "empty"
    bool_set = {"true", "false"}
    if all(v.lower() in bool_set for v in non_empty):
        return "bool"
    all_int = True
    for v in non_empty:
        candidate = v.lstrip("-+")
        if not candidate.isdigit():
            all_int = False
            break
    if all_int:
        return "int"
    all_float = True
    for v in non_empty:
        try:
            float(v)
        except ValueError:
            all_float = False
            break
    if all_float:
        return "float"
    return "str"


def profile_columns(
    columns: list[str],
    sample_rows: list[list[str]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Infer dtypes and lightweight column profiles from string sample rows."""
    column_count = len(columns)
    dtype_sample = sample_rows[: constants.INSPECT_FILES_DTYPE_SAMPLE]
    dtypes: list[str] = []
    for col_idx in range(column_count):
        col_vals = [row[col_idx] if col_idx < len(row) else "" for row in dtype_sample]
        dtypes.append(classify_dtype(col_vals))

    profile: list[dict[str, Any]] = []
    profile_count = len(sample_rows)
    for col_idx in range(column_count):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in sample_rows]
        null_count = sum(1 for v in col_values if v.strip() == "")
        null_rate = round(null_count / profile_count, 3) if profile_count > 0 else 0.0
        col_profile: dict[str, Any] = {"null_rate": null_rate}
        col_dtype = dtypes[col_idx] if col_idx < len(dtypes) else "str"

        if col_dtype in ("int", "float"):
            numeric_vals: list[float] = []
            for v in col_values:
                v_stripped = v.strip()
                if v_stripped:
                    with contextlib.suppress(ValueError):
                        numeric_vals.append(float(v_stripped))
            if numeric_vals:
                if col_dtype == "int":
                    col_profile["min"] = int(min(numeric_vals))
                    col_profile["max"] = int(max(numeric_vals))
                else:
                    col_profile["min"] = min(numeric_vals)
                    col_profile["max"] = max(numeric_vals)
        else:
            non_empty = sorted({v.strip() for v in col_values if v.strip()})
            if len(non_empty) <= constants.INSPECT_FILES_DISTINCT_VALUES_CAP:
                col_profile["values"] = non_empty
            else:
                col_profile["unique_count"] = len(non_empty)
                col_profile["sample_values"] = non_empty[:5]

        profile.append(col_profile)

    return dtypes, profile


def read_csv_preview(task: PublicTask, relative_path: str) -> dict[str, Any]:
    """Return CSV schema plus head/tail samples. This is not full data access."""
    path = resolve_context_path(task, relative_path)
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "columns": [],
            "dtypes": [],
            "row_count": 0,
            "head": [],
            "tail": [],
        }

    header = rows[0]
    data_rows = rows[1:]
    row_count = len(data_rows)
    head_rows = data_rows[: constants.READ_CSV_HEAD_ROWS]
    tail_start = max(constants.READ_CSV_HEAD_ROWS, row_count - constants.READ_CSV_TAIL_ROWS)
    tail_rows = data_rows[tail_start:]

    column_count = len(header)
    dtypes: list[str] = []
    sample_for_dtype = head_rows[: constants.INSPECT_FILES_DTYPE_SAMPLE]
    for column_index in range(column_count):
        column_values = [
            row[column_index] if column_index < len(row) else "" for row in sample_for_dtype
        ]
        dtypes.append(classify_dtype(column_values))

    return {
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        "head": head_rows,
        "tail": tail_rows,
    }


def summarize_csv(path: Path) -> dict[str, Any]:
    """Stream a CSV enough for inspect_files schema and column profiling."""
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "columns": [],
                "dtypes": [],
                "row_count": 0,
                "dtypes_confidence": "approx",
            }
        sample_rows: list[list[str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(sample_rows) < constants.INSPECT_FILES_PROFILE_SAMPLE:
                sample_rows.append(row)
    dtypes, profile = profile_columns(header, sample_rows)

    return {
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        "dtypes_confidence": "approx",
        "profile": profile,
        "profile_sample_rows": min(len(sample_rows), row_count),
    }


@function_tool
def read_csv(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(file_kind="CSV file", examples=("csv/member.csv", "data/train.csv")),
    ],
) -> ToolExecutionResult:
    """Preview a single CSV file: returns {columns, dtypes, row_count,
    head (first 20 data rows), tail (last 5 data rows)}. Use when you
    need to inspect column names, data types, or sample values for one
    CSV before deciding on a query strategy. This is a SAMPLE preview,
    not full data access — rows between head and tail are not returned.
    For filtering, aggregation, joins, or any full-file scan, use
    execute_python instead. The path parameter is relative to the task
    context directory (e.g. 'csv/member.csv', not 'context/csv/member.csv').
    Example: read_csv({"path": "csv/member.csv"})"""
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path))
