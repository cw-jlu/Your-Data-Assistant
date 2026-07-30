"""answer tool: terminal tool for submitting the final answer table."""

from __future__ import annotations

import csv
import io
import pathlib
import shutil
import warnings
from contextlib import suppress
from typing import Annotated, Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError, ParserWarning
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.benchmark.schema import AnswerTable, PublicTask
from agents.tools.contracts import (
    INLINE_ANSWER_CELL_LIMIT,
    INLINE_ANSWER_ROW_LIMIT,
    answer_artifact_expression,
    artifact_answer_size_phrase,
    inline_answer_size_phrase,
)
from agents.tools.decorator import function_tool
from agents.tools.read_csv import classify_dtype
from agents.tools.registry import ToolExecutionResult

_ANSWER_SCRATCH_ROOT = pathlib.Path("/tmp/dabench")
_FROM_CSV_SIZE_LIMIT_BYTES = 5 * 1024 * 1024

_FULLWIDTH_COMMA = "，"


def _answer_dir_for(task: PublicTask) -> pathlib.Path:
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Column header names. Leave empty when using from_csv. "
                "Inline is rejected above 10 rows or 50 cells — use from_csv instead."
            ),
        ),
    ]
    rows: Annotated[
        list[list[str | int | float | bool | None]],
        Field(
            default_factory=list,
            description=(
                "Answer rows; each row's length must match the columns array. "
                "Leave empty when using from_csv. "
                "Inline is rejected above 10 rows or 50 cells — use from_csv instead."
            ),
        ),
    ]
    from_csv: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Absolute path to a CSV file containing the answer table "
                "(header row + data rows). REQUIRED when the answer exceeds "
                "10 rows or 50 cells. When set, `columns` "
                "and `rows` MUST be empty. Convention: write the CSV from "
                "`execute_python` under `os.environ['DABENCH_ANSWER_DIR']`."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _check_payload(self) -> AnswerInput:
        if self.from_csv is not None:
            if self.columns or self.rows:
                raise ValueError("provide either from_csv or columns+rows, not both")
            return self
        if not self.columns:
            raise ValueError("columns must be non-empty when from_csv is not set")
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} cells but columns has {width}")
        return self


def _coerce_row(row: list[str], dtypes: list[str]) -> list[Any]:
    out: list[Any] = []
    for cell, dtype in zip(row, dtypes, strict=True):
        if cell == "":
            out.append(None)
        elif dtype == "int":
            out.append(int(cell))
        elif dtype == "float":
            out.append(float(cell))
        elif dtype == "bool":
            out.append(cell.lower() == "true")
        else:
            out.append(cell)
    return out


def _normalize_header_fullwidth_commas(text: str) -> str | None:
    """Return a copy with unquoted full-width commas fixed in the header only.

    The known recovery case is a hand-written header using `，` while data rows
    use ASCII commas.  Cell values may legitimately contain `，`, so the fallback
    is limited to the first CSV record and only runs after pandas reports a
    header/data width mismatch.
    """
    if _FULLWIDTH_COMMA not in text:
        return None
    header: list[str] = []
    rest_start = len(text)
    in_quotes = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            if in_quotes and i + 1 < n and text[i + 1] == '"':
                header.append('""')
                i += 2
                continue
            in_quotes = not in_quotes
            header.append(ch)
        elif not in_quotes and ch == _FULLWIDTH_COMMA:
            header.append(",")
        elif not in_quotes and ch in {"\n", "\r"}:
            if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
                header.append("\r\n")
                rest_start = i + 2
            else:
                header.append(ch)
                rest_start = i + 1
            break
        else:
            header.append(ch)
        i += 1
    normalized_header = "".join(header)
    if normalized_header == text[:rest_start]:
        return None
    return normalized_header + text[rest_start:]


def _typecast_string_table(
    header: list[str], data: list[list[str]]
) -> tuple[list[str], list[list[Any]], list[str]]:
    """共享的 dtype 推断 + 强转入口：常规路径与 header 救援路径都走这条。"""
    width = len(header)
    dtypes: list[str] = []
    for col_idx in range(width):
        column_values = [row[col_idx] for row in data]
        dtypes.append(classify_dtype(column_values))
    typed = [_coerce_row(row, dtypes) for row in data]
    return header, typed, dtypes


def _from_csv_shape_error(detail: str | None = None) -> str:
    msg = (
        "from_csv rows must have the same number of cells as the header row. "
        "Rewrite the answer CSV with a complete header row, use "
        "pandas.DataFrame.to_csv(..., index=False), and quote text fields that "
        "contain commas or newlines."
    )
    if detail:
        msg += f" Parser detail: {detail}"
    return msg


def _validate_csv_row_widths(raw: str) -> None:
    """Reject ragged rows while allowing intentionally preserved blank lines."""
    reader = csv.reader(io.StringIO(raw))
    try:
        header = next(reader)
    except StopIteration:
        return
    except csv.Error as exc:
        raise ValueError(f"failed to parse from_csv: {exc}") from exc

    width = len(header)
    try:
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != width:
                detail = f"row {row_idx} has {len(row)} cells; header has {width}"
                raise ValueError(_from_csv_shape_error(detail))
    except csv.Error as exc:
        raise ValueError(f"failed to parse from_csv: {exc}") from exc


def parse_answer_csv(path: pathlib.Path) -> tuple[list[str], list[list[Any]], list[str]]:
    """Read and type-cast an answer artifact CSV.

    Returns ``(header, typed_rows, dtypes)``.  All failures raise
    ``ValueError``; the dispatcher translates to ``ToolErrorEvent``.
    """
    if not path.exists():
        raise ValueError(f"from_csv file not found: {path}")
    if path.stat().st_size > _FROM_CSV_SIZE_LIMIT_BYTES:
        raise ValueError("from_csv exceeds 5MB; split the table or use inline columns/rows")
    try:
        raw = path.read_text("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"from_csv is not valid UTF-8: {exc}") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ParserWarning)
            df = pd.read_csv(
                io.StringIO(raw),
                dtype=str,
                engine="c",
                index_col=False,
                keep_default_na=False,
                na_filter=False,
                skip_blank_lines=False,
            )
    except EmptyDataError as exc:
        raise ValueError("from_csv is empty; expected header row + data rows") from exc
    except ParserWarning as exc:
        normalized = _normalize_header_fullwidth_commas(raw)
        if normalized is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ParserWarning)
                    df = pd.read_csv(
                        io.StringIO(normalized),
                        dtype=str,
                        engine="c",
                        index_col=False,
                        keep_default_na=False,
                        na_filter=False,
                        skip_blank_lines=False,
                    )
            except (EmptyDataError, ParserError, ParserWarning):
                pass
            else:
                header = [str(column) for column in df.columns]
                if header and not df.isna().to_numpy().any():
                    _validate_csv_row_widths(normalized)
                    data = df.astype(str).values.tolist()
                    return _typecast_string_table(header, data)
        raise ValueError(_from_csv_shape_error(str(exc))) from exc
    except ParserError as exc:
        raise ValueError(f"failed to parse from_csv: {exc}") from exc
    header = [str(column) for column in df.columns]
    if not header:
        raise ValueError("from_csv header row is empty")
    _validate_csv_row_widths(raw)
    if df.isna().to_numpy().any():
        raise ValueError(_from_csv_shape_error("one or more rows have fewer cells than the header"))
    data = df.astype(str).values.tolist()
    return _typecast_string_table(header, data)


def _prune_null_columns(
    columns: list[str], rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]], list[str]]:
    if not rows or not columns:
        return columns, rows, []

    width = len(columns)
    keep_indices: list[int] = []
    removed: list[str] = []

    for col_idx in range(width):
        all_null = all(
            row[col_idx] is None or (isinstance(row[col_idx], str) and row[col_idx].strip() == "")
            for row in rows
            if col_idx < len(row)
        )
        if all_null and len(rows) > 0:
            removed.append(columns[col_idx])
        else:
            keep_indices.append(col_idx)

    if not removed:
        return columns, rows, []

    pruned_columns = [columns[i] for i in keep_indices]
    pruned_rows = [[row[i] for i in keep_indices] for row in rows]
    return pruned_columns, pruned_rows, removed


def _inline_threshold_error(columns: list[str], rows: list[list[Any]]) -> str | None:
    """Return the inline-size threshold error, or None when the payload is allowed."""
    n_rows = len(rows)
    n_cells = n_rows * len(columns)
    if n_rows > INLINE_ANSWER_ROW_LIMIT or n_cells > INLINE_ANSWER_CELL_LIMIT:
        return (
            f"inline rows payload exceeds artifact-handoff threshold "
            f"(got {n_rows} rows, {n_cells} cells; inline is allowed only for "
            f"<= {INLINE_ANSWER_ROW_LIMIT} rows AND <= {INLINE_ANSWER_CELL_LIMIT} cells; "
            "if either limit is exceeded, use from_csv); "
            f"write the table with DataFrame.to_csv to "
            f"{answer_artifact_expression()} in "
            "execute_python, then resubmit with from_csv (no columns/rows)"
        )
    return None


_ANSWER_DESC = (
    "Submit the final answer table. This is the ONLY way to submit; plain "
    "text answers are ignored. Returns {status, column_count, row_count}. "
    f"For small answers ({inline_answer_size_phrase()}): pass inline columns/rows. "
    f"For {artifact_answer_size_phrase()}: write CSV in execute_python first, "
    "then pass from_csv (inline is rejected at this threshold). "
    'Inline example: answer({"columns": ["name", "total"], '
    '"rows": [["Alice", 100], ["Bob", 200]]}) '
    'CSV example: answer({"from_csv": '
    '"/tmp/dabench/<task_id>/_answer/answer.csv"})'
)


@function_tool(
    description=_ANSWER_DESC,
    input_model=AnswerInput,
    is_terminal=True,
)
def answer(task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    artifact_path = _answer_dir_for(task) / "answer.csv"
    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"from_csv must be an absolute path, got '{args.from_csv}' "
                "(hint: use os.environ['DABENCH_ANSWER_DIR'] to build the path)"
            )
        columns, rows, dtypes = parse_answer_csv(path)
        columns, rows, pruned_cols = _prune_null_columns(columns, rows)
        answer_table = AnswerTable(columns=columns, rows=rows)
        path.unlink(missing_ok=True)
        shutil.rmtree(artifact_path.parent, ignore_errors=True)
        content: dict[str, Any] = {
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "from_csv": {
                "path": str(path),
                "dtypes": dtypes,
                "head_preview": rows[:5],
            },
        }
        if pruned_cols:
            content["pruned_null_columns"] = pruned_cols
        return ToolExecutionResult(
            ok=True,
            content=content,
            is_terminal=True,
            answer=answer_table,
        )
    # Check artifact existence before inline threshold so the retry points at from_csv.
    if artifact_path.exists() and artifact_path.stat().st_size > 0:
        raise ValueError(
            f"answer artifact already exists at {artifact_path} but inline "
            "columns/rows were sent; the artifact is the canonical answer "
            "and inline cannot override it. Resubmit as "
            f'answer({{"from_csv": "{artifact_path}"}}) with no columns/rows.'
        )
    threshold_error = _inline_threshold_error(args.columns, list(args.rows))
    if threshold_error is not None:
        raise ValueError(threshold_error)
    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    columns, rows, pruned_cols = _prune_null_columns(columns, rows)
    answer_table = AnswerTable(columns=columns, rows=rows)
    shutil.rmtree(artifact_path.parent, ignore_errors=True)
    content: dict[str, Any] = {
        "status": "submitted",
        "column_count": len(columns),
        "row_count": len(rows),
    }
    if pruned_cols:
        content["pruned_null_columns"] = pruned_cols
    return ToolExecutionResult(
        ok=True,
        content=content,
        is_terminal=True,
        answer=answer_table,
    )


def build_answer_table_without_side_effects(
    task: PublicTask,
    arguments: dict[str, Any],
) -> AnswerTable | None:
    """Validate and normalize raw answer arguments without committing."""
    try:
        args = AnswerInput.model_validate(arguments)
    except ValidationError:
        return None

    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if not path.is_absolute():
            return None
        try:
            columns, rows, _dtypes = parse_answer_csv(path)
        except (ValueError, OSError):
            return None
        columns, rows, _pruned_cols = _prune_null_columns(columns, rows)
        return AnswerTable(columns=columns, rows=rows)

    artifact_path = _answer_dir_for(task) / "answer.csv"
    try:
        if artifact_path.exists() and artifact_path.stat().st_size > 0:
            return None
    except OSError:
        return None
    # 与 `answer.handler` 同源：阈值检查现在落在 handler，本函数也必须复用
    # 同一份判定，否则验证器拒答的 fallback 会把超阈值 inline 当合法答案晋级。
    if _inline_threshold_error(args.columns, list(args.rows)) is not None:
        return None

    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    columns, rows, _pruned_cols = _prune_null_columns(columns, rows)
    return AnswerTable(columns=columns, rows=rows)


def cleanup_answer_artifacts(task: PublicTask, arguments: dict[str, Any]) -> None:
    """Best-effort cleanup after promoting a fallback answer."""
    try:
        args = AnswerInput.model_validate(arguments)
    except ValidationError:
        return

    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if path.is_absolute():
            with suppress(OSError):
                path.unlink(missing_ok=True)

    shutil.rmtree(_answer_dir_for(task), ignore_errors=True)
