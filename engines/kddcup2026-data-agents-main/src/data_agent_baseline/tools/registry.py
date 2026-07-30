from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.scoring.answer_validator import (
    ValidationReport,
    validate_answer,
)
from data_agent_baseline.scoring.normalize import normalize_answer_table
from data_agent_baseline.tools.filesystem import (
    dataframe_describe,
    dataframe_head,
    inspect_file,
    list_context_tree,
    read_archive_listing,
    read_csv_preview,
    read_doc_preview,
    read_excel_preview,
    read_image_meta,
    read_json_preview,
    read_parquet_preview,
    read_pdf_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.python_kernel import PersistentKernel
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema
from data_agent_baseline.tools.streaming_json import (
    streaming_json_aggregate,
    streaming_json_count,
    streaming_json_keys,
)

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30
PYTHON_KERNEL_MODES: frozenset[str] = frozenset({"persistent", "ephemeral"})

# Conditional terminal: when validate_answer surfaces blocking warnings on
# the FIRST `_answer` call for a task, we return them as observation instead
# of committing the answer. The agent can then re-emit (corrected) or pass
# `confirm: true` to bypass the warnings. After this many bypassable rounds
# we always commit so the agent can never get stuck.
ANSWER_VALIDATION_MAX_BYPASS = 1


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
    normalized_answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _dataframe_describe(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 10_000))
    head_rows = int(action_input.get("head_rows", 5))
    return ToolExecutionResult(
        ok=True,
        content=dataframe_describe(task, path, max_rows=max_rows, head_rows=head_rows),
    )


def _dataframe_head(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    n = int(action_input.get("n", 10))
    return ToolExecutionResult(ok=True, content=dataframe_head(task, path, n=n))


def _read_pdf(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_pages = int(action_input.get("max_pages", 5))
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(
        ok=True, content=read_pdf_preview(task, path, max_pages=max_pages, max_chars=max_chars)
    )


def _read_excel(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    sheet_raw = action_input.get("sheet")
    sheet = str(sheet_raw) if isinstance(sheet_raw, str) and sheet_raw else None
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(
        ok=True, content=read_excel_preview(task, path, sheet=sheet, max_rows=max_rows)
    )


def _read_parquet(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(
        ok=True, content=read_parquet_preview(task, path, max_rows=max_rows)
    )


def _read_image_meta(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    return ToolExecutionResult(ok=True, content=read_image_meta(task, path))


def _read_archive_listing(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_entries = int(action_input.get("max_entries", 200))
    return ToolExecutionResult(
        ok=True, content=read_archive_listing(task, path, max_entries=max_entries)
    )


def _inspect_file(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    return ToolExecutionResult(ok=True, content=inspect_file(task, path))


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
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    # v5 S-7: Auto-fallback for CSV-on-SQL confusion. The error patterns memory
    # logged 7 medium "no such table" + 4 easy "file is not a database" on v4
    # — the agent kept calling execute_context_sql on csv/ paths or
    # directories. Try the original sqlite path first; if it raises one of
    # those two signatures, load the CSV(s) into in-memory sqlite and retry.
    try:
        result = execute_read_only_sql(path, sql, limit=limit)
        return ToolExecutionResult(ok=True, content=result)
    except sqlite3.DatabaseError as exc:
        message = str(exc).lower()
        # Three signatures we know are CSV/dir-on-DB confusion:
        #   "file is not a database" — path is a non-DB file (likely .csv)
        #   "no such table" — path is a real DB but query targets a CSV table name
        #   "disk i/o error" — sqlite3 was handed a directory path
        if not any(s in message for s in ("file is not a database", "no such table", "disk i/o error")):
            raise
        fallback = _csv_sqlite_fallback(path, sql, limit, original_error=str(exc))
        if fallback is None:
            raise
        return ToolExecutionResult(ok=True, content=fallback)


def _csv_sqlite_fallback(
    path: Path, sql: str, limit: int, *, original_error: str
) -> dict[str, Any] | None:
    """Load CSV(s) into in-memory SQLite and re-run the query.

    Returns ``None`` when the path doesn't look like CSV-loadable (e.g. a true
    SQLite DB that's just missing a table — in that case the caller should
    re-raise so the agent sees the original error). Returns a regular result
    dict otherwise, with ``fallback_used: True`` and ``original_error`` so
    the agent can verify the answer if it cares.

    Tables are named after each CSV file's stem. If ``path`` is a single CSV
    file, only that one is loaded under its stem. If ``path`` is a directory,
    every ``*.csv`` inside is loaded.
    """
    if path.is_file() and path.suffix.lower() != ".csv":
        return None  # not a csv — original sqlite error stands
    csv_paths: list[Path]
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            return None
    elif path.is_file():
        csv_paths = [path]
    else:
        return None
    try:
        import pandas as pd  # noqa: PLC0415 — defer import to fallback path only
    except ImportError:
        return None
    conn = sqlite3.connect(":memory:")
    try:
        loaded = []
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception:  # noqa: BLE001 — skip unparseable; agent can investigate
                continue
            table_name = csv_path.stem.replace("-", "_").replace(" ", "_")
            df.to_sql(table_name, conn, index=False, if_exists="replace")
            loaded.append({"table": table_name, "rows": int(len(df)), "csv": str(csv_path.name)})
        if not loaded:
            return None
        normalized = sql.lstrip().lower()
        if not normalized.startswith(("select", "with", "pragma")):
            return None  # caller's safety check still applies
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        return {
            "path": str(path),
            "columns": column_names,
            "rows": [list(row) for row in rows[:limit]],
            "row_count": len(rows[:limit]),
            "truncated": truncated,
            "fallback_used": True,
            "original_error": original_error,
            "loaded_tables": loaded,
        }
    finally:
        conn.close()


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _streaming_json_keys(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    records_path = str(action_input.get("records_path", "records") or "records")
    sample_records = int(action_input.get("sample_records", 50))
    content = streaming_json_keys(
        task, path, records_path=records_path, sample_records=sample_records
    )
    return ToolExecutionResult(ok="error" not in content, content=content)


def _streaming_json_count(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    records_path = str(action_input.get("records_path", "records") or "records")
    raw_filter = action_input.get("filter")
    filt = dict(raw_filter) if isinstance(raw_filter, dict) else None
    filter_op = str(action_input.get("filter_op", "eq") or "eq")
    content = streaming_json_count(
        task, path, records_path=records_path, filter=filt, filter_op=filter_op
    )
    return ToolExecutionResult(ok="error" not in content, content=content)


def _streaming_json_aggregate(
    task: PublicTask, action_input: dict[str, Any]
) -> ToolExecutionResult:
    path = str(action_input["path"])
    field = str(action_input.get("field", "") or "")
    operation = str(action_input.get("operation", "") or "")
    records_path = str(action_input.get("records_path", "records") or "records")
    raw_filter = action_input.get("filter")
    filt = dict(raw_filter) if isinstance(raw_filter, dict) else None
    filter_op = str(action_input.get("filter_op", "eq") or "eq")
    content = streaming_json_aggregate(
        task,
        path,
        field=field,
        operation=operation,
        records_path=records_path,
        filter=filt,
        filter_op=filter_op,
    )
    return ToolExecutionResult(ok="error" not in content, content=content)


def _build_answer_payload(
    action_input: dict[str, Any],
) -> tuple[AnswerTable, AnswerTable, Any, bool]:
    """Validate the action_input shape and produce raw + normalized answers.

    Returns ``(raw_answer, normalized_answer, normalization_report, confirm)``.
    Raises ``ValueError`` if the columns / rows shape is invalid (caught by
    the agent loop and surfaced as an ``__error__`` observation).
    """
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
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
    normalized_answer, normalization_report = normalize_answer_table(answer)
    confirm = bool(action_input.get("confirm", False))
    return answer, normalized_answer, normalization_report, confirm


def _answer_terminal_result(
    answer: AnswerTable,
    normalized_answer: AnswerTable,
    normalization_report: Any,
    validation: ValidationReport,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(answer.columns),
            "row_count": len(answer.rows),
            "normalization": {
                "column_policies": list(normalization_report.column_policies),
                "cell_changes": normalization_report.cell_changes,
            },
            "validation": validation.to_dict(),
        },
        is_terminal=True,
        answer=answer,
        normalized_answer=normalized_answer,
    )


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """Stateless `_answer` — used when validation gate is disabled.

    Always commits the answer terminally. The conditional-terminal variant
    is built by ``_make_answer_handler`` and is the registry default.
    """
    answer, normalized_answer, normalization_report, _confirm = _build_answer_payload(action_input)
    validation = validate_answer(
        answer,
        normalized=normalized_answer,
        question="",
    )
    return _answer_terminal_result(answer, normalized_answer, normalization_report, validation)


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    _kernels: dict[str, PersistentKernel] = field(default_factory=dict)
    # Per-task counter of how many times we have soft-rejected an `_answer`
    # call due to validation warnings. Capped by ANSWER_VALIDATION_MAX_BYPASS
    # so the agent cannot loop forever; cleaned up in cleanup_task.
    _answer_validation_count: dict[str, int] = field(default_factory=dict)

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)

    def get_or_create_kernel(self, task: PublicTask) -> PersistentKernel:
        kernel = self._kernels.get(task.task_id)
        if kernel is None:
            kernel = PersistentKernel(task.context_dir)
            self._kernels[task.task_id] = kernel
        return kernel

    def cleanup_task(self, task_id: str) -> None:
        kernel = self._kernels.pop(task_id, None)
        if kernel is not None:
            kernel.close()
        self._answer_validation_count.pop(task_id, None)

    def cleanup_all(self) -> None:
        for kernel in list(self._kernels.values()):
            kernel.close()
        self._kernels.clear()
        self._answer_validation_count.clear()


def _make_execute_python_persistent(registry: "ToolRegistry") -> ToolHandler:
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        code = str(action_input["code"])
        kernel = registry.get_or_create_kernel(task)
        content = kernel.execute(code, timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS)
        return ToolExecutionResult(ok=bool(content.get("success")), content=content)

    return handler


def _make_answer_handler(registry: "ToolRegistry") -> ToolHandler:
    """Conditional-terminal `_answer`: blocks once on validator warnings.

    Behaviour:
    - First call with blocking warnings (no ``confirm``): returns ``is_terminal=False``
      with the warnings as observation. The agent can re-emit a corrected
      answer or pass ``confirm: true`` on the next call.
    - Second call (or any call with ``confirm``, or no blocking warnings):
      commits the answer terminally.

    The per-task bypass counter is reset by ``ToolRegistry.cleanup_task``,
    which the agent's run() invokes in a finally block.
    """

    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        answer, normalized_answer, normalization_report, confirm = _build_answer_payload(action_input)
        validation = validate_answer(
            answer,
            normalized=normalized_answer,
            question=task.question,
        )

        bypass_count = registry._answer_validation_count.get(task.task_id, 0)
        # v_next: first-call (bypass_count == 0) ignores `confirm: true`.
        # v6 task_180 forensic: agent emitted 153 rows where gold is 9 and
        # set confirm:true on the FIRST call, bypassing the duplicate-rows
        # validator entirely. Forcing a retry on first warning means
        # severe over-emission gets at least one corrective pass.
        is_first_call = bypass_count == 0
        soft_reject = (
            validation.has_blocking
            and (is_first_call or not confirm)
            and bypass_count < ANSWER_VALIDATION_MAX_BYPASS
        )
        if soft_reject:
            registry._answer_validation_count[task.task_id] = bypass_count + 1
            suggestion = (
                "Address the blocking warnings, then re-call answer with the "
                "fixed table. If you have already verified the answer is correct "
                "as-is, re-call answer with `confirm: true` to bypass."
            )
            if is_first_call and confirm:
                suggestion = (
                    "First-call `confirm: true` is ignored to enforce one "
                    "validator pass. Re-check the warnings (especially "
                    "duplicate_rows / column_count_mismatch); if the answer "
                    "is genuinely correct, re-call with `confirm: true` again "
                    "(bypass will succeed on the second call)."
                )
            return ToolExecutionResult(
                ok=False,
                is_terminal=False,
                content={
                    "status": "validation_blocking",
                    "warnings": [w.to_dict() for w in validation.warnings],
                    "suggestion": suggestion,
                    "bypass_remaining": ANSWER_VALIDATION_MAX_BYPASS - (bypass_count + 1),
                },
            )

        return _answer_terminal_result(answer, normalized_answer, normalization_report, validation)

    return handler


def create_default_tool_registry(*, python_kernel_mode: str = "persistent") -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table. This is the only valid terminating action. "
                "On the first call, a quick validator may surface blocking warnings (e.g. all-null "
                "column, suspicious column count vs the question) — in that case the call returns "
                "without committing and you should re-emit a corrected table OR re-call with "
                "confirm=true to bypass. The validator's purpose is to catch the v2 failure modes "
                "(over/under-emission); pass confirm=true once you are confident the answer is correct."
            ),
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
                "confirm": False,
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds. "
                "In persistent mode (default) the namespace is preserved across calls within "
                "the same task, so variables defined in one call are available in the next. "
                "A timeout or kernel crash resets the namespace and the result carries "
                "namespace_reset=True."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "dataframe_describe": ToolSpec(
            name="dataframe_describe",
            description=(
                "Read a CSV/TSV file (cap at max_rows) and return shape, dtypes, "
                "pandas.describe() summary, and a small head preview. Use this as a "
                "cheap prepass on large tabular files before deciding what queries to run."
            ),
            input_schema={
                "path": "relative/path/to/file.csv",
                "max_rows": 10_000,
                "head_rows": 5,
            },
        ),
        "dataframe_head": ToolSpec(
            name="dataframe_head",
            description=(
                "Read the first n rows of a CSV/TSV with proper dtype inference. "
                "Lighter than dataframe_describe."
            ),
            input_schema={
                "path": "relative/path/to/file.csv",
                "n": 10,
            },
        ),
        "read_pdf": ToolSpec(
            name="read_pdf",
            description=(
                "Extract text from a .pdf inside context. Returns per-page text up to "
                "max_pages, with the total truncated to max_chars. Deterministic — "
                "uses pypdf, no OCR or vision pipeline."
            ),
            input_schema={
                "path": "relative/path/to/file.pdf",
                "max_pages": 5,
                "max_chars": 4000,
            },
        ),
        "read_excel": ToolSpec(
            name="read_excel",
            description=(
                "Read sheet names + sampled rows from .xlsx/.xlsm inside context. "
                "First row is treated as the header. Pass `sheet` to target a "
                "specific sheet name; defaults to the first sheet."
            ),
            input_schema={
                "path": "relative/path/to/file.xlsx",
                "sheet": "Sheet1",
                "max_rows": 20,
            },
        ),
        "read_parquet": ToolSpec(
            name="read_parquet",
            description=(
                "Read Parquet schema + sampled head rows from a .parquet inside "
                "context."
            ),
            input_schema={
                "path": "relative/path/to/file.parquet",
                "max_rows": 20,
            },
        ),
        "read_image_meta": ToolSpec(
            name="read_image_meta",
            description=(
                "Image metadata only — format, mode, dimensions, byte size. "
                "team1438 single-model policy: NO OCR or vision-LLM pipeline runs "
                "on the file. Use this to inventory raster formats; do not "
                "expect content extraction."
            ),
            input_schema={
                "path": "relative/path/to/file.png",
            },
        ),
        "read_archive_listing": ToolSpec(
            name="read_archive_listing",
            description=(
                "List entries inside a .zip/.tar/.tar.gz/.tgz file. Names + sizes "
                "only — does not extract contents. Use this to inventory an "
                "archive, then read individual files separately if needed."
            ),
            input_schema={
                "path": "relative/path/to/file.zip",
                "max_entries": 200,
            },
        ),
        "inspect_file": ToolSpec(
            name="inspect_file",
            description=(
                "Hierarchical catalog: dispatch by file extension to surface "
                "type-specific metadata in one cheap call. Returns ext_kind + "
                "size + per-type summary (CSV columns, JSON top-level keys, "
                "SQLite tables, PDF page count, parquet schema, image dimensions, "
                "archive entry sample, doc head preview) PLUS recommended_tool "
                "so you can pick the next reader without guessing. Prefer this "
                "over running multiple read_* calls speculatively."
            ),
            input_schema={
                "path": "relative/path/to/file",
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
        "streaming_json_keys": ToolSpec(
            name="streaming_json_keys",
            description=(
                "Discover the schema of a large JSON record array WITHOUT loading the file. "
                "Streams via ijson and samples the first N records (default 50), returning the "
                "set of keys observed plus example values. Use this on JSON files >50 MB instead "
                "of read_json (capped). The default records_path is 'records' (matches the "
                "DABench convention {'table': ..., 'records': [...]}); pass a dotted path for "
                "nested arrays (e.g. 'payload.items')."
            ),
            input_schema={
                "path": "relative/path/to/file.json",
                "records_path": "records",
                "sample_records": 50,
            },
        ),
        "streaming_json_count": ToolSpec(
            name="streaming_json_count",
            description=(
                "Count records in a JSON array without loading the file. Optionally apply a "
                "flat AND-filter (dotted-path keys → exact-match values). filter_op may be "
                "eq (default), gt, lt, gte, lte for numeric comparisons. Returns "
                "{total, matched}."
            ),
            input_schema={
                "path": "relative/path/to/file.json",
                "records_path": "records",
                "filter": {"status": "active"},
                "filter_op": "eq",
            },
        ),
        "streaming_json_aggregate": ToolSpec(
            name="streaming_json_aggregate",
            description=(
                "Aggregate over a numeric field (dotted path inside each record) WITHOUT "
                "loading the file. operation ∈ {count, sum, avg, min, max}. Optionally "
                "filtered like streaming_json_count. Returns "
                "{result, n, skipped, matched_records}. Skipped counts records where the "
                "field was non-numeric."
            ),
            input_schema={
                "path": "relative/path/to/file.json",
                "field": "score",
                "operation": "avg",
                "records_path": "records",
                "filter": {"status": "active"},
                "filter_op": "eq",
            },
        ),
    }
    if python_kernel_mode not in PYTHON_KERNEL_MODES:
        raise ValueError(
            f"python_kernel_mode must be one of {sorted(PYTHON_KERNEL_MODES)}, got {python_kernel_mode!r}"
        )

    handlers: dict[str, ToolHandler] = {
        "answer": _answer,  # placeholder; replaced below by conditional-terminal variant
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,  # placeholder; replaced below for persistent mode
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "dataframe_describe": _dataframe_describe,
        "dataframe_head": _dataframe_head,
        "read_pdf": _read_pdf,
        "read_excel": _read_excel,
        "read_parquet": _read_parquet,
        "read_image_meta": _read_image_meta,
        "read_archive_listing": _read_archive_listing,
        "inspect_file": _inspect_file,
        "streaming_json_keys": _streaming_json_keys,
        "streaming_json_count": _streaming_json_count,
        "streaming_json_aggregate": _streaming_json_aggregate,
    }
    registry = ToolRegistry(specs=specs, handlers=handlers)
    registry.handlers["answer"] = _make_answer_handler(registry)
    if python_kernel_mode == "persistent":
        registry.handlers["execute_python"] = _make_execute_python_persistent(registry)
    return registry
