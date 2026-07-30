"""inspect_files tool: one-shot context file inventory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT
from agents.tools import constants
from agents.tools.decorator import function_tool
from agents.tools.preview import preview_pdf_summary
from agents.tools.read_csv import profile_columns, summarize_csv
from agents.tools.read_doc import summarize_doc
from agents.tools.read_json import summarize_json
from agents.tools.registry import ToolExecutionResult
from agents.tools.units import load_conversions


def _summarize_pdf(path: Path) -> dict[str, Any]:
    """Summarize a PDF using record-id entity paragraph samples."""
    summary = preview_pdf_summary(path)
    summary.pop("format", None)
    return summary


def _summarize_sqlite(path: Path) -> dict[str, Any]:
    """Summarize SQLite tables, row counts, dtypes, and column profiles."""
    tables: dict[str, dict[str, Any]] = {}
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (name,) in table_rows:
            if not isinstance(name, str):
                continue
            quoted = name.replace('"', '""')
            cursor = conn.execute(f'PRAGMA table_info("{quoted}")')
            columns = [row[1] for row in cursor.fetchall()]
            count_cursor = conn.execute(f'SELECT COUNT(*) FROM "{quoted}"')
            row_count = count_cursor.fetchone()[0]
            table_info: dict[str, Any] = {"columns": columns, "row_count": row_count}

            if row_count == 0 or len(columns) > constants.SQLITE_PROFILE_MAX_COLUMNS:
                if len(columns) > constants.SQLITE_PROFILE_MAX_COLUMNS:
                    table_info["profile_skipped"] = "too_many_columns"
                tables[name] = table_info
                continue

            sample_cursor = conn.execute(
                f'SELECT * FROM "{quoted}" LIMIT {constants.INSPECT_FILES_PROFILE_SAMPLE}'
            )
            raw_rows = sample_cursor.fetchall()
            sample_rows = [[str(v) if v is not None else "" for v in row] for row in raw_rows]
            dtypes, profile = profile_columns(columns, sample_rows)
            table_info["dtypes"] = dtypes
            table_info["profile"] = profile
            table_info["profile_sample_rows"] = len(sample_rows)
            tables[name] = table_info
    finally:
        conn.close()
    return {"tables": tables}


def _format_for_inspect(rel_path: str, abs_path: Path) -> dict[str, Any]:
    """Route one file to the appropriate inspect_files summarizer."""
    suffix = abs_path.suffix.lower()
    size = abs_path.stat().st_size
    base: dict[str, Any] = {
        "path": rel_path,
        "size": size,
    }
    if suffix in constants.CSV_EXTS:
        base["format"] = "csv"
        base["supported"] = True
        base["schema"] = summarize_csv(abs_path)
        return base
    if suffix in constants.JSON_EXTS:
        base["format"] = "json"
        base["supported"] = True
        base["schema"] = summarize_json(abs_path)
        return base
    if suffix in constants.SQLITE_EXTS:
        base["format"] = "sqlite"
        base["supported"] = True
        base["schema"] = _summarize_sqlite(abs_path)
        return base
    if suffix in constants.PDF_EXTS:
        base["format"] = "pdf"
        base["supported"] = True
        base["schema"] = _summarize_pdf(abs_path)
        return base
    if suffix in constants.DOC_EXTS:
        base["format"] = "markdown" if suffix == ".md" else "text"
        base["supported"] = True
        base["schema"] = summarize_doc(abs_path)
        if abs_path.name.lower() == "knowledge.md" and size <= constants.KNOWLEDGE_MD_PRELOAD_LIMIT:
            try:
                base["schema"]["full_text"] = abs_path.read_text(encoding="utf-8", errors="replace")
                base["schema"]["knowledge_preloaded"] = True
            except OSError:
                pass
        return base
    base["format"] = suffix.lstrip(".") or "unknown"
    base["supported"] = False
    base["reason"] = constants.UNSUPPORTED_HINT_BY_EXT.get(
        suffix,
        "Unsupported by inspect_files; use execute_python or another preview tool.",
    )
    return base


def inspect_context_files(task: PublicTask) -> dict[str, Any]:
    """Scan context_dir and return lightweight schemas for supported files."""
    context_root = task.context_dir.resolve()
    files_summary: list[dict[str, Any]] = []
    unsupported_formats: set[str] = set()

    candidates: list[tuple[str, Path]] = []
    for child in context_root.rglob("*"):
        if not child.is_file():
            continue
        if any(part.startswith(".") for part in child.relative_to(context_root).parts):
            continue
        rel = child.relative_to(context_root).as_posix()
        candidates.append((rel, child))
    candidates.sort(key=lambda item: item[0])

    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"

    truncated = len(candidates) > constants.INSPECT_FILES_MAX_FILES
    for rel_path, abs_path in candidates[: constants.INSPECT_FILES_MAX_FILES]:
        entry = _format_for_inspect(rel_path, abs_path)
        if not entry["supported"]:
            unsupported_formats.add(str(entry["format"]))
        if cache_dir.is_dir():
            units_path = cache_dir / f"{abs_path.stem}_units.json"
            if units_path.is_file():
                col_conversions = {c.field: c.label for c in load_conversions(units_path)}
                if col_conversions:
                    schema: dict[str, Any] | None = entry.get("schema")
                    if isinstance(schema, dict) and "columns" in schema:
                        reordered: dict[str, Any] = {}
                        for key in schema:
                            reordered[key] = schema[key]
                            if key == "columns":
                                reordered["column_conversions"] = col_conversions
                        entry["schema"] = reordered
                    elif isinstance(schema, dict):
                        schema["column_conversions"] = col_conversions
        files_summary.append(entry)

    result: dict[str, Any] = {
        "context_root": str(context_root),
        "files": files_summary,
        "unsupported_formats": sorted(unsupported_formats),
        "file_count": len(files_summary),
        "truncated": truncated,
    }
    return result


@function_tool
def inspect_files(task: PublicTask) -> ToolExecutionResult:
    """Use FIRST to bootstrap context exploration. Returns a one-shot summary
    of every file under context: {path, size, format, schema} where schema
    contains columns, dtypes, row_count, profile (min/max/null_rate/distinct
    values) for CSV/SQLite, keys/length for JSON, line_count/head for docs,
    and page_count/entity_groups_sample for PDF. Also includes
    column_conversions when unit metadata is available. This replaces calling
    preview_file on each file individually for initial discovery. Does NOT read
    full data — for deeper inspection use preview_file, for filtering or
    aggregation use execute_context_sql or execute_python.
    Example: inspect_files({})"""
    return ToolExecutionResult(ok=True, content=inspect_context_files(task))
