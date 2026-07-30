"""grep_context tool: regex search across all context files."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.constants import CSV_EXTS, DOC_EXTS, JSON_EXTS, SQLITE_EXTS
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult

TEXT_EXTS = CSV_EXTS | JSON_EXTS | DOC_EXTS
ALL_SEARCHABLE_EXTS = TEXT_EXTS | SQLITE_EXTS

GREP_MAX_RESULTS = 30
GREP_MAX_LINE_CHARS = 500
GREP_MAX_FILE_BYTES = 50 * 1024 * 1024
SQLITE_SAMPLE_PER_TABLE = 3


def _grep_text_file(
    path: Path,
    rel_path: str,
    compiled: re.Pattern[str],
    remaining: int,
) -> list[dict[str, Any]]:
    """Search a text file line by line, returning up to remaining matches."""
    if path.stat().st_size > GREP_MAX_FILE_BYTES:
        return []
    hits: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if compiled.search(line):
                    text = line.rstrip("\n\r")
                    if len(text) > GREP_MAX_LINE_CHARS:
                        text = text[:GREP_MAX_LINE_CHARS] + "..."
                    hits.append({"file": rel_path, "line": line_no, "text": text})
                    if len(hits) >= remaining:
                        break
    except OSError:
        pass
    return hits


def _grep_sqlite(
    path: Path,
    rel_path: str,
    pattern: str,
    compiled: re.Pattern[str],
    remaining: int,
) -> list[dict[str, Any]]:
    """Search text columns across all SQLite tables with the compiled regex."""
    hits: list[dict[str, Any]] = []
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:

        def regexp(_pattern: str, value: object) -> int:
            if value is None:
                return 0
            return 1 if compiled.search(str(value)) else 0

        conn.create_function("regexp", 2, regexp)
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (table_name,) in tables:
            if not isinstance(table_name, str):
                continue
            quoted = table_name.replace('"', '""')
            cols_info = conn.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            text_cols = [
                row[1] for row in cols_info if isinstance(row[2], str) and "text" in row[2].lower()
            ]
            if not text_cols:
                text_cols = [row[1] for row in cols_info if isinstance(row[1], str)]
            if not text_cols:
                continue
            for col in text_cols:
                col_quoted = col.replace('"', '""')
                sql = f'SELECT COUNT(*) FROM "{quoted}" WHERE "{col_quoted}" REGEXP ?'
                try:
                    total = conn.execute(sql, [pattern]).fetchone()[0]
                except sqlite3.Error:
                    continue
                if total == 0:
                    continue
                sample_sql = (
                    f'SELECT rowid, "{col_quoted}" FROM "{quoted}" '
                    f'WHERE "{col_quoted}" REGEXP ? LIMIT {SQLITE_SAMPLE_PER_TABLE}'
                )
                try:
                    sample_rows = conn.execute(sample_sql, [pattern]).fetchall()
                except sqlite3.Error:
                    sample_rows = []
                samples: list[str] = []
                for row in sample_rows:
                    val = str(row[1]) if row[1] is not None else ""
                    if len(val) > GREP_MAX_LINE_CHARS:
                        val = val[:GREP_MAX_LINE_CHARS] + "..."
                    samples.append(val)
                hits.append(
                    {
                        "file": rel_path,
                        "table": table_name,
                        "column": col,
                        "total_matches": total,
                        "samples": samples,
                    }
                )
                if len(hits) >= remaining:
                    break
            if len(hits) >= remaining:
                break
    finally:
        conn.close()
    return hits


def run_grep_context(
    task: PublicTask,
    pattern: str,
    *,
    path_filter: str | None = None,
    max_results: int = GREP_MAX_RESULTS,
) -> dict[str, Any]:
    """Search for a regex pattern across all searchable context files."""
    context_root = task.context_dir.resolve()

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}", "pattern": pattern}

    candidates: list[tuple[str, Path]] = []
    for child in context_root.rglob("*"):
        if not child.is_file():
            continue
        if any(part.startswith(".") for part in child.relative_to(context_root).parts):
            continue
        suffix = child.suffix.lower()
        if suffix not in ALL_SEARCHABLE_EXTS:
            continue
        rel = child.relative_to(context_root).as_posix()
        candidates.append((rel, child))
    candidates.sort(key=lambda item: item[0])

    if path_filter:
        candidates = [(rel, path) for rel, path in candidates if path_filter in rel]

    all_hits: list[dict[str, Any]] = []
    files_searched = 0

    for rel_path, abs_path in candidates:
        if len(all_hits) >= max_results:
            break
        remaining = max_results - len(all_hits)
        suffix = abs_path.suffix.lower()

        if suffix in TEXT_EXTS:
            all_hits.extend(_grep_text_file(abs_path, rel_path, compiled, remaining))
        elif suffix in SQLITE_EXTS:
            all_hits.extend(_grep_sqlite(abs_path, rel_path, pattern, compiled, remaining))
        files_searched += 1

    return {
        "pattern": pattern,
        "files_searched": files_searched,
        "match_count": len(all_hits),
        "truncated": len(all_hits) >= max_results,
        "results": all_hits,
    }


@function_tool
def grep_context(
    task: PublicTask,
    pattern: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Regex pattern (case-insensitive). "
                "E.g. '\\d{4}-\\d{2}-\\d{2}' for dates, "
                "'(?:Alice|Bob)' for names, '价格|金额' for keywords."
            ),
        ),
    ],
    path: Annotated[
        str | None,
        Field(
            description=(
                "Optional path substring filter. When set, only files whose "
                "path contains this string are searched."
            ),
        ),
    ] = None,
) -> ToolExecutionResult:
    """Search for a regex pattern across ALL context files at once: CSV
    column values, JSON fields, Markdown/text content, and SQLite text
    columns. Use when you need to check whether a specific value, ID, or
    format exists anywhere in the data without reading each file
    individually. Do NOT use for aggregation or counting — use
    execute_context_sql or execute_python for that. Returns up to 30
    matching lines/rows with file path and location. For SQLite hits,
    reports {table, column, total_matches, samples} instead of raw lines.
    The pattern parameter is a Python regex (case-insensitive); the
    optional path parameter filters to files whose path contains that
    substring.
    Example: grep_context({"pattern": "全国"})
    With path filter: grep_context({"pattern": "\\d{4}-\\d{2}", "path": "csv/"})"""
    content = run_grep_context(task, pattern=pattern, path_filter=path)
    return ToolExecutionResult(ok=True, content=content)
