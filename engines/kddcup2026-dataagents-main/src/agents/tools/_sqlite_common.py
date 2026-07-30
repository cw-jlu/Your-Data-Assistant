"""Shared SQLite helpers for tool modules."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from agents.tools.contracts import SQLITE_QUERY_TIMEOUT_SECONDS

_SPECIAL_CHAR_RE = re.compile(r"[() /\-]")

_NON_SQLITE_PATH_REDIRECT: dict[str, str] = {
    ".csv": "`preview_file` or `execute_python` (full scan with pandas)",
    ".json": "`preview_file` or `execute_python` (nested extraction)",
    ".md": "`preview_file`",
    ".txt": "`preview_file`",
}


def maybe_raise_non_sqlite_path_error(path: Path, exc: sqlite3.DatabaseError) -> None:
    """Translate sqlite's opaque non-database error into an actionable tool hint."""
    if "file is not a database" not in str(exc).lower():
        return
    ext = path.suffix.lower()
    redirect = _NON_SQLITE_PATH_REDIRECT.get(ext, "`execute_python`")
    raise ValueError(
        f"Path '{path.name}' is not a sqlite database "
        f"(extension: {ext or 'none'}). "
        f"For this file type, use {redirect} instead of "
        f"preview_file / execute_context_sql."
    ) from exc


def guard_non_sqlite_path(path: Path) -> None:
    """Reject known non-SQLite extensions before opening a connection."""
    ext = path.suffix.lower()
    redirect = _NON_SQLITE_PATH_REDIRECT.get(ext)
    if redirect is not None:
        raise ValueError(
            f"Path '{path.name}' is not a sqlite database "
            f"(extension: {ext}). "
            f"For this file type, use {redirect} instead of "
            f"preview_file / execute_context_sql."
        ) from None


def _install_query_timeout(
    conn: sqlite3.Connection, timeout: float = SQLITE_QUERY_TIMEOUT_SECONDS
) -> None:
    """Install a progress handler that raises OperationalError on timeout."""
    deadline = time.monotonic() + timeout

    def _check() -> int:
        if time.monotonic() >= deadline:
            return 1  # non-zero → interrupt
        return 0

    conn.set_progress_handler(_check, 10_000)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rewrite_code_span(span: str, identifiers: list[str]) -> str:
    pattern = re.compile("|".join(re.escape(name) for name in identifiers))
    return pattern.sub(lambda m: _quote_identifier(m.group(0)), span)


def _auto_quote_code_spans(sql: str, identifiers: list[str]) -> str:
    """Rewrite only executable SQL text, leaving literals/comments/quoted names intact."""
    out: list[str] = []
    start = 0
    i = 0
    n = len(sql)

    def flush_code(until: int) -> None:
        nonlocal start
        if until > start:
            out.append(_rewrite_code_span(sql[start:until], identifiers))
        start = until

    while i < n:
        ch = sql[i]

        if ch == "'":
            flush_code(i)
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if ch in {'"', "`", "["}:
            flush_code(i)
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close in {'"', "`"} and j + 1 < n and sql[j + 1] == close:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if sql.startswith("--", i):
            flush_code(i)
            j = sql.find("\n", i + 2)
            j = n if j == -1 else j + 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if sql.startswith("/*", i):
            flush_code(i)
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(sql[i:j])
            i = j
            start = i
            continue

        i += 1

    flush_code(n)
    return "".join(out)


def auto_quote_identifiers(sql: str, conn: sqlite3.Connection) -> str:
    """Auto-quote table/column names containing special characters like ``() / -``."""
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]

    def _mentioned(name: str) -> bool:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
        return bool(pat.search(sql))

    identifiers: list[str] = []
    for tbl in tables:
        if _mentioned(tbl) or _quote_identifier(tbl) in sql:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote_identifier(tbl)})")]
            identifiers.extend(cols)
        identifiers.append(tbl)

    need_quote = sorted(
        {name for name in identifiers if _SPECIAL_CHAR_RE.search(name)},
        key=len,
        reverse=True,
    )
    if not need_quote:
        return sql
    return _auto_quote_code_spans(sql, need_quote)


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode with query timeout."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    _install_query_timeout(conn)
    return conn


def connect_writable(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in read-write mode (for DDL like CREATE INDEX)."""
    conn = sqlite3.connect(str(path.resolve()))
    _install_query_timeout(conn)
    return conn


# ---------------------------------------------------------------------------
# Subquery column validation
# ---------------------------------------------------------------------------

_SUBQUERY_RE = re.compile(
    r"""\(\s*SELECT\s+                 # opening paren + SELECT
    (?:DISTINCT\s+)?                   # optional DISTINCT
    (\w+)                              # captured column name
    \s+FROM\s+                         # FROM keyword
    (\w+)                              # captured table name
    """,
    re.IGNORECASE | re.VERBOSE,
)


def validate_subquery_columns(sql: str, schema: dict[str, list[str]]) -> list[str]:
    """Check that columns in (SELECT col FROM table) exist in the named table.

    *schema* maps table names (lower-cased) to lists of column names (lower-cased).
    Returns a list of warning strings (empty if everything checks out).
    """
    warnings: list[str] = []
    for match in _SUBQUERY_RE.finditer(sql):
        col = match.group(1).lower()
        table = match.group(2).lower()
        if table not in schema:
            continue
        if col in schema[table]:
            continue
        present_in = [t for t, cols in schema.items() if col in cols]
        if present_in:
            warnings.append(
                f"Column '{match.group(1)}' does not exist in table "
                f"'{match.group(2)}' (found in: {', '.join(present_in)}). "
                f"SQLite silently resolves this to an outer-scope column, "
                f"which almost certainly produces wrong results."
            )
    return warnings
