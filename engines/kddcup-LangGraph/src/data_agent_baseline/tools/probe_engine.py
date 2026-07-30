"""Probe engine for executing data-level queries against task data files."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb

from data_agent_baseline.inspectors.semantic_catalog import iter_logical_tables
from data_agent_baseline.tools.duckdb_schema import (
    asset_view_name,
    connect_sqlite_read_only,
    create_duckdb_views,
    json_schema_has_records_fields,
    quote_duckdb_identifier,
    sqlite_view_names,
)


def _validate_read_only_sql(sql: str) -> None:
    normalized = sql.strip().lstrip("(").lstrip()
    if not normalized.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Only SELECT/WITH statements are allowed. Got: {sql[:120]}")


def _strip_leading_sql_comments(sql: str) -> str:
    """Remove standalone SQL comments before read-only validation.

    Models often include natural-language comments in generated SQL batches.
    DuckDB can execute those comments, but our read-only guard validates the
    first non-whitespace token.  Strip only full-line comments so inline SQL
    semantics are left intact.
    """
    lines = sql.splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if not stripped:
            start += 1
            continue
        if stripped.startswith("--"):
            start += 1
            continue
        break
    return "\n".join(lines[start:]).strip()


def _clean_probe_queries(queries: list[str]) -> list[str]:
    cleaned: list[str] = []
    for query in queries:
        stripped = _strip_leading_sql_comments(query)
        if stripped:
            stripped = _normalize_unicode_punctuation(stripped)
            cleaned.append(stripped)
    return cleaned


def _normalize_unicode_punctuation(sql: str) -> str:
    """Replace common fullwidth Unicode punctuation with ASCII equivalents.

    LLMs often emit fullwidth commas (U+FF0C) and semicolons (U+FF1B)
    when writing SQL with Chinese identifiers. SQL engines treat these
    as part of the identifier rather than as syntax separators, causing
    parse errors like "column 'A，B，C' not found".
    """
    sql = sql.replace("，", ",")  # fullwidth comma → ASCII comma
    sql = sql.replace("；", ";")  # fullwidth semicolon → ASCII semicolon
    return sql


def _quote_identifier(name: str) -> str:
    return quote_duckdb_identifier(name)


def create_probe_connection(
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    sql: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with task logical table views registered."""
    conn = duckdb.connect(":memory:")
    try:
        create_duckdb_views(
            conn,
            context_dir,
            catalog,
            logical_tables=iter_logical_tables(catalog),
            sql=sql,
        )
    except Exception:
        conn.close()
        raise
    return conn


def _asset_sql_replacements(catalog: dict[str, Any]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    view_names = sqlite_view_names(catalog)
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if not asset_path:
            continue
        if kind in ("csv", "json"):
            table_ref = _quote_identifier(asset_view_name(asset_path))
            if kind == "json" and json_schema_has_records_fields(schema):
                replacements.append((f"{asset_path}.records", table_ref))
            replacements.append((asset_path, table_ref))
            # Also match single-quoted asset paths (common model mistake)
            replacements.append((f"'{asset_path}'", table_ref))
            continue
        if kind == "sqlite":
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                view_name = view_names.get((asset_path, table_name))
                if table_name and view_name:
                    ref = f"{asset_path}.{table_name}"
                    qref = _quote_identifier(view_name)
                    replacements.append((ref, qref))
                    replacements.append((f"'{ref}'", qref))
    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def _normalize_probe_sql(catalog: dict[str, Any], sql: str) -> str:
    normalized = sql
    for source, target in _asset_sql_replacements(catalog):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
        normalized = re.sub(pattern, target, normalized)
    return normalized


def _column_expression(kind: str, column: str) -> str:
    if kind == "json" and column.startswith("records."):
        return ".".join(_quote_identifier(part) for part in column.split("."))
    return _quote_identifier(column)


def _make_json_safe(value: Any) -> Any:
    """Convert DuckDB-returned types to JSON-serializable values (recursive)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    return value


def _fetch_query_result(
    result: duckdb.DuckDBPyConnection,
    *,
    limit: int | None,
) -> dict[str, Any]:
    columns = [desc[0] for desc in result.description or []]
    if limit is None:
        rows = result.fetchall()
        truncated = False
        returned_rows = rows
    else:
        rows = result.fetchmany(limit + 1)
        truncated = len(rows) > limit
        returned_rows = rows[:limit]
    return {
        "ok": True,
        "columns": columns,
        "rows": [[_make_json_safe(cell) for cell in row] for row in returned_rows],
        "row_count": len(returned_rows),
        "truncated": truncated,
    }


def execute_probe_query(
    context_dir: Path,
    catalog: dict[str, Any],
    queries: list[str],
    *,
    limit: int | None = 200,
) -> dict[str, Any]:
    """Execute multiple read-only SQL queries against task data files.

    Creates in-memory DuckDB views for CSV/JSON/SQLite assets from the catalog
    once, then executes each query in *queries*. Only SELECT / WITH statements
    are allowed. Table names must match file-name stems for CSV/JSON assets, or
    SQLite table names when they do not collide with existing view names.

    Returns a list of per-query results under ``results`` so the caller can
    map each query to its output. A single failing query does not abort the
    batch.
    """
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        queries = _clean_probe_queries(queries)
        all_sql = " ".join(queries)
        conn = create_probe_connection(context_dir, catalog, sql=all_sql)
        results: list[dict[str, Any]] = []
        for sql in queries:
            try:
                _validate_read_only_sql(sql)
                normalized_sql = _normalize_probe_sql(catalog, sql)
                result = conn.execute(normalized_sql)
                payload = _fetch_query_result(result, limit=limit)
                if normalized_sql != sql:
                    payload["normalized_sql"] = normalized_sql
                results.append(payload)
            except Exception as exc:
                error_msg = str(exc)
                if "does not exist" in error_msg.lower() or "not exist" in error_msg.lower():
                    try:
                        views = conn.execute(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema='main' AND table_type='VIEW' "
                            "ORDER BY table_name"
                        ).fetchall()
                        view_names = [row[0] for row in views]
                        if view_names:
                            error_msg += (
                                "\nAvailable tables/views: " + ", ".join(view_names)
                            )
                    except Exception:
                        pass
                results.append({"ok": False, "error": error_msg, "sql": sql})
        top_level_ok = all(r.get("ok", True) for r in results)
        return {"ok": top_level_ok, "results": results, "query_count": len(results)}
    finally:
        if conn is not None:
            conn.close()


def get_column_distinct_values(
    context_dir: Path,
    catalog: dict[str, Any],
    table: str,
    column: str,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return the most frequent distinct values for a column.

    *table* is the file-name stem for CSV/JSON assets, or the SQLite
    table name for ``.db`` assets.
    """
    resolved = _resolve_table(catalog, table)
    if resolved is None:
        return {"ok": False, "error": f"Table '{table}' not found in catalog"}

    asset_path: str = resolved["asset_path"]
    kind: str = resolved["kind"]
    file_path = context_dir / asset_path

    if kind in ("csv", "json"):
        return _distinct_values_duckdb(context_dir, catalog, resolved, column, top_n)
    if kind == "sqlite":
        return _distinct_values_sqlite(file_path, resolved["sqlite_table"], column, top_n)
    return {"ok": False, "error": f"Unsupported asset kind: {kind}"}


def _resolve_table(
    catalog: dict[str, Any], table: str,
) -> dict[str, Any] | None:
    normalized_table = table.strip().strip('"').replace("\\", "/")
    for logical in iter_logical_tables(catalog):
        logical_name = str(logical.get("table", ""))
        if normalized_table == logical_name or normalized_table.lower() == logical_name.lower():
            if logical.get("source_kind") == "sqlite":
                return {
                    "asset_path": logical["source_asset_path"],
                    "kind": "sqlite",
                    "sqlite_table": logical["source_table"],
                }
            return {
                "asset_path": logical["source_asset_path"],
                "kind": logical["source_kind"],
            }
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        stem = Path(asset_path).stem

        if kind in ("csv", "json") and normalized_table in {stem, asset_path, f"{asset_path}.records"}:
            return {"asset_path": asset_path, "kind": kind}

        if kind == "sqlite":
            for t in schema.get("tables", []):
                if t.get("name") == normalized_table:
                    return {
                        "asset_path": asset_path,
                        "kind": "sqlite",
                        "sqlite_table": normalized_table,
                    }
    return None


def _distinct_values_duckdb(
    context_dir: Path, catalog: dict[str, Any], resolved: dict[str, Any], column: str, top_n: int,
) -> dict[str, Any]:
    conn = duckdb.connect(":memory:")
    try:
        create_duckdb_views(
            conn,
            context_dir,
            catalog,
            logical_tables=iter_logical_tables(catalog),
        )
        asset_path: str = resolved["asset_path"]
        kind: str = resolved["kind"]
        table_name = asset_view_name(asset_path)
        quoted_table = _quote_identifier(table_name)
        column_expr = _column_expression(kind, column)
        result = conn.execute(
            f"SELECT {column_expr} AS value, COUNT(*) AS cnt "
            f"FROM {quoted_table} "
            f"GROUP BY {column_expr} "
            f"ORDER BY cnt DESC "
            f"LIMIT ?",
            [top_n],
        )
        values = [{"value": _make_json_safe(row[0]), "count": row[1]} for row in result.fetchall()]
        return {"ok": True, "table": table_name, "column": column, "values": values, "value_count": len(values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


def _distinct_values_sqlite(
    file_path: Path, table: str, column: str, top_n: int,
) -> dict[str, Any]:
    try:
        with connect_sqlite_read_only(file_path) as conn:
            quoted_table = _quote_identifier(table)
            quoted_col = _quote_identifier(column)
            rows = conn.execute(
                f"SELECT {quoted_col} AS value, COUNT(*) AS cnt "
                f"FROM {quoted_table} "
                f"GROUP BY {quoted_col} "
                f"ORDER BY cnt DESC "
                f"LIMIT ?",
                (top_n,),
            ).fetchall()
            values = [{"value": _make_json_safe(row[0]), "count": row[1]} for row in rows]
            return {"ok": True, "table": table, "column": column, "values": values, "value_count": len(values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
